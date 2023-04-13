import os
import pickle

import cv2
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tqdm.contrib import tzip

from models import make_model, prepare_model


def load_data(filepath: str = "data/hmdb.pickle"):
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    X_train = np.array(data["train_files"])
    y_train = np.array(data["train_labels"])
    X_val = np.array(data["val_files"])
    y_val = np.array(data["val_labels"])
    X_test = np.array(data["test_files"])
    y_test = np.array(data["test_labels"])

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_video(filename: str, label: str, frame_n: int = "halfway", greyscale: bool = True, resize=(112, 112)):
    """Load a video from a filepath and return two consecutive frames.

    Set frame_n to "halfway" to get the middle frame, or an integer to get a specific frame.
    Set resize to None to not resize the image, or a tuple to resize the image.
    """
    fp = os.path.join("./video_data", label, label, filename)  # Class label twice due to folder structure
    cap = cv2.VideoCapture(fp)

    if frame_n == "halfway":  # Exception for the middle frame
        frame_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2)  # Get the middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)  # Set the frame number

    imgs = []
    for _ in range(2):  # Get the next two frames
        ret, img = cap.read()  # Only read the frame we want

        if ret:  # If the frame was read successfully
            if greyscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Greyscale the image
            if resize is not None:
                img = cv2.resize(img, resize)  # Resize the image
            imgs.append(img)

    cap.release()

    if len(imgs) != 2:
        raise ValueError(f"Could not load two frames from: {filename}")

    return imgs


def deepflow(img1, img2):
    """Calculate the optical flow between two images using the DeepFlow algorithm.
    Based on: https://gist.github.com/FingerRec/eba088d6d7a50e17c875d74684ec2849
    """
    deepflow = cv2.optflow.createOptFlow_DeepFlow()
    result = deepflow.calc(img1, img2, None)

    return result


if __name__ == "__main__":
    # Load the data from the HMDB dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("data/hmdb.pickle")
    print(f"{X_train.shape=}; {y_train.shape=}; {X_val.shape=}; {y_val.shape=}; {X_test.shape=}; {y_test.shape=}")

    # Convert filepaths to images
    X_train = [load_video(filepath, label) for filepath, label in tzip(X_train[:10], y_train[:10], desc="Loading X_train data")]
    # X_val = [load_video(filepath, label) for filepath, label in tzip(X_val, y_val, desc="Loading X_val data")]
    # X_test = [load_video(filepath, label) for filepath, label in tzip(X_test, y_test, desc="Loading X_test data")]

    print(f"{X_train[0][0].shape=}; {X_train[0][1].shape=}")

    result = deepflow(X_train[0][0], X_train[0][1])
    print(f"{result=}")
    print(f"{result.shape=}")

    exit()

    kernel_size = 3
    pool_size = 2
    pooling_type = "max"
    dropout_value = None
    conv_act = "relu"
    normalise = False

    learning_rate = 0.01
    batch_size = 64
    total_size = X_train.shape[0]
    epochs = 15

    model_variation = "averagejoe"  # Choose from: {baseline, nike, collegedropout, normaliser2000, averagejoe}

    if model_variation == "baseline":
        pass  # Default values are already set
    elif model_variation == "nike":
        conv_act = "swish"
    elif model_variation == "collegedropout":
        dropout_value = 0.5
    elif model_variation == "normaliser2000":
        normalise = True
    elif model_variation == "averagejoe":
        pooling_type = "avg"
    else:
        raise ValueError(f"Invalid model variation: {model_variation}")

    # Create the model
    model = make_model(kernel_size=kernel_size, pool_size=pool_size, pooling_type=pooling_type,
                       dropout_value=dropout_value, conv_act=conv_act)

    # Prepare the model
    model = prepare_model(model, learning_rate=learning_rate, batch_size=batch_size, total_size=total_size)

    tensorboard_callback = TensorBoard(log_dir=f"./logs/{model_variation}")
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(
        X_val, y_val), batch_size=batch_size, callbacks=[tensorboard_callback])
    model.save(f"./models/model_{model_variation}.h5")

    # Save the history
    with open(f"./history/history_{model_variation}.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")
