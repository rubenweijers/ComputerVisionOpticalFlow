import os
import pickle

import cv2
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tqdm.contrib import tzip

from models import make_model, prepare_model


def load_data(filepath: str = "data/hmdb.pickle"):
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    X_train = np.array(data["train_files"])
    X_val = np.array(data["val_files"])
    X_test = np.array(data["test_files"])

    y_train = np.array(data["train_labels"])
    y_val = np.array(data["val_labels"])
    y_test = np.array(data["test_labels"])

    action_categories = data["action_categories"]
    mapping = {i: label for i, label in enumerate(action_categories)}
    print(f"Action categories ({len(action_categories)}):\n{action_categories}")

    # Map the labels to integers based on index in list
    # Action categories are in same order as Stanford40 dataset
    y_train = np.array([action_categories.index(label) for label in y_train])
    y_val = np.array([action_categories.index(label) for label in y_val])
    y_test = np.array([action_categories.index(label) for label in y_test])

    # Convert the labels to categorical
    y_train = to_categorical(y_train, num_classes=len(action_categories))
    y_val = to_categorical(y_val, num_classes=len(action_categories))
    y_test = to_categorical(y_test, num_classes=len(action_categories))

    # Create a mapping from integer to label
    return X_train, y_train, X_val, y_val, X_test, y_test, mapping


def load_video(filename: str, label: str, mapping: dict, frame_n: int = "halfway", greyscale: bool = True, resize=(112, 112)):
    """Load a video from a filepath and return two consecutive frames.

    Set frame_n to "halfway" to get the middle frame, or an integer to get a specific frame.
    Set resize to None to not resize the image, or a tuple to resize the image.
    """
    label_str = mapping[np.argmax(label)]  # Get the label string from the mapping
    fp = os.path.join("./video_data", label_str, label_str, filename)  # Class label twice due to folder structure
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
    X_train, y_train, X_val, y_val, X_test, y_test, mapping = load_data("data/hmdb.pickle")
    print(f"{X_train.shape=}; {y_train.shape=}; {X_val.shape=}; {y_val.shape=}; {X_test.shape=}; {y_test.shape=}")

    model_variation = "model2"  # Either {"model2", "model3"}

    if model_variation == "model2":  # Do not use optical flow, but use the middle frame
        # Convert filepaths to images
        X_train = [load_video(filepath, label, mapping, greyscale=False)
                   for filepath, label in tzip(X_train, y_train, desc="Loading X_train data")]
        X_val = [load_video(filepath, label, mapping, greyscale=False)
                 for filepath, label in tzip(X_val, y_val, desc="Loading X_val data")]
        # X_test = [load_video(filepath, label, mapping, greyscale=False)
        #           for filepath, label in tzip(X_test, y_test, desc="Loading X_test data")]

        X_train = np.array([imgs[0] for imgs in X_train])  # Only use one frame
        X_val = np.array([imgs[0] for imgs in X_val])
        # X_test = np.array([imgs[0] for imgs in X_test])

    elif model_variation == "model3":  # Use optical flow
        # Convert filepaths to optical flow images
        X_train = [deepflow(*load_video(filepath, label, greyscale=True))
                   for filepath, label in tzip(X_train[:10], y_train[:10], desc="Loading X_train data")]
        # X_val = [deepflow(*load_video(filepath, label, greyscale=True))
        #          for filepath, label in tzip(X_val, y_val, desc="Loading X_val data")]
        # X_test = [deepflow(*load_video(filepath, label, greyscale=True))
        #           for filepath, label in tzip(X_test, y_test, desc="Loading X_test data")]

        X_train = np.array(X_train)
        # X_val = np.array(X_val)
        # X_test = np.array(X_test)

    kernel_size = 3
    pool_size = 2
    pooling_type = "avg"
    dropout_value = None
    conv_act = "relu"
    normalise = False

    learning_rate = 0.01
    batch_size = 64
    total_size = X_train.shape[0]
    epochs = 15

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
