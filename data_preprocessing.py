import os
import pickle

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from tqdm.contrib import tzip


def load_data_hmdb(filepath: str = "data/hmdb.pickle"):
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

        # Get frame at three/fourth of the way through the video
        # This way, more motion is visible in the optical flow
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n + int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 4))

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


def preprocess_model2(X_train, X_val, X_test):
    """Preprocess the data for model2, which does not use optical flow, but uses the middle frame."""
    X_train = np.array([imgs[0] for imgs in X_train])  # Only use one frame
    X_val = np.array([imgs[0] for imgs in X_val])
    X_test = np.array([imgs[0] for imgs in X_test])

    return X_train, X_val, X_test


def preprocess_model3(X_train, X_val, X_test):
    """Preprocess the data for model3, which uses optical flow."""
    X_train = np.array([deepflow(imgs[0], imgs[1]) for imgs in tqdm(X_train, desc="Calculating X_train optical flow")])
    X_val = np.array([deepflow(imgs[0], imgs[1]) for imgs in tqdm(X_val, desc="Calculating X_val optical flow")])
    X_test = np.array([deepflow(imgs[0], imgs[1]) for imgs in tqdm(X_test, desc="Calculating X_test optical flow")])

    return X_train, X_val, X_test


def filepath2videos(X_train, X_val, X_test, y_train, y_val, y_test, mapping, greyscale: bool = True, resize=(112, 112)):
    X_train = [load_video(filepath, label, mapping, greyscale=greyscale, resize=resize)
               for filepath, label in tzip(X_train, y_train, desc="Loading X_train data")]
    X_val = [load_video(filepath, label, mapping, greyscale=greyscale, resize=resize)
             for filepath, label in tzip(X_val, y_val, desc="Loading X_val data")]
    X_test = [load_video(filepath, label, mapping, greyscale=greyscale, resize=resize)
              for filepath, label in tzip(X_test, y_test, desc="Loading X_test data")]

    return X_train, X_val, X_test


def preprocess_data_hmdb(model_variation: str = "model2", pickle_location: str = "data/hmdb.pickle", resize=(112, 112)):
    # Load the data from the HMDB dataset
    X_train, y_train, X_val, y_val, X_test, y_test, mapping = load_data_hmdb(pickle_location)

    if model_variation in ("model2", "model3"):  # Load data for models 2 and 3, model 4 requires different preprocessing
        # Use greyscale if using optical flow, otherwise use RGB
        greyscale = False if model_variation == "model2" else True

        X_train, X_val, X_test = filepath2videos(X_train, X_val, X_test, y_train, y_val, y_test, mapping,
                                                 greyscale=greyscale, resize=resize)

    if model_variation == "model2":  # Do not use optical flow, but use the middle frame
        X_train, X_val, X_test = preprocess_model2(X_train, X_val, X_test)

    elif model_variation == "model3":  # Use optical flow
        X_train, X_val, X_test = preprocess_model3(X_train, X_val, X_test)

    elif model_variation == "model4":  # Use optical flow and the middle frame
        # We need both the RGB and greyscale images for model 4
        X_train_frames, X_val_frames, X_test_frames = filepath2videos(X_train, X_val, X_test, y_train, y_val, y_test, mapping,
                                                                      greyscale=False, resize=resize)

        # Greyscale data for optical flow, instead of calling filepath2videos again
        X_train_flow = [(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY))
                        for imgs in tqdm(X_train_frames, desc="Converting X_train to greyscale")]
        X_val_flow = [(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY))
                      for imgs in tqdm(X_val_frames, desc="Converting X_val to greyscale")]
        X_test_flow = [(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY))
                       for imgs in tqdm(X_test_frames, desc="Converting X_test to greyscale")]

        X_train_frames, X_val_frames, X_test_frames = preprocess_model2(X_train_frames, X_val_frames, X_test_frames)
        X_train_flow, X_val_flow, X_test_flow = preprocess_model3(X_train_flow, X_val_flow, X_test_flow)

        X_train = [X_train_frames, X_train_flow]
        X_val = [X_val_frames, X_val_flow]
        X_test = [X_test_frames, X_test_flow]

    else:
        raise ValueError(f"Invalid model variation: {model_variation}")

    return X_train, y_train, X_val, y_val, X_test, y_test
