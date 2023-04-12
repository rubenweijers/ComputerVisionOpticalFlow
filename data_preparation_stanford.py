import io
import pickle
import zipfile
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split

DOWNLOAD = False  # Set to True to download the data
IMAGE_N = 234  # change this to a number between [0, 1200] and you can see a different training image, None for no image
TEST_SIZE = 0.1  # Proportion of all data to use for testing
VAL_SIZE = 0.1  # Proportion of the training set to use for validation
RANDOM_STATE = 0
pickle_location = "./data/Stanford40.pickle"  # Location to save the data to

keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid", "riding_a_bike", "riding_a_horse",
                   "running", "shooting_an_arrow", "smoking", "throwing_frisby", "waving_hands"]

if DOWNLOAD:  # Download the data if it is not already downloaded
    jpeg = "http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip"
    splits = "http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip"

    # Download the file from the url and save it locally under Stanford40/
    for url in (jpeg, splits):
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("Stanford40")


with open("./Stanford40/ImageSplits/train.txt", "r") as f:
    # We won't use these splits but split them ourselves
    train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if "_".join(file_name.split("_")[:-1]) in keep_stanford40]
    train_labels = ["_".join(name.split("_")[:-1]) for name in train_files]

with open("Stanford40/ImageSplits/test.txt", "r") as f:
    # We won't use these splits but split them ourselves
    test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if "_".join(file_name.split("_")[:-1]) in keep_stanford40]
    test_labels = ["_".join(name.split("_")[:-1]) for name in test_files]

# Combine the splits and split for keeping more images in the training set than the test set.
all_files = train_files + test_files
all_labels = train_labels + test_labels
remainder_files, test_files = train_test_split(all_files, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=all_labels)

remainder_labels = ["_".join(name.split("_")[:-1]) for name in remainder_files]  # Get the labels for the remainder
train_files, val_files = train_test_split(remainder_files, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=remainder_labels)

train_labels = ["_".join(name.split("_")[:-1]) for name in train_files]
test_labels = ["_".join(name.split("_")[:-1]) for name in test_files]
val_labels = ["_".join(name.split("_")[:-1]) for name in val_files]

print(f"Train files ({len(train_files)})")
print(f"Train Distribution:{list(Counter(sorted(train_labels)).items())}\n")

print(f"Validation files ({len(val_files)})")
print(f"Validation Distribution:{list(Counter(sorted(val_labels)).items())}\n")

print(f"Test files ({len(test_files)})")
print(f"Test Distribution:{list(Counter(sorted(test_labels)).items())}\n")

action_categories = sorted(list(set(train_labels)))
print(f"Action categories ({len(action_categories)}):\n{action_categories}")

# Save data to pickle file
data = {"train_files": train_files, "train_labels": train_labels,
        "test_files": test_files, "test_labels": test_labels,
        "val_files": val_files, "val_labels": val_labels,
        "action_categories": action_categories}

with open(pickle_location, "wb") as f:
    pickle.dump(data, f)

if IMAGE_N is not None:  # Show an image if IMAGE_N is not None
    img = cv2.imread(f"./Stanford40/JPEGImages/{train_files[IMAGE_N]}")
    print(f"An image with the label: {train_labels[IMAGE_N]}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(img)  # Show image with matplotlib
    plt.show()
