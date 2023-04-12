import cv2
from sklearn.model_selection import train_test_split
from collections import Counter
import requests
import zipfile
import io
import matplotlib.pyplot as plt

DOWNLOAD = False  # Set to True to download the data
IMAGE_N = 234  # change this to a number between [0, 1200] and you can see a different training image
TEST_SIZE = 0.1
RANDOM_STATE = 0

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
train_files, test_files = train_test_split(all_files, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=all_labels)
train_labels = ["_".join(name.split("_")[:-1]) for name in train_files]
test_labels = ["_".join(name.split("_")[:-1]) for name in test_files]

print(f"Train files ({len(train_files)}):\n\t{train_files}")
print(f"Train labels ({len(train_labels)}):\n\t{train_labels}\n"
      f"Train Distribution:{list(Counter(sorted(train_labels)).items())}\n")

print(f"Test files ({len(test_files)}):\n\t{test_files}")
print(f"Test labels ({len(test_labels)}):\n\t{test_labels}\n"
      f"Test Distribution:{list(Counter(sorted(test_labels)).items())}\n")

action_categories = sorted(list(set(train_labels)))
print(f"Action categories ({len(action_categories)}):\n{action_categories}")

img = cv2.imread(f"./Stanford40/JPEGImages/{train_files[IMAGE_N]}")
print(f"An image with the label - {train_labels[IMAGE_N]}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
plt.imshow(img)  # Show image with matplotlib
plt.show()
