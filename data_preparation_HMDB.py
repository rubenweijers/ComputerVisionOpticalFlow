import glob
import os
import pickle
from collections import Counter
from pathlib import Path

import patoolib
import requests
from sklearn.model_selection import train_test_split

DOWNLOAD = False  # Set to True to download the data
EXTRACT = False  # Set to True to extract the data
VAL_SIZE = 0.1  # Proportion of the training set to use for validation
RANDOM_STATE = 0
pickle_location = "./data/hmdb.pickle"  # Location to save the data to

keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
               "run", "shoot_bow", "smoke", "throw", "wave"]


# Use requests to download the data
if DOWNLOAD:
    # Make dirs
    for dir in "video_data", "test_train_splits":
        os.makedirs(dir, exist_ok=True)

    videos = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    splits = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar"

    for url in (videos, splits):
        path = "./data/" + url.split("/")[-1]

        r = requests.get(url)
        with open(path, "wb") as f:  # Save rars to disk
            f.write(r.content)

if EXTRACT:  # Extract the rar files
    patoolib.extract_archive("./data/hmdb51_org.rar", outdir="video_data")
    patoolib.extract_archive("./data/test_train_splits.rar", outdir="test_train_splits")

    for files in os.listdir("video_data"):
        foldername = files.split(".")[0]
        if foldername in keep_hmdb51:  # Extract only the relevant classes for the assignment.
            Path(f"video_data/{foldername}").mkdir(parents=True, exist_ok=True)
            patoolib.extract_archive(f"video_data/{files}", outdir=f"video_data/{foldername}")

TRAIN_TAG, TEST_TAG = 1, 2
train_files, test_files = [], []
train_labels, test_labels = [], []
split_pattern_name = f"*test_split1.txt"
split_pattern_path = os.path.join("test_train_splits", "testTrainMulti_7030_splits", split_pattern_name)
annotation_paths = glob.glob(split_pattern_path)
print(f"Found {len(annotation_paths)} annotation files.")

for filepath in annotation_paths:
    class_name = "_".join(filepath.split(os.sep)[-1].split("_")[:-2])  # os.sep is the path separator for the current OS.

    if class_name not in keep_hmdb51:
        continue  # skipping the classes that we won't use.

    with open(filepath) as fid:
        lines = fid.readlines()

    for line in lines:
        video_filename, tag_string = line.split()
        tag = int(tag_string)

        if tag == TRAIN_TAG:
            train_files.append(video_filename)
            train_labels.append(class_name)

        elif tag == TEST_TAG:
            test_files.append(video_filename)
            test_labels.append(class_name)

# Split the train data into train and validation sets.
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels,
                                                                    test_size=VAL_SIZE, random_state=RANDOM_STATE,
                                                                    stratify=train_labels)

print(f"Train files ({len(train_files)})")
print(f"Train Distribution:{list(Counter(sorted(train_labels)).items())}")

print(f"Validation files ({len(val_files)})")
print(f"Validation Distribution:{list(Counter(sorted(val_labels)).items())}")

print(f"Test files ({len(test_files)})")
print(f"Test Distribution:{list(Counter(sorted(test_labels)).items())}")

action_categories = sorted(list(set(train_labels)))
print(f"Action categories ({len(action_categories)}):\n{action_categories}")

data = {"train_files": train_files, "train_labels": train_labels,
        "val_files": val_files, "val_labels": val_labels,
        "test_files": test_files, "test_labels": test_labels,
        "action_categories": action_categories}

with open(pickle_location, "wb") as f:
    pickle.dump(data, f)
