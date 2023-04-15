import pickle
import numpy as np
import cv2
import os

from tensorflow.keras.callbacks import TensorBoard

from models import make_model, prepare_model



# Load data from pickle file
with open("data/Stanford40.pickle", "rb") as f:
    data = pickle.load(f)

# Load image data for each file path
train_data = []
for file_path in data["train_files"]:
    img = cv2.imread(f"./Stanford40/JPEGImages/{file_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (112, 112))
    train_data.append((resized_img, data["train_labels"][data["train_files"].index(file_path)]))

test_data = []
for file_path in data["test_files"]:
    img = cv2.imread(f"./Stanford40/JPEGImages/{file_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (112, 112))
    test_data.append((resized_img, data["test_labels"][data["test_files"].index(file_path)]))

val_data = []
for file_path in data["val_files"]:
    img = cv2.imread(f"./Stanford40/JPEGImages/{file_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (112, 112))
    val_data.append((resized_img, data["val_labels"][data["val_files"].index(file_path)]))

# Convert data to numpy arrays
train_images = np.array([i[0] for i in train_data])
train_labels = np.array([i[1] for i in train_data])

test_images = np.array([i[0] for i in test_data])
test_labels = np.array([i[1] for i in test_data])

val_images = np.array([i[0] for i in val_data])
val_labels = np.array([i[1] for i in val_data])

# Split data into features and labels
x_train = train_images
y_train = train_labels

x_test = test_images
y_test = test_labels

x_val = val_images
y_val = val_labels


if __name__ == "__main__":
    # Load the data from Stanford40.pickle


    print(f"{x_train.shape=}; {y_train.shape=}; {x_val.shape=}; {y_val.shape=}; {x_test.shape=}; {y_test.shape=}")
    
    kernel_sizes = (7, 5, 3)
    pool_sizes = (2, 2, 2)
    filter_sizes = (32, 64, 128)
    pooling_type = "avg"
    dropout_value = None
    conv_act = "relu"
    normalise = False

    learning_rate = 0.01
    batch_size = 64
    total_size = x_train.shape[0]
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
    model = make_model(kernel_sizes=kernel_sizes, pool_sizes=pool_sizes, pooling_type=pooling_type,
                       dropout_value=dropout_value, conv_act=conv_act)

    # Prepare the model
    model = prepare_model(model, learning_rate=learning_rate, batch_size=batch_size, total_size=total_size)

    tensorboard_callback = TensorBoard(log_dir=f"./logs/{model_variation}")
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(
        x_val, y_val), batch_size=batch_size, callbacks=[tensorboard_callback])
    model.save(f"./models/model_{model_variation}.h5")

    # Save the history
    with open(f"./history/history_{model_variation}.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")