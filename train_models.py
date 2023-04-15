import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
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


def encode_labels(y_train, y_val, y_test):
    label_map = {label: idx for idx, label in enumerate(np.unique(y_train))}
    y_train = np.array([label_map[label] for label in y_train])
    y_val = np.array([label_map[label] for label in y_val])
    y_test = np.array([label_map[label] for label in y_test])

    return (to_categorical(y_train), to_categorical(y_val), to_categorical(y_test))


y_train, y_val, y_test = encode_labels(y_train, y_val, y_test)

# Create image data generator with augmentation options
train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Fit the data generator to the training data
train_datagen.fit(x_train)

# Use the generator to create augmented image batches
train_generator = train_datagen.flow(x_train, y_train, batch_size=64)

# Define a function to plot the train and validation accuracy and loss graphs
def plot_acc_loss(history):
    # Plot the train and validation accuracy
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot the train and validation loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load the data from Stanford40.pickle

    print(f"{x_train.shape=}; {y_train.shape=}; {x_val.shape=}; {y_val.shape=}; {x_test.shape=}; {y_test.shape=}")

    kernel_sizes = (7, 5, 3)
    pool_sizes = (2, 2, 2)
    filter_sizes = (32, 64, 128)
    dense_sizes = (1024, 512)
    pooling_type = "max"
    dropout_value = 0.5
    conv_act = "relu"
    normalise = False

    learning_rate = 0.001
    batch_size = 64
    total_size = x_train.shape[0]
    epochs = 15
    lr_schedule = "decay"
    opt = "adam"

    model_variation = "model1_augmented"

    # Create the model
    model = make_model(kernel_sizes=kernel_sizes, pool_sizes=pool_sizes, pooling_type=pooling_type,
                       dense_sizes=dense_sizes, filter_sizes=filter_sizes, normalise=normalise,
                       dropout_value=dropout_value, conv_act=conv_act)

    # Prepare the model
    model = prepare_model(model, learning_rate=learning_rate, batch_size=batch_size, total_size=total_size,
                          lr_schedule=lr_schedule, opt=opt)

    tensorboard_callback = TensorBoard(log_dir=f"./logs/{model_variation}")

    history = model.fit(train_generator,
                    steps_per_epoch=len(x_train)//batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    callbacks=[tensorboard_callback])

    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val),
                        batch_size=batch_size, callbacks=[tensorboard_callback])

    model.save(f"./models/{model_variation}.h5")

    # # Save the history
    with open(f"./history/{model_variation}.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")
    nparams = model.count_params()
    print(nparams)

    # Plot the accuracy and loss graphs
    plot_acc_loss(history)
