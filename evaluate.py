import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow_addons.optimizers import CyclicalLearningRate

from data_preprocessing import preprocess_data_hmdb
from models import DecayingLRSchedule, scale_fn


def plot_history(history: dict, model_variation: str) -> None:
    """Plot the history of a model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    ax1.plot(history["loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(np.arange(0, len(history["loss"]), 1))
    ax1.set_xticklabels(np.arange(1, len(history["loss"]) + 1, 1))
    ax1.legend()

    ax2.plot(history["accuracy"], label="Training Accuracy")
    ax2.plot(history["val_accuracy"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticks(np.arange(0, len(history["accuracy"]), 1))
    ax2.set_xticklabels(np.arange(1, len(history["accuracy"]) + 1, 1))
    ax2.legend()

    plt.suptitle(f"Model: {model_variation}")

    ax1.yaxis.grid(True)  # Gridlines, only in the horizontal direction
    ax2.yaxis.grid(True)

    ax1.set_ylim(0, None)  # Loss has no upper limit
    ax2.set_ylim(0, 1)

    # plt.show()

    fig.savefig(f"./img/plot_{model_variation}.png")


def plot_confusion_matrix(y_test, y_pred, variation: str):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))

    ax = plt.gca().set_aspect("equal")  # Set aspect ratio to square
    sns.heatmap(cm, annot=True, fmt="d")

    plt.title(f"Confusion matrix for model: {variation}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # Use classes as labels
    from data_preparation_HMDB import keep_hmdb51
    plt.xticks(np.arange(0.5, 12.5, 1), keep_hmdb51, rotation=90)
    plt.savefig(f"./img/cm_{variation}.png")
    # plt.show()


if __name__ == "__main__":
    scores_train = {}
    scores_val = {}
    scores_test = {}

    resize = (112, 112)

    for model_variation in ("model1", "model1_cyclic", "model1_augmented", "model2", "model3", "model4"):  # TODO: add model1
        if model_variation == "model1_augmented":
            fp = f"./data/model1_{resize[0]}.pickle"  # Load the original model1 data, to make results comparable
        else:
            fp = f"./data/{model_variation}_{resize[0]}.pickle"

        # If data is not already saved, preprocess it and save it to disk
        if not os.path.exists(fp):
            if model_variation in ("model1", "model1_augmented"):
                raise ValueError("Run the train_models.py script first")  # Preprocessing of model1 is done in train_models.py

            elif model_variation in ("model2", "model3", "model4"):
                X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data_hmdb(model_variation=model_variation, resize=resize)
            else:
                raise ValueError("Invalid model variation")

            # Save the data to disk
            with open(fp, "wb") as f:
                pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)
        else:
            # Load the data from disk
            with open(fp, "rb") as f:
                X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

        # Load model
        model = load_model(f"./models/{model_variation}.h5", custom_objects={"DecayingLRSchedule": DecayingLRSchedule,
                                                                             "CyclicalLearningRate": CyclicalLearningRate,
                                                                             "scale_fn": scale_fn})

        # Graph model
        plot_model(model, to_file=f"./img/summary_{model_variation}.png", show_shapes=True, show_layer_names=True)

        # Print model summary
        model.summary()

        # Evaluate model on training set
        loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=0)
        print(f"Model: {model_variation} - Training loss: {loss_train:.4f} - Training accuracy: {accuracy_train:.4f}")

        # Evaluate model on validation set
        loss_val, accuracy_val = model.evaluate(X_val, y_val, verbose=0)
        print(f"Model: {model_variation} - Validation loss: {loss_val:.4f} - Validation accuracy: {accuracy_val:.4f}")

        # Evaluate model on test set
        loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=0)
        print(f"Model: {model_variation} - Test loss: {loss_test:.4f} - Test accuracy: {accuracy_test:.4f}")

        # Add to dictionary
        scores_train[model_variation] = accuracy_train
        scores_val[model_variation] = accuracy_val
        scores_test[model_variation] = accuracy_test

        # Load history
        with open(f"./history/{model_variation}.pkl", "rb") as f:
            history = pickle.load(f)

        # Plot history
        plot_history(history, model_variation)

        # Plot confusion matrix
        y_pred = np.argmax(model.predict(X_test), axis=1)
        plot_confusion_matrix(np.argmax(y_test, axis=1), y_pred, model_variation)

    # Sort by accuracy
    scores_train_sorted = sorted(scores_train.items(), key=lambda x: x[1], reverse=True)
    scores_val_sorted = sorted(scores_val.items(), key=lambda x: x[1], reverse=True)
    scores_test_sorted = sorted(scores_test.items(), key=lambda x: x[1], reverse=True)

    # Print sorted scores, round to 4 decimals
    print([f"{variation}: {round(accuracy, 4)}" for variation, accuracy in scores_train_sorted])
    print([f"{variation}: {round(accuracy, 4)}" for variation, accuracy in scores_val_sorted])
    print([f"{variation}: {round(accuracy, 4)}" for variation, accuracy in scores_test_sorted])
