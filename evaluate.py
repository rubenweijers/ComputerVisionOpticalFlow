import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from data_preprocessing import preprocess_data_hmdb
from models import DecayingLRSchedule


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


if __name__ == "__main__":
    scores_train = {}
    scores_val = {}
    scores_test = {}

    resize = (112, 112)

    for model_variation in ("model2", "model3", "model4"):
        # If data is not already saved, preprocess it and save it to disk
        if not os.path.exists(f"./data/{model_variation}_{resize[0]}.pickle"):
            X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data_hmdb(model_variation=model_variation, resize=resize)

            # Save the data to disk
            with open(f"./data/{model_variation}_{resize[0]}.pickle", "wb") as f:
                pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)
        else:
            # Load the data from disk
            with open(f"./data/{model_variation}_{resize[0]}.pickle", "rb") as f:
                X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

        # Load model
        model = load_model(f"./models/{model_variation}.h5", custom_objects={"DecayingLRSchedule": DecayingLRSchedule})

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

    # Sort by accuracy
    scores_train_sorted = sorted(scores_train.items(), key=lambda x: x[1], reverse=True)
    scores_val_sorted = sorted(scores_val.items(), key=lambda x: x[1], reverse=True)
    scores_test_sorted = sorted(scores_test.items(), key=lambda x: x[1], reverse=True)

    # Print sorted scores, round to 4 decimals
    print([f"{variation}: {round(accuracy, 4)}" for variation, accuracy in scores_train_sorted])
    print([f"{variation}: {round(accuracy, 4)}" for variation, accuracy in scores_val_sorted])
    print([f"{variation}: {round(accuracy, 4)}" for variation, accuracy in scores_test_sorted])
