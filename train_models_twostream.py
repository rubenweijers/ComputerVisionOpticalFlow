import os
import pickle

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from data_preprocessing import preprocess_data_hmdb
from models import DecayingLRSchedule, combine_models, prepare_model

if __name__ == "__main__":
    kernel_size = 3
    pool_size = 2
    pooling_type = "avg"
    dropout_value = None
    conv_act = "relu"
    normalise = False

    learning_rate = 0.01
    batch_size = 64
    epochs = 15
    lr_schedule = "decay"
    opt = "adam"

    model_variation = "model4"  # Fusion model
    resize = (112, 112)  # Make all images the same size

    # Load the data from the HMDB dataset
    if not os.path.exists(f"./data/{model_variation}_{resize[0]}.pickle"):
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data_hmdb(model_variation=model_variation, resize=resize)

        # Save the data to disk
        with open(f"./data/{model_variation}_{resize[0]}.pickle", "wb") as f:
            pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)
    else:
        # Load the data from disk
        with open(f"./data/{model_variation}_{resize[0]}.pickle", "rb") as f:
            X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

    print(f"{X_train[0].shape=}; {X_train[1].shape=}; {y_train.shape=};\n"
          f"{X_val[0].shape=}; {X_val[1].shape=}; {y_val.shape=};\n"
          f"{X_test[0].shape=}; {X_test[1].shape=}; {y_test.shape=};")

    total_size = X_train[0].shape[0]  # For the learning rate scheduler

    # Load trained models from file
    model_frames = load_model(f"./models/model2.h5", custom_objects={"DecayingLRSchedule": DecayingLRSchedule})
    model_deepflow = load_model(f"./models/model3.h5", custom_objects={"DecayingLRSchedule": DecayingLRSchedule})

    # Combine the models
    model = combine_models(model_frames, model_deepflow)

    # Graph the model
    plot_model(model, to_file=f"./models/twostream.png", show_shapes=True, show_layer_names=True)

    # Prepare the model
    model = prepare_model(model, learning_rate=learning_rate, batch_size=batch_size, total_size=total_size,
                          lr_schedule=lr_schedule, opt=opt)

    model.summary()

    tensorboard_callback = TensorBoard(log_dir=f"./logs/{model_variation}")
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                        batch_size=batch_size, callbacks=[tensorboard_callback])
    model.save(f"./models/{model_variation}.h5")

    # # Save the history
    with open(f"./history/{model_variation}.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")
