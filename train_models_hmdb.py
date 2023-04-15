import os
import pickle

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model

from data_preprocessing import preprocess_data_hmdb
from models import DecayingLRSchedule, make_model, prepare_model

if __name__ == "__main__":
    batch_size = 64
    epochs = 15

    model_variation = "model3"  # Either {"model2", "model3"}
    resize = (112, 112)  # Make all images the same size

    # If data is not already saved, preprocess it and save it to disk
    if not os.path.exists(f"./data/{model_variation}_{resize[0]}.pickle"):
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data_hmdb(model_variation=model_variation, resize=resize)

        # Save the data to disk
        with open(f"./data/{model_variation}_{resize[0]}.pickle", "wb") as filter:
            pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), filter)
    else:
        # Load the data from disk
        with open(f"./data/{model_variation}_{resize[0]}.pickle", "rb") as filter:
            X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(filter)

    print(f"{X_train.shape=}; {y_train.shape=}; {X_val.shape=}; {y_val.shape=}; {X_test.shape=}; {y_test.shape=}")

    if model_variation == "model2":
        # Use the pretrained model from model1 for colour images
        pretrained_model_fp = "./models/model1.h5"  # Path to the pretrained model
        model = load_model(pretrained_model_fp, custom_objects={"DecayingLRScheduler": DecayingLRSchedule})
    elif model_variation == "model3":  # Start from scratch, optical flow images
        kernel_sizes = (7, 5, 3)
        pool_sizes = (2, 2, 2)
        filter_sizes = (32, 64, 128)
        dense_sizes = (1024, 512)
        pooling_type = "max"
        dropout_value = 0.5
        conv_act = "relu"
        normalise = False

        learning_rate = 0.001
        lr_schedule = "decay"
        opt = "adam"

        total_size = X_train.shape[0]  # For the learning rate scheduler
        input_shape = X_train.shape[1:]  # For the model

        model = make_model(kernel_sizes=kernel_sizes, pool_sizes=pool_sizes, filter_sizes=filter_sizes,
                           dense_sizes=dense_sizes, pooling_type=pooling_type, dropout_value=dropout_value,
                           conv_act=conv_act, normalise=normalise, input_shape=input_shape, model_variation=model_variation)
        model = prepare_model(model, learning_rate=learning_rate, batch_size=batch_size, total_size=total_size, opt=opt,
                              lr_schedule=lr_schedule)

    model.summary()

    tensorboard_callback = TensorBoard(log_dir=f"./logs/{model_variation}")
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                        batch_size=batch_size, callbacks=[tensorboard_callback])
    model.save(f"./models/{model_variation}.h5")

    # Save the history
    with open(f"./history/{model_variation}.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss:.2f}; Accuracy: {accuracy:.2f}")
