import pickle

from tensorflow.keras.callbacks import TensorBoard

from data_preprocessing import preprocess_data_hmdb
from models import make_model, prepare_model

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

    model_variation = "model2"  # Either {"model2", "model3"}
    resize = (112, 112)  # Make all images the same size

    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data_hmdb(model_variation=model_variation, resize=resize)

    print(f"{X_train.shape=}; {y_train.shape=}; {X_val.shape=}; {y_val.shape=}; {X_test.shape=}; {y_test.shape=}")

    total_size = X_train.shape[0]  # For the learning rate scheduler
    input_shape = X_train.shape[1:]  # For the model

    # Create the model
    model = make_model(kernel_size=kernel_size, pool_size=pool_size, pooling_type=pooling_type, dropout_value=dropout_value,
                       conv_act=conv_act, input_shape=input_shape, normalise=normalise, model_variation=model_variation)

    # Prepare the model
    model = prepare_model(model, learning_rate=learning_rate, batch_size=batch_size, total_size=total_size)
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
