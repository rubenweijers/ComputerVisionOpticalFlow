import tensorflow as tf
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization,
                                     Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow_addons.optimizers import CyclicalLearningRate


def make_model(kernel_sizes=(7, 5, 3), pool_sizes=(2, 2, 2), filter_sizes=(64, 32, 16), dense_sizes=(128, 64), pooling_type: str = "avg",
               dropout_value: float = None, conv_act: str = "relu", normalise: bool = False, input_shape: tuple = (112, 112, 3),
               model_variation: str = "model2") -> Sequential:
    """The model used for the experiments."""

    if pooling_type == "max":
        pooling = MaxPooling2D
    elif pooling_type == "avg":
        pooling = AveragePooling2D
    elif pooling_type is None:
        ...
    else:
        raise ValueError("Pooling must be either 'max' or 'average'.")

    if conv_act == "relu":
        conv_act = "relu"
    elif conv_act == "swish":
        conv_act = tf.nn.swish
    else:
        raise ValueError("conv_act must be either 'relu' or 'swish'.")

    # Input shape defined here is (112, 112, 3) because the images are resized to 112x112
    model = Sequential()

    # Make conv blocks
    for layer_n, (kernel, pool, filter) in enumerate(zip(kernel_sizes, pool_sizes, filter_sizes), start=1):
        model.add(Conv2D(filter, kernel, activation=conv_act, input_shape=input_shape, name=f"{model_variation}_conv2d_{layer_n}"))

        if normalise:
            model.add(BatchNormalization(name=f"{model_variation}_batchnorm_{layer_n}"))
        if pooling_type is not None:
            model.add(pooling(pool, name=f"{model_variation}_pooling_{layer_n}"))

    model.add(Flatten(name=f"{model_variation}_flatten"))

    if dropout_value is not None:
        model.add(Dropout(dropout_value, name=f"{model_variation}_dropout"))

    for layer_n, dense in enumerate(dense_sizes, start=1):
        model.add(Dense(dense, activation="relu", name=f"{model_variation}_dense_{layer_n}"))

    model.add(Dense(12, activation="softmax", name=f"{model_variation}_output"))
    return model


def scale_fn(x):
    return 1.0 / (2.0 ** (x - 1))


def prepare_model(model, learning_rate: float = 0.01, batch_size: int = 64, total_size: int = 48_000, lr_schedule: str = "const", opt="adam"):
    if lr_schedule == "const":
        learning_rate_schedule = learning_rate
    elif lr_schedule == "decay":
        # Decrease the learning rate at a 1/2 of the value every 5 epochs
        learning_rate_schedule = DecayingLRSchedule(learning_rate, batch_size, total_size)
    elif lr_schedule == "cyclic":
        # Cyclic learning rate
        learning_rate_schedule = CyclicLRSchedule(learning_rate, batch_size, total_size)
    else:
        raise ValueError("lr_schedule must be either 'const', 'decay' or 'cyclic'.")

    if opt == "adam":
        opt = Adam(learning_rate=learning_rate_schedule)
    elif opt == "sgd":
        opt = SGD(learning_rate=learning_rate_schedule)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def combine_models(model_frames, model_deepflow, n_classes: int = 12):
    """Combine two models into one mode using late fusion and the functional API.
    Remove the last layer of each model and add a new layer to the combined model."""
    inputs = [model_frames.input, model_deepflow.input]

    # Get the last layer of each model
    outputs_frames = model_frames.layers[-2].output
    outputs_deepflow = model_deepflow.layers[-2].output

    # Concatenate the outputs of the two models
    concat = tf.keras.layers.concatenate([outputs_frames, outputs_deepflow])

    # Add dense layers
    decoder = Dense(128, activation="relu", name="decoder_dense_1")(concat)
    decoder = Dense(64, activation="relu", name="decoder_dense_2")(decoder)
    decoder = Dense(32, activation="relu", name="decoder_dense_3")(decoder)

    # Add the final layer
    outputs = Dense(n_classes, activation="softmax")(decoder)

    # Create a new model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


class CyclicLRSchedule(LearningRateSchedule):
    def __init__(self, learning_rate: float, batch_size: int, total_size: int, decay_rate: float = 0.5,
                 decay_steps: int = 5, warmup_epochs: int = 5, min_lr: float = 1e-6):
        super(CyclicLRSchedule, self).__init__()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.total_size = total_size
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

    def __call__(self, step):
        # Warmup
        if step < self.warmup_epochs * tf.cast(tf.math.floor(self.total_size / self.batch_size), dtype=tf.int64):
            return float(tf.math.floor(self.learning_rate * (step + 1) / (tf.cast(tf.math.floor(self.warmup_epochs * self.total_size / self.batch_size), dtype=tf.float32).astype(tf.float32))))

        # Decay
        if step % tf.cast(tf.math.floor(self.total_size / self.batch_size) * self.decay_steps, dtype=tf.int64) == 0:
            self.learning_rate *= self.decay_rate

        # Cyclic
        lr = float(tf.math.floor(self.learning_rate * (step + 1) / tf.cast(tf.math.floor(self.warmup_epochs *
                   self.total_size / self.batch_size), dtype=tf.float32).astype(tf.float32)))
        return tf.math.maximum(lr, self.min_lr)

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "total_size": self.total_size,
            "decay_rate": self.decay_rate,
            "decay_steps": self.decay_steps,
            "warmup_epochs": self.warmup_epochs,
            "min_lr": self.min_lr
        }


class DecayingLRSchedule(LearningRateSchedule):
    def __init__(self, lr, batch_size, total_size) -> None:
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.total_size = total_size

        self.n_steps = self.total_size // self.batch_size  # Number of steps per epoch, e.g. number of batches per epoch

    def __call__(self, step):
        """Decrease the learning rate at a 1/2 of the value every 5 epochs"""
        epoch = step / self.n_steps  # Current epoch, e.g. epoch 3 or epoch 3.5
        return self.lr * tf.math.pow(0.5, tf.math.floor(epoch / 5))

    def get_config(self):
        return {"lr": self.lr, "batch_size": self.batch_size, "total_size": self.total_size}


if __name__ == "__main__":
    model = make_model()
    model.summary()
    model.summary()
