from tensorflow import keras
from tensorflow.keras import layers


def create_model(inputs, name):
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10, name=name)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_model_2(inputs, name):
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(inputs)
    outputs = layers.Dense(10, name=name)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_model_sm(inputs, name):
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(inputs)
    outputs = layers.Dense(10, activation="softmax", name=name)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
