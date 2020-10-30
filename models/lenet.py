from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization


def create_model(inputs):
    """ Original model of lenet"""
    x = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')(inputs)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    # x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)

    x = Dense(64, activation="relu")(x)
    outputs = Dense(10, name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def create_model_do(inputs):
    """ Add Dropout layer """
    x = Conv2D(filters=6, kernel_size=5, padding='same', activation='relu')(inputs)
    x = AvgPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=16, kernel_size=5, padding='same', activation='relu')(x)
    x = AvgPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(120, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(84, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(10, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def create_model_bn(inputs):
    """ Add BatchNormalization layer """
    x = Conv2D(filters=6, kernel_size=5, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = AvgPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=16, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = AvgPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(120, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(84, activation="relu")(x)
    x = BatchNormalization()(x)
    outputs = Dense(10, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def create_model_single_output(inputs):
    """ Change output """
    x = Conv2D(filters=6, kernel_size=5, padding='same', activation='relu')(inputs)
    x = AvgPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=16, kernel_size=5, padding='same', activation='relu')(x)
    x = AvgPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(120, activation="relu")(x)
    x = Dense(84, activation="relu")(x)
    x = Dense(10, activation="relu")(x)
    outputs = Dense(1, name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
