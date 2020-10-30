from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, \
    Activation, add


def rest_block(inputs, down_factor=1, num_filters=16):
    """ Original model of lenet"""
    x = Conv2D(filters=num_filters, strides=down_factor, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=num_filters, strides=1, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)

    y = Conv2D(filters=num_filters, strides=down_factor, kernel_size=1, padding='same')(inputs)

    x = add([x, y])
    x = Activation('relu')(x)
    return x


def create_model(inputs, name="resnet"):
    """ Original model of lenet"""
    x = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')(inputs)
    x = rest_block(inputs=x, num_filters=16)
    x = rest_block(inputs=x, num_filters=16)
    x = rest_block(inputs=x, num_filters=16)
    x = rest_block(inputs=x, down_factor=2, num_filters=32)
    x = rest_block(inputs=x, num_filters=32)
    x = rest_block(inputs=x, num_filters=32)
    x = rest_block(inputs=x, num_filters=32)
    x = rest_block(inputs=x, down_factor=2, num_filters=64)
    x = rest_block(inputs=x, num_filters=64)
    x = rest_block(inputs=x, num_filters=64)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(10, name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def create_model_2(inputs, name="resnet"):
    """ Original model of lenet"""
    x = rest_block(inputs=inputs, down_factor=1, num_filters=16)
    x = rest_block(inputs=x, down_factor=2, num_filters=32)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(10, name=name)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def create_model_3(inputs, name="resnet"):
    """ Original model of lenet"""
    x = rest_block(inputs=inputs, down_factor=1, num_filters=16)
    x = rest_block(inputs=x, down_factor=2, num_filters=32)
    x = rest_block(inputs=x, down_factor=2, num_filters=64)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(10, name=name)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
