from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal


def create_model(inputs, name='alexnet'):
    """ Original model of alexnet"""
    x = Conv2D(filters=32, kernel_size=11, strides=4, padding='same', activation='relu', kernel_regularizer=l2(0.01),
               kernel_initializer=RandomNormal(mean=0.0, stddev=2.0))(inputs)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(0.01),
               kernel_initializer=RandomNormal(mean=0.0, stddev=2.0))(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.01),
               kernel_initializer=RandomNormal(mean=0.0, stddev=2.0))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.01),
               kernel_initializer=RandomNormal(mean=0.0, stddev=2.0))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.01),
               kernel_initializer=RandomNormal(mean=0.0, stddev=2.0))(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(10, name=name)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def create_model_2(inputs, name='alexnet'):
    """ Original model of alexnet"""
    x = Conv2D(filters=32, kernel_size=11, strides=4, padding='same', activation='relu', kernel_regularizer=l2(0.01))(
        inputs)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.01))(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(10, name=name)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
