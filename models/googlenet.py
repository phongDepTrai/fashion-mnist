from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, Concatenate, GlobalAvgPool2D


def inception(inputs, output_channels):
    # path 1
    p_1 = Conv2D(filters=output_channels[0], kernel_size=1, activation='relu')(inputs)

    # path 2
    p_2 = Conv2D(filters=output_channels[1], kernel_size=1, activation='relu')(inputs)
    p_2 = Conv2D(filters=output_channels[2], kernel_size=3, padding='same', activation='relu')(p_2)

    # path 3
    p_3 = Conv2D(filters=output_channels[3], kernel_size=1, activation='relu')(inputs)
    p_3 = Conv2D(filters=output_channels[4], kernel_size=3, padding='same', activation='relu')(p_3)

    # path 4
    p_4 = MaxPool2D(pool_size=3, strides=1, padding='same')(inputs)
    p_4 = Conv2D(filters=output_channels[5], kernel_size=1, activation='relu')(p_4)

    outputs = Concatenate()([p_1, p_2, p_3, p_4])
    return outputs


def create_model(inputs, name='googlenet'):
    x = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')(inputs)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = inception(inputs=x, output_channels=[16, 32, 32, 4, 8, 8])
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = inception(inputs=x, output_channels=[16, 32, 32, 4, 8, 8])
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(10, name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def create_model_2(inputs, name='googlenet'):
    x = Conv2D(filters=16, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
    x = Conv2D(filters=16, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
    x = inception(inputs=x, output_channels=[32, 32, 64, 8, 16, 16])
    x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
    x = inception(inputs=x, output_channels=[32, 32, 64, 8, 16, 16])
    x = GlobalAvgPool2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(10, name=name)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
