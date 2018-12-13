from keras.layers import (Conv2D, Input, MaxPool2D, Conv2DTranspose, Cropping2D,
                          Concatenate, Dropout)
from keras.models import Model


def make_unet(input_shape, n_classes):
    """
    Make a vanilla UNet.

    Parameters
    ----------

    input_shape : tuple
        Shape of the images in the training data.
    n_classes : int
        Number of classes.

    Returns
    -------
    A Keras model of the UNet.

    """
    ip = Input(shape=input_shape)
    c1 = Conv2D(64, 3, kernel_initializer='he_normal', activation='relu', padding='same')(ip)
    c1 = Conv2D(64, 3, kernel_initializer='he_normal', activation='relu', padding='same')(c1)
    c1 = Dropout(0.1)(c1)

    m1 = MaxPool2D()(c1)
    c2 = Conv2D(128, 3, kernel_initializer='he_normal', activation='relu', padding='same')(m1)
    c2 = Conv2D(128, 3, kernel_initializer='he_normal', activation='relu', padding='same')(c2)
    c2 = Dropout(0.2)(c2)

    m2 = MaxPool2D()(c2)
    c3 = Conv2D(256, 3, kernel_initializer='he_normal', activation='relu', padding='same')(m2)
    c3 = Conv2D(256, 3, kernel_initializer='he_normal', activation='relu', padding='same')(c3)
    c3 = Dropout(0.2)(c3)

    m3 = MaxPool2D()(c3)
    c4 = Conv2D(512, 3, kernel_initializer='he_normal', activation='relu', padding='same')(m3)
    c4 = Conv2D(512, 3, kernel_initializer='he_normal', activation='relu', padding='same')(c4)
    c4 = Dropout(0.2)(c4)

    m4 = MaxPool2D()(c4)
    c5 = Conv2D(1024, 3, kernel_initializer='he_normal', activation='relu', padding='same')(m4)
    c5 = Conv2D(1024, 3, kernel_initializer='he_normal', activation='relu', padding='same')(c5)
    c5 = Dropout(0.3)(c5)

    u1 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    crop1 = Cropping2D(4)(c4)
    conc1 = Concatenate()([u1, crop1])
    c6 = Conv2D(512, 3, kernel_initializer='he_normal', activation='relu', padding='same')(conc1)
    c6 = Conv2D(512, 3, kernel_initializer='he_normal', activation='relu', padding='same')(c6)
    c6 = Dropout(0.2)(c6)

    u2 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    crop2 = Cropping2D(16)(c3)
    conc2 = Concatenate()([u2, crop2])
    c7 = Conv2D(256, 3, kernel_initializer='he_normal', activation='relu', padding='same')(conc2)
    c7 = Conv2D(256, 3, kernel_initializer='he_normal', activation='relu', padding='same')(c7)
    c7 = Dropout(0.2)(c7)

    u3 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    crop3 = Cropping2D(40)(c2)
    conc3 = Concatenate()([u3, crop3])
    c8 = Conv2D(128, 3, kernel_initializer='he_normal', activation='relu', padding='same')(conc3)
    c8 = Conv2D(128, 3, kernel_initializer='he_normal', activation='relu', padding='same')(c8)
    c8 = Dropout(0.2)(c8)

    u4 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    crop4 = Cropping2D(88)(c1)
    conc4 = Concatenate()([u4, crop4])
    c9 = Conv2D(64, 3, kernel_initializer='he_normal', activation='relu', padding='same')(conc4)
    c9 = Conv2D(64, 3, kernel_initializer='he_normal', activation='relu', padding='same')(c9)
    c9 = Dropout(0.1)(c9)

    c10 = Conv2D(n_classes, 1, activation='softmax', kernel_initializer='he_normal')(c9)

    model = Model(inputs=[ip], outputs=[c10])
    return model
