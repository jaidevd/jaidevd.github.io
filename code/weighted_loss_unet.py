
# coding: utf-8

from keras import layers as L
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np

_epsilon = tf.convert_to_tensor(K.epsilon(), np.float32)


def my_loss(target, output):
    """
    A custom function defined to simply sum the pixelwise loss.
    This function doesn't compute the crossentropy loss, since that is made a
    part of the model's computational graph itself.

    Parameters
    ----------

    target : tf.tensor
        A tensor corresponding to the true labels of an image.
    output : tf.tensor
        Model output

    Returns
    -------
    tf.tensor
        A tensor holding the aggregated loss.

    """
    return - tf.reduce_sum(target * output,
                           len(output.get_shape()) - 1)


def make_weighted_loss_unet(input_shape, n_classes):
    # two inputs, one for the image and one for the weight maps
    ip = L.Input(shape=input_shape)
    # the shape of the weight maps has to be such that it can be element-wise
    # multiplied to the softmax output.
    weight_ip = L.Input(shape=input_shape[:2] + (n_classes,))

    # adding the layers
    conv1 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ip)
    conv1 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = L.Dropout(0.1)(conv1)
    mpool1 = L.MaxPool2D()(conv1)

    conv2 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool1)
    conv2 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = L.Dropout(0.2)(conv2)
    mpool2 = L.MaxPool2D()(conv2)

    conv3 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool2)
    conv3 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = L.Dropout(0.3)(conv3)
    mpool3 = L.MaxPool2D()(conv3)

    conv4 = L.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool3)
    conv4 = L.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = L.Dropout(0.4)(conv4)
    mpool4 = L.MaxPool2D()(conv4)

    conv5 = L.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool4)
    conv5 = L.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = L.Dropout(0.5)(conv5)

    up6 = L.Conv2DTranspose(512, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv5)
    conv6 = L.Concatenate()([up6, conv4])
    conv6 = L.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = L.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = L.Dropout(0.4)(conv6)

    up7 = L.Conv2DTranspose(256, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
    conv7 = L.Concatenate()([up7, conv3])
    conv7 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = L.Dropout(0.3)(conv7)

    up8 = L.Conv2DTranspose(128, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
    conv8 = L.Concatenate()([up8, conv2])
    conv8 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = L.Dropout(0.2)(conv8)

    up9 = L.Conv2DTranspose(64, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
    conv9 = L.Concatenate()([up9, conv1])
    conv9 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = L.Dropout(0.1)(conv9)

    c10 = L.Conv2D(n_classes, 1, activation='softmax', kernel_initializer='he_normal')(conv9)

    # Add a few non trainable layers to mimic the computation of the crossentropy
    # loss, so that the actual loss function just has to peform the
    # aggregation.
    c11 = L.Lambda(lambda x: x / tf.reduce_sum(x, len(x.get_shape()) - 1, True))(c10)
    c11 = L.Lambda(lambda x: tf.clip_by_value(x, _epsilon, 1. - _epsilon))(c11)
    c11 = L.Lambda(lambda x: K.log(x))(c11)
    weighted_sm = L.multiply([c11, weight_ip])

    model = Model(inputs=[ip, weight_ip], outputs=[weighted_sm])
    return model
