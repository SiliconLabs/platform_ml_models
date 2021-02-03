import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Add
from tensorflow.keras.regularizers import l1_l2

#define model
def resnet_v1_eembc(input_shape=[32, 32, 3], num_classes=10, num_filters=[16, 32, 64], 
                    kernel_sizes=[3, 1], strides=[1, 2], l1p=1e-4, l2p=0):

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # First stack
    # Weight layers
    y = Conv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    y = BatchNormalization()(y)
  
    # Overall residual, connect weight layer and identity paths
    x = Add()([x, y]) 
    x = Activation('relu')(x)

    # Second stack
    # Weight layers
    y = Conv2D(num_filters[1],
                  kernel_size=kernel_sizes[0],
                  strides=strides[1],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters[1],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    y = BatchNormalization()(y)
  
    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters[1],
                  kernel_size=kernel_sizes[1],
                  strides=strides[1],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)

    # Overall residual, connect weight layer and identity paths
    x = Add()([x, y])
    x = Activation('relu')(x)

    # Third stack
    # Weight layers
    y = Conv2D(num_filters[2],
                  kernel_size=kernel_sizes[0],
                  strides=strides[1],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters[2],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    y = BatchNormalization()(y)
  
    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters[2],
                  kernel_size=kernel_sizes[1],
                  strides=strides[1],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)

    # Overall residual, connect weight layer and identity paths
    x = Add()([x, y])
    x = Activation('relu')(x)

    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v1_eembc_tiny(input_shape=[32, 32, 3], num_classes=10, num_filters=[8], 
                         kernel_sizes=[3, 1], strides=[1, 2], l1p=1e-4, l2p=0):

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # First stack
    # Weight layers
    y = Conv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=strides[1],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters[0],
                  kernel_size=kernel_sizes[1],
                  strides=strides[1],
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)

    # Overall residual, connect weight layer and identity paths
    x = Add()([x, y]) 
    x = Activation('relu')(x)

    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
