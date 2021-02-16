import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Add
from tensorflow.keras.regularizers import l1_l2
from qkeras.qlayers import QDense, QActivation
from qkeras.qconvolutional import QConv2D
from qkeras.qpooling import QAveragePooling2D
from qkeras.quantizers import quantized_bits, quantized_relu

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


#quantized model
def resnet_v1_eembc_quantized(input_shape=[32, 32, 3], num_classes=10, num_filters=[16, 32, 64], 
                    kernel_sizes=[3, 1], strides=[1, 2], l1p=1e-4, l2p=0,
                    logit_total_bits=7, logit_int_bits=0, activation_total_bits=7, activation_int_bits=3):

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = QConv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(inputs)
    x = BatchNormalization()(x)
    x = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(x)

    # First stack
    # Weight layers
    y = QConv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
    y = BatchNormalization()(y)
    y = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(y)
    y = QConv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    y = BatchNormalization()(y)
  
    # Overall residual, connect weight layer and identity paths
    x = Add()([x, y]) 
    x = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(x)

    # Second stack
    # Weight layers
    y = QConv2D(num_filters[1],
                  kernel_size=kernel_sizes[0],
                  strides=strides[1],
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
    y = BatchNormalization()(y)
    y = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(y)
    y = QConv2D(num_filters[1],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    y = BatchNormalization()(y)
  
    # Adjust for change in dimension due to stride in identity
    x = QConv2D(num_filters[1],
                  kernel_size=kernel_sizes[1],
                  strides=strides[1],
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)

    # Overall residual, connect weight layer and identity paths
    x = Add()([x, y])
    x = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(x)

    # Third stack
    # Weight layers
    y = QConv2D(num_filters[2],
                  kernel_size=kernel_sizes[0],
                  strides=strides[1],
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
    y = BatchNormalization()(y)
    y = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(y)
    y = QConv2D(num_filters[2],
                  kernel_size=kernel_sizes[0],
                  strides=strides[0],
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    y = BatchNormalization()(y)
  
    # Adjust for change in dimension due to stride in identity
    x = QConv2D(num_filters[2],
                  kernel_size=kernel_sizes[1],
                  strides=strides[1],
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)

    # Overall residual, connect weight layer and identity paths
    x = Add()([x, y])
    x = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(x)

    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    # Do we want Qlayer of this below?
    x = QAveragePooling2D(pool_size=pool_size, quantizer=quantized_relu(activation_total_bits, activation_int_bits))(x)
    y = Flatten()(x)
    # Changed output to separate QDense but did not quantize softmax as specified, is this the way you wanted it?
    outputs = QDense(num_classes,
                     kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                     bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                     kernel_initializer='he_normal',
                     kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    outputs = Activation(activation='softmax')(outputs)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
