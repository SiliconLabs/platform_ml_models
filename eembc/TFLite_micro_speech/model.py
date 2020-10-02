import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, BatchNormalization, Activation

def micro_speech_eembc():
  model = Sequential()
  model.add(Input(
	  [49, 40 ,1]))
  model.add(Conv2D(
	  filters=8,
	  kernel_size=(10, 8),
	  use_bias=True,
	  padding="same",
	  strides=(2,2)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(
	  rate=0.5))
  model.add(Flatten())
  model.add(Dense(
	  units=4,
	  use_bias=True,
	  activation='softmax'))
  return model

