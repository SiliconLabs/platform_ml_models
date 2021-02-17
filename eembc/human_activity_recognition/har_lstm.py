from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np

def lstm_activity_classification_eembc():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(None,9)))
    model.add(Dropout(0.1))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid',
                  return_sequences=False, kernel_regularizer=regularizers.l2(0.00001),
              recurrent_regularizer=regularizers.l2(0.00001)))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    return model

model = lstm_activity_classification_eembc()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

# %% load train data
tot_x = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/inertial/total_acc_x_train.txt').astype('float16')
tot_y = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/inertial/total_acc_y_train.txt').astype('float16')
tot_z = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/inertial/total_acc_z_train.txt').astype('float16')
acc_x = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/inertial/body_acc_x_train.txt').astype('float16')
acc_y = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/inertial/body_acc_y_train.txt').astype('float16')
acc_z = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/inertial/body_acc_z_train.txt').astype('float16')
gyr_x = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/inertial/body_gyro_x_train.txt').astype('float16')
gyr_y = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/inertial/body_gyro_y_train.txt').astype('float16')
gyr_z = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/inertial/body_gyro_z_train.txt').astype('float16')

# %%
x_train = np.stack((tot_x, tot_y, tot_z, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z), axis=2)
y_train_raw = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/train/y_train.txt').astype('int8')
y_train = to_categorical(y_train_raw-1)

# %% load test data
tot_x = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/inertial/total_acc_x_test.txt').astype('float16')
tot_y = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/inertial/total_acc_y_test.txt').astype('float16')
tot_z = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/inertial/total_acc_z_test.txt').astype('float16')
acc_x = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/inertial/body_acc_x_test.txt').astype('float16')
acc_y = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/inertial/body_acc_y_test.txt').astype('float16')
acc_z = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/inertial/body_acc_z_test.txt').astype('float16')
gyr_x = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/inertial/body_gyro_x_test.txt').astype('float16')
gyr_y = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/inertial/body_gyro_y_test.txt').astype('float16')
gyr_z = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/inertial/body_gyro_z_test.txt').astype('float16')

# %%
x_test = np.stack((tot_x, tot_y, tot_z, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z), axis=2)
y_test_raw = np.loadtxt('/data/jaelenes/datasets/UCI_HAR/test/y_test.txt').astype('int8')
y_test = to_categorical(y_test_raw-1)


# %% train model
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
    ],
)

# %% test model
y_pred = model.predict(x_test)

# compute accuracy
correct = np.sum(1.0*(np.argmax(y_test,axis=1)==np.argmax(y_pred,axis=1)))
tot = y_test.shape[0]
acc = correct/tot
print(acc)


