# Import all modules
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import csv

from distutils.util import strtobool

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataset import load_dataset

import sys
sys.path.append('../Methodology/')
from eval_functions_eembc import calculate_ae_pr_accuracy, calculate_ae_auc
from toyadmos_autoencoder_eembc import toyadmos_autoencoder_eembc
#from tflite_model import TfliteModel
from callbacks import lr_schedule, TrainingCallbacks, TrainingSequence

def main():
    # Parse the args
    parser = argparse.ArgumentParser(description='This script evaluates an classification model')
    parser.add_argument("--training", '-t', default='False', help='Determines whether to run the training phase or just infer')
    parser.add_argument("--lite", '-l', default='False', help='Determines whether to use the the tflite or floating point model')
    parser.add_argument("--epochs", '-e', default='100', help='Number of epochs to run')
    args = parser.parse_args()

    try:
        do_training = bool(strtobool(args.training))
    except ValueError:
        do_training = True

    try:
        use_tflite_model = bool(strtobool(args.lite))
    except ValueError:
        use_tflite_model = False

    # FIXME, make these configurable via command line
    do_extraction = False
    base_dir = "C:/cygwin64/home/antorrin/altus/modeling/projects/toyadmos/source/matlab"
    model = "ToyCar"
    cases = (1,)

    # Training parameters
    validation_split = .2
    name = "toyadmos_autoencoder_eembc"
    epochs = int(args.epochs)
    batch_size = 40
    log_dir = 'log_toyadmos'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get dataset
    X_normal_train, X_normal_val, X_anomalous = load_dataset(base_dir=base_dir, model=model, 
        cases=cases, log_dir=log_dir, validation_split=validation_split, do_extraction=do_extraction)

    #init_gpus()

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Build model
    model = toyadmos_autoencoder_eembc()
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    model.summary()

    # Augmentation
    sequence_normal_train = TrainingSequence(x=X_normal_train,batch_size=batch_size)
    sequence_normal_val = TrainingSequence(x=X_normal_val,batch_size=batch_size,is_training=False)
    sequence_anomalous = TrainingSequence(x=X_anomalous,batch_size=batch_size,is_training=False)

    if( do_training ):
        # Train
        callbacks = [LearningRateScheduler(lr_schedule), TrainingCallbacks()]
        model_t = model.fit(sequence_normal_train, epochs=epochs, callbacks=callbacks, validation_data=sequence_normal_val)

        # Plot training results
        plt.figure()
        plt.plot(model_t.history['mean_squared_error'],'r')
        plt.plot(model_t.history['val_mean_squared_error'],'g')
        plt.ylim([0, 1])
        plt.xticks(np.arange(0, epochs+1, 50.0))
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.xlabel("Num of Epochs")
        plt.ylabel("MSE")
        plt.title("Training results")
        plt.legend(['train','validation'])
    
        plt.show(block=False)
        plt.pause(0.1)  

        # Save the weights
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model.save_weights(log_dir + '/weights.hdf5')

    # Load and evaluate the quantized tflite model
    if( use_tflite_model ):
        tflite_model = TfliteModel()
        tflite_model.load(log_dir + '/model.tflite')
        y_pred = tflite_model.predict(x_test, batch_size)
    # Or else normal model
    else:
        model.load_weights(log_dir + '/weights.hdf5')
        y_pred_normal_val = model.predict(sequence_normal_val, verbose=1)
        y_pred_anomalous = model.predict(sequence_anomalous, verbose=1)

    # Get y_true
    y_true_normal_val = np.zeros_like(y_pred_normal_val)
    for index in range(sequence_normal_val.__len__()):
        y_true_normal_val[index * batch_size:(index + 1) * batch_size], _ = sequence_normal_val.__getitem__(index)
    y_true_anomalous = np.zeros_like(y_pred_anomalous)
    for index in range(sequence_anomalous.__len__()):
        y_true_anomalous[index * batch_size:(index + 1) * batch_size], _ = sequence_anomalous.__getitem__(index)

    # Calculate MSE
    y_pred_normal_mse = np.mean((y_pred_normal_val - y_true_normal_val)**2, axis=(1,2,3))
    y_pred_anomalous_mse = np.mean((y_pred_anomalous - y_true_anomalous)**2, axis=(1,2,3))

    plt.figure()
    plt.plot(y_pred_normal_mse,'r')
    plt.plot(y_pred_anomalous_mse,'g')
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Examples")
    plt.ylabel("MSE")
    plt.title("Classification results")
    plt.legend(['normal','anomalous'])
    
    plt.show(block=False)
    plt.pause(0.1)  

    # Calculate and print the results
    calculate_ae_pr_accuracy(np.concatenate((y_pred_normal_mse, y_pred_anomalous_mse)), 
        np.concatenate((np.zeros_like(y_pred_normal_mse), np.ones_like(y_pred_anomalous_mse))))
    calculate_ae_auc(np.concatenate((y_pred_normal_mse, y_pred_anomalous_mse)), 
        np.concatenate((np.zeros_like(y_pred_normal_mse), np.ones_like(y_pred_anomalous_mse))),
        name)

    input('Press CTRL+C to exit')
    try:
        plt.show()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    #tensorflow_import_error_workaround()
    main()

