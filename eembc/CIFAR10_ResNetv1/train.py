import setGPU
import os
import glob
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import resnet_v1_eembc
import yaml

#from keras_flops import get_flops #(different flop calculation)
import kerop

from tensorflow.keras.datasets import cifar10

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    print('Learning rate = %f'%lrate)
    return lrate

def main(args):

    # parameters
    input_shape = [32,32,3]
    num_classes = 10
    config = yaml_load(args.config)
    num_filters = config['model']['filters']
    kernel_sizes = config['model']['kernels']
    strides = config['model']['strides']
    l1p = float(config['model']['l1'])
    l2p = float(config['model']['l2'])
    batch_size = config['fit']['batch_size']
    num_epochs = config['fit']['epochs']
    verbose = config['fit']['verbose']
    patience = config['fit']['patience']
    save_dir = config['save_dir']
    model_name = config['model']['name']
    model_file_path = os.path.join(save_dir, 'model_best.h5')

    #optimizer
    optimizer = tf.keras.optimizers.Adam()

    # load dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # define data generator
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        #brightness_range=(0.9, 1.2),
        #contrast_range=(0.9, 1.2)
    )

    # run preprocessing on training dataset
    datagen.fit(X_train)

    kwargs = {'input_shape': input_shape,
              'num_classes': num_classes,
              'num_filters': num_filters,
              'kernel_sizes': kernel_sizes,
              'strides': strides,
              'l1p': l1p,
              'l2p': l2p}

    # define model
    model = getattr(resnet_v1_eembc,model_name)(**kwargs)

    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(model.summary())
    print('#################')

    # analyze FLOPs (see https://github.com/kentaroy47/keras-Opcounter)
    layer_name, layer_flops, inshape, weights = kerop.profile(model)

    # visualize FLOPs results
    total_flop = 0
    for name, flop, shape in zip(layer_name, layer_flops, inshape):
        print("layer:", name, shape, " MFLOPs:", flop/1e6)
        total_flop += flop
    print("Total FLOPs: {} GFLOPs".format(total_flop/1e9))

    # Alternative FLOPs calculation (see https://github.com/tokusumi/keras-flops), ~same answer
    #total_flop = get_flops(model, batch_size=1)
    #print("FLOPS: {} GLOPs".format(total_flop/1e9))

    # compile model with optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # callbacks
    from tensorflow.keras.callbacks import EarlyStopping,History,ModelCheckpoint,ReduceLROnPlateau

    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=verbose),
                 tf.keras.callbacks.ModelCheckpoint(model_file_path, monitor='val_loss', verbose=verbose, save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose, restore_best_weights=True)
    ]

    # train
    model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
              steps_per_epoch=X_train.shape[0] // batch_size,
              epochs=num_epochs,
              validation_data=(X_test, y_test),
              callbacks=callbacks,
              verbose=verbose)


    # restore "best" model
    model.load_weights(model_file_path)

    # get predictions
    y_pred = model.predict(X_test)

    # evaluate with test dataset and share same prediction results
    evaluation = model.evaluate(datagen.flow(X_test, y_test, batch_size=batch_size),
                                steps=X_test.shape[0] // batch_size)

    
    auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

    print('Model accuracy = %.3f' % evaluation[1])
    print('Model weighted average AUC = %.3f' % auc)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "baseline.yaml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
