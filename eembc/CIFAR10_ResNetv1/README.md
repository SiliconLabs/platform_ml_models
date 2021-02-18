# Sources
* Dataset
    * [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* Model Topology
    * [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
    * [https://keras.io/api/applications/resnet/](https://keras.io/api/applications/resnet/)
    
# Required software packages
- Python [3.7.x or 3.8.x](https://www.python.org/downloads/) 
- matplotlib to install type `pip install matplotlib` and follow the same approach for packages below.
- tensorflow (3.4.x)
- sklearn

# To run the code
To simply train and test with CIFAR10, type `python cifar10_main.py`. The code will train the network and show the training history. Then it will test with the validation data and show accuracy, AUC and confusion matrix.
Type `python cifar10_main.py --help` to see all available options.

__NOTE__ Under Windows, the following error may be encountered `AttributeError: module 'tensorflow.lite.python.schema_py_generated' has no attribute 'Model'`. If that happens typically `schema_py_generated.py`, located in `Python3x\Lib\site-packages\tensorflow\lite\python` is empty. Find a non-empty copy - ideally under Linux - and copy the content.

# Training details

## Optimizer
The optimizer is Adam with default options. 
```python
#optimizer
optimizer = tf.keras.optimizers.Adam()
```

## Learning rate
We start with rate of .001 and follow an exponential decay 
``` python
#learning rate schedule
def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    print(f"Learning rate = {lrate:.5f}")
    return lrate
```
For code reference see [`callbacks.py`](./callbacks.py).

## Augmentation
```python
#define data generator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    #brightness_range=(0.9, 1.2),
    #contrast_range=(0.9, 1.2)
)
```
For the code reference see [`cifar10_main.py`](./cifar10_main.py#L86).
    
# Performance (floating point model)
* Accuracy
    * 86.2%
* AUC
    * .989

# Performance (quantized tflite model)
* Accuracy
    * 86.1%
* AUC
    * .988