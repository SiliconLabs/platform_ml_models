# Sources
* Dataset
    * [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* Model Topology
    * [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
    * [https://keras.io/api/applications/resnet/](https://keras.io/api/applications/resnet/)
    
# Training details
``` python
#learning rate schedule
def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    print('Learning rate = %f'%lrate)
    return lrate

#optimizer
optimizer = tf.keras.optimizers.Adam()

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