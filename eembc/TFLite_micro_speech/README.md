# Sources
* Dataset
    * [2.3GB wave-file archive](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
    * [Paper Describing the Dataset](https://arxiv.org/abs/1804.03209)
* Model Topology & Training
    * [TensorFlow Lite Micro README.md](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/micro_speech/train)
    * [Training Instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb)
* Trained Models
    * See the "trained_models" folder here. This is an exact duplicate of the 2020/04/13 models from TFLite.

# Performance
* Accuracy
    * 93.7%
* AUC
    * .993
* NOTE
    * Performance is somewhat sensitive to the exact process of spectrogram generation. We may need to precisely define that.

