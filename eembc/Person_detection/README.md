# Sources
* Dataset
    * MSCOCO14 based [https://cocodataset.org/#download]
    * Extraction based on COCO API [https://github.com/cocodataset/cocoapi]
    * Person mimimal bounding box 2.5%
    * 96x96 images resized with bilinear scaling, no aspect ratio preservation
    * All images converted to RGB
    * Training and validation sets combined
* Extracted Reference Dataset
   * [vw_coco2014_96.tar.gz](https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz)
* Model Topology
    * Based on [https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)
        * Chosen configuration is a MobileNet_v1_0.25_96

# Performance
* Accuracy
    * 85.4%
* AUC
    * .931

