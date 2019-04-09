# ONNX Toolset



The goal of this repo is to build a toolset for automatically transforming the pretrained model into ONNX model.

### What is ONNX

ONNX is a open format to represent deep learning models. With ONNX, AI developers can more easily move models between state-of-the-art tools and choose the combination that is best for them. ONNX is developed and supported by a community of partners.

### Installation

`git clone https://github.com/Joeywzr/ONNX.git`

### How to use

**--type/-t**: input 'caffe2' or 'tensorflow' or 'pytorch'

**--net/-n:** support

**--download/-d:** download the models or not

**--d_path/-dp:** download path

**--convert2onnx/-c2o:** convert the models to onnx models or not

**--output/-o:** output path of onnx models 



* To print settings of pretrained models:

~~~shell
python main.py -t pytorch -n resnet18
~~~

* To download pretrained models:

~~~shell
python main.py -t pytorch -n resnet18 -d True
~~~

* To convert pretrained models to onnx models:

~~~shell
python main.py -t pytorch -n resnet18 -c2o True
~~~

### Evaluation on imagenet

Accuracy on validation set:

| Model             | Version | Acc@1  | Acc@5  |
| ----------------- | ------- | ------ | ------ |
| InceptionResNetV2 | Pytorch | 80.270 | 95.140 |
| InceptionV4       | Pytorch | 80.082 | 94.890 |
| ResNet152         | Pytorch | 78.428 | 94.046 |
| SE-ResNet50       | Pytorch | 77.636 | 93.752 |
| DenseNet161       | Pytorch | 77.138 | 93.560 |
| ResNet101         | Pytorch | 77.374 | 93.546 |
| Inceptionv3       | Pytorch | 77.320 | 93.434 |
| DenseNet201       | Pytorch | 76.896 | 93.370 |
| DenseNet169       | Pytorch | 75.600 | 92.806 |
| ResNet50          | Pytorch | 76.130 | 92.862 |
| DenseNet121       | Pytorch | 74.434 | 91.972 |
| ResNet34          | Pytorch | 73.314 | 91.420 |
| BNInception       | Pytorch | 73.524 | 91.562 |
| VGG19             | Pytorch | 72.376 | 90.876 |
| VGG16             | Pytorch | 71.592 | 90.382 |
| ResNet18          | Pytorch | 69.758 | 89.076 |



### Reference

https://github.com/Cadene/pretrained-models.pytorch
