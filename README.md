# tfcancer
A transfer learning approach using deep and wide learning for cancer prediction using ISPY1 dataset


# pre-requisites
Clone this keras [fork](https://github.com/hasnainv/keras) to convert the model.

# Script for converting models
python -m keras.caffe.caffe2keras -load_path ../data/models/vgg/caffe/ -prototxt VGG_ILSVRC_16_layers_deploy.prototxt -caffemodel VGG_ILSVRC_16_layers.caffemodel