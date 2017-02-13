# SSD: Single Shot MultiBox Detector

This is an implementation of SSD (Single Shot MultiBox Detector) using Chainer

## Requirement

- Python 3.5+
- [Chainer](https://github.com/pfnet/chainer) 1.20+
    - DilatedConvolution2D is required.
- OpenCV 3

## Usage
### Testing
#### 1\. Download pre-traind Caffe model from https://github.com/weiliu89/caffe/tree/ssd#models
```
curl -LO http://www.cs.unc.edu/%7Ewliu/projects/SSD/models_VGGNet_VOC0712Plus_SSD_300x300.tar.gz
tar xf models_VGGNet_VOC0712Plus_SSD_300x300.tar.gz
```
#### 2\. Convert weights
```
python3 convert_caffe.py models/VGGNet/VOC0712Plus/SSD_300x300/VGG_VOC0712Plus_SSD_300x300_iter_240000.caffemodel ssd300.npz
```
#### 3\. Predict
```
python3 predict.py ssd300.npz image.jpg
```
(press 'q' to exit)  
![result](result.jpg "result")

### Training (on going)
#### 1\. Download pre-trained VGG16 model (fc reduced) from https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6
```
curl -LO http://cs.unc.edu/~wliu/projects/ParseNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
```
#### 2\. Convert weights
```
python3 convert_caffe.py --baseonly VGG_ILSVRC_16_layers_fc_reduced.caffemodel vgg16.npz
```
#### 3\. Download VOC2007 dataset
```
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar
```
#### 4\. Train
```
python3 train.py --init vgg16.npz --root VOCdevkit/VOC2007/ [--gpu gpu]
```

## ToDo
- Add data augmentation
- Evaluate converted/trained models

## References
+ Liu, Wei, et al. "SSD: Single shot multibox detector." ECCV2016.
+ [Original implementation](https://github.com/weiliu89/caffe/tree/ssd)
