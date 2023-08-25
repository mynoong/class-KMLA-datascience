# Classifying Handwritten numbers

## Description
The application classifies single-digit integers ranging from 0 to 9 via a neural network architecture. The software proceeds the training process by leveraging the MNIST dataset of handwritten numerals, sourced from https://yann.lecun.com/exdb/mnist/. Notably, the MNIST dataset comprises a collection of 60,000 training samples and 10,000 testing samples. Each datum consists of a 28x28 pixel image, accompanied by an associated true-label denoting the depicted numeral. 

The program employs three distinct neural network configurations, each tailored to optimize the network's capacity for accurate digit recognition. Here, the Keras library is harnessed to configure the requisite network layers.

## Related Topics
* Feed Forward Neural Network
* ReLU Activation Function
* Convolutional Neural Network (CNN)
* Dropout
* Max Pooling, Average Pooling
* Keras Library

## Classification Result

| 2-Layered Feed Forward Neural Network (Acc = 89.6%, Cal Time = 12.0sec) | 2-Layered CNN  (Acc = 98.0%, Cal Time = 145sec) | Customized CNN w/ Dropout and Pooling (Acc = 99.1%, Cal Time = 927sec) |       
| ----------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------|
| ![2-layered FNN](https://github.com/mynoong/machine-learning-basics/assets/113654157/c27250d8-9f71-4e4a-b5d1-49c0e7e17e7e) | ![2-layered CNN](https://github.com/mynoong/machine-learning-basics/assets/113654157/66ab497d-0c7b-4076-a7bd-391024a0c9d1) |  ![customized CNN](https://github.com/mynoong/machine-learning-basics/assets/113654157/8e58f1d8-6bda-4ec1-bb3f-2de6643736b8) |

Green horizontal line on a top of image indicates the wrong classification.
