# CS182 Final Project: YOLO

Implementation of the YOLO Object Detection model based on the paper ["You Only Look Once: Unified, Real-Time Object Detection"](https://arxiv.org/abs/1506.02640) along with a simple two class dataset to train on.

This project was created by Eric Li, Jonathan Yue, Vikas Ummadisetty, and Nishant Bhakar
for the CS182 final project.

![alt text](./yoloV1test.png "title")

## Project Overview

For this project, we created an original dataset the CircleSquares dataset that is a simple two class dataset for object detection. This test set was created for simplicitly and faster learning for student notebooks as the original paper on PASCAL VOC would have required 2 weeks on a GPU. Our simpler dataset however can be trained in 10 minutes. 

This implementation of the model also uses batch-Norm rather than dropout to prevent overfitting.

For the model and loss function, we adapated an implementation by Aladdin Persson https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO

Our modifications include fixing bugs and removing hard coded components to properly interface with our new dataset. 

## How to use this repo

To run this project, first run `createDatasets.py` then `train.py`. This will show a test prediciton on screen. Here are descriptions of every file in the project

* `train.py` is the main file for this project. Running it will train the model and show a test prediction on screen
* `createDatasets.py` can be run as a script and will create a fixed training dataset in the same directory
* `loss.py` contains the loss function used for YOLO
* `model.py` contains the YOLO model. It is essentially just a nn.Sequential() on a lot of conv layers and FC layers
* `overfit.pth.tar` contains a model we trained for 10 epochs (around 15 mins on a laptop)

The contents of `train.py` can be modified fairly freely depending on what you want to do. You could either load the pre-trained model and perform test predictions or train your own model. Set `LOAD_MODEL` to true if you want to load our pretrained model. It is set to false by default.

You will have to tweak the hyperparameters to get the model training on your machine. For example, you may need to lower the batch size if you are getting a `CUDNN_STATUS_NOT_INITIALIZED` error which means you are running out of GPU memory.

Don't tweak the learning rate too much. It is pretty much as high as it can go. The model diverges at the start very quickly if learning rate is high. The original paper actually ramps up the learning rate during training then lets it decay.

## Final words
If you want to play around more with this implementation, note that we hard coded B=2 for a lot of the code; however, the model can easily be modified to remedy this. We did not find more than two boxes per cell necessary so we followed the original paper. If you want to modify the height and width of the input image, you will have to recalculate convolution kernel dimensions or else the output of the Convolution layers will not be the correct shape for the fully connected layers.