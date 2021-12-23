# CNN for Handwritten Character Recognition - EMNIST balanced dataset

## Description

This is a convolutional neural network (CNN) trained to recognize handwritten characters. It was trained with the EMNIST balanced ByMerge dataset and implemented using the Pytorch API. An accuracy of 86% was achieved.

## How to use
To use the CNN for inference, the Character_Recognition API must be imported.
```
from CNN_EMNIST import Character_Recognition as CR
```
The main.py code exemplifies how to :
  - Test the CNN using de test EMNIST balanced dataset 
  - Perform inference using a local image.

The dimensions of th einput image must be 28x28.

It is also possible to re-train the network and get the loss value curve by executing the *CNN_EMNIST.py* file



