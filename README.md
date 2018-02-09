# Traffic_Light_Recognition
1. Spatial Transformer Network with modified IDSIA network with Global normalization and batch normalization. Added Data Augmentation as part of transformation code in PyTorch. 99.4%
Link: http://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
Papers: Traffic Sign Recognition with Multi-Scale Convolutional Networks http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
2. Modified IDSIA Conv Net with batch normalization and preprocessing  98.6%
Link: http://people.idsia.ch/~juergen/nn2012traffic.pdf
3. Densenet 121 : Did not perform as well as other networks. Accuracy was around 90 %
4. Experimented with various kernel combinations for convolution layers and varied number of convolution layers.
The experimented code is attached along.
Types of Preprocessing:
1. Global Normalization: subtracting mean and division by standard deviation considering all the images
Added Jittering
1. Followed Yanns and Sermanet paper on traffic signal classification and added random jittering  that is random scaling / transformation / rotation
2. Random Cropped the images and center cropped as part of transformations
Batch Normalization:
Paper: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
Link: https://arxiv.org/abs/1502.03167
Hyperparameters Tuning:
1. Learning Rate: Tried learning rate in a range of 0.0005  0.01
2. Experimented with optimizers: Used SGD, ADAM optimizer


2. Modified Spatial Transformer Network (PyTorch) with above network as normal forward
network
Spatial Transformer Network has 3 main components:
 The localization network is a regular CNN which regresses the transformation parameters. The
 transformation is never learned explicitly from this dataset, instead the network learns
 automatically the spatial transformations that enhances the global accuracy.
  The grid generator generates a grid of coordinates in the input image corresponding to each pixel
  from the output image.
   The sampler uses the parameters of the transformation and apply it to the input image.

