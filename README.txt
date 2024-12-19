# Image Captioning
# Medical-Imaging
This is a project from the Computer Vision class at UAB University.

## Task Description
The objective of this task is to detect helicobacter pylori in patients and 
classify them as either infected or healthy using their medical images.

For this task we have implemented different models to different degrees of 
efficiency and accuracy.

## Git Description
Inside the code folder you will find the files used for this project.
These include the models labeled accordingly, files to evaluate their
accuracy and to load data. 

Inside the Models directory we have some models we have made so you can 
try them yourself without having to train yourself.

Unfortunately we do not provide the QUIRON dataset used for this project 
due to limitations.

## Autoencoder Model
An Autoencoder trained with healthy patches so that the reconstruction 
error on infected images is significant enough so that it can be 
detected as so. It is a symmetrical model.It is composed of 4 convolutional 
layers, 4 max pooling layers and the corresponding 4 transpose convolutional 
layers and 4 unpooling layers. 

The best patch accuracy is 86.4%.<br>
The best patch F1 Score is 44.7%.

The best patient diagnosis accuracy is 60.3%.<br>
The best patient diagnosis F1 Score is 67.1%.

## Classifier
A simple convolutional neural network. 
It makes use of three convolutional layers and a linear layer at the end. 
It is complemented with several pooling layers to simplify the result.
It uses ReLU as the activation function.

The model is able to classify inidividual patches at a 96.8 % accuracy. 
Then we use a threshold to determine what percentage of images must be
considered infected to declare the patient as so. To obtain the best
threshold we have made use of the ROC curve matric. 

The model has an accuray of 83% in patient diagnosis.

## "A ojo" Model
A simple algorithm that reads the image pixels row by row and if a severe difference
is detected between a pixel and the next it is classified as infected. This difference
is defined as a modifiable threshold.

At its best threshold value it holds a 88% accuracy.
