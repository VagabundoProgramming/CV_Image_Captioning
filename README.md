# Image Captioning
This is a project from the Computer Vision class at UAB University.

## Task Description
The objective of this task is to, with a given image, discribe its content and make a caption, in our case it would be based in Food and dishes.

For this task we have implemented different models to different degrees of efficiency and accuracy.

## Git Description
Inside the code folder you will find the files used for this project.
These include the models labeled accordingly, files to evaluate their
accuracy and to load data. 

Inside the Models directory we have some models we have made so you can 
try them yourself without having to train yourself.

Unfortunately we do not provide the Food dataset used for this project 
due to limitations.

## Main Model
The main model works with the join of a CNN and two Transformes working as encoder and decoder, the map feature given by the CNN and input of the Transformer encoder is (añadir valores) and the parameter given by both transformers are able to change but with the trained models we got that the output embedding from the Transformer encoder is as embedding dimensions of 512 and in the feed-forward network dimension of 512 and the output of the Transformer decoder are one-hot vectors of size of the vocabulary (añadir tamaño vocabulario - no me acuerdo)

Accuracy section

## Metrics definitions

#### BLEU

#### ROUGE-L

#### METEOR

#### CER

#### CHR_F

## Other methods
we tried different methods
