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
The main model works with the join of a CNN and two Transformes working as encoder and decoder, the map feature given by the CNN and input of the Transformer encoder is (añadir valores) and the parameter given by both transformers are able to change but with the trained models we got that the output embedding from the Transformer encoder is as embedding dimensions of 512 and in the feed-forward network dimension of 512 and the output of the Transformer decoder are one-hot vectors of size of the vocabulary (añadir tamaño vocabulario - no me acuerdo).

In our best model we could get:
BLEU-1  -   0.23
BLEU-2   -   0.18
ROUGE-L   -   0.28
METEOR   -   0.24
CER   -   0.76
CHR_F   -   0.28

## Metrics definitions

#### BLEU
Accuracy based on n-grams, in our case we use BLEU-1 (unigrams) and BLEU-2 (bigrams).

#### ROUGE
Measure the longest subsequence of words (sequences in order but not necessarily conecutively), in our case we use ROUGE-L that takes recall in count too.

#### METEOR
Measures the amount of candidates of unigrams and its reference related to the predicted (kind of a wordle).

#### CER
Metric at character level based on substitution, delation and insertion.

#### CHR_F
bla bla bla

## Other methods (trained embeddings)
We tried with models that involves trained embeddings that does not provide better results, they are also in this github and are two models that uses CNN and Inception_v3 (pretrained model) respectevly as encoders and a GRU as decoder in both models.
This models where tried with two different embeddings, Word2Vec as a simple word embedding and FastText based in syllables, this last designed for small corpus training since the corpus are the captions around 10 words each (the remaining spaces were added as "<pad>").

We have not provided that much models from this section since does not provide any substancial change, only the Inception model with a Word2Vec embedding.
