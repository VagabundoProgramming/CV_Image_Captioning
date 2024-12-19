import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from my_funcs import *

import nltk
import gensim
from gensim.models import Word2Vec

#If this cell fails you need to change the runtime of your colab notebook to GPU
# Go to Runtime -> Change Runtime Type and select GPU
assert torch.cuda.is_available(), "GPU is not enabled"

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dividing the dataset into 3 different tipes (train, validation, test)
train_dataset, val_dataset, test_dataset = split_data("archive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv")

# loading pretrained embedding
embedding = Word2Vec.load("pretrained/embedding.kvmodel")
# Otherwise to train your own embedding:
# embedding = create_embedding(train_dataset["Title"].to_list())

# Creating data loader for each dataset
train_loader = DataLoader(Image_Captioning(embedding, train_dataset, "archive"), batch_size=32, shuffle=False)
val_loader = DataLoader(Image_Captioning(embedding, val_dataset, "archive"), batch_size=32, shuffle=False)
test_loader = DataLoader(Image_Captioning(embedding, test_dataset, "archive"), batch_size=32, shuffle=False)

model = Inception_GRU().to(device)
# model = CNN_GRU(50).to(device)

# Create an optimizer object
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Loss function
criterion = nn.MSELoss()

# change epochs to the amount of runs you want the model to train (each epoch runs through all the train_loader)
epochs = 1
model = model.to(device)
for epoch in range(epochs):
    train(model, train_loader, optimizer, criterion, epoch, epochs)
    test(model, val_loader, criterion, epoch, epochs)

images, captions = next(iter(test_loader))
model = model.to("cpu")
outputs = model(images)
for i in range(32):
    output_phrase = []
    true_phrase = []
    for u in range(10):
        output_word = embedding.wv.similar_by_vector(outputs[i][u].detach().numpy())[0][0]
        true_word = embedding.wv.similar_by_vector(captions[i][u].detach().numpy())[0][0]
        output_phrase.append(output_word)
        true_phrase.append(true_word)
    print(output_phrase)
    print(" ".join(true_phrase))
