import numpy as np
import pandas as pd
import cv2
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import nltk
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer

#------------------- Relevant parameters ---------------------
# Change as needed
seq_length = 10 # desired length of the predicted phrase
tokenizer = RegexpTokenizer(r'\w+') # personalized tokenizer to only grab word (extracts punctuations)
#----------------------------------------------------------

assert torch.cuda.is_available(), "GPU is not enabled"

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------------ Main functions ---------------------------------

def split_data(csv_file, density=(0.75, 0.125, 0.125)):
    df = pd.read_csv(csv_file)[["Title", "Image_Name"]]
    df = df.dropna()
    not_valid = []
    for path in "archive/Food Images/" + df["Image_Name"] + ".jpg":
        img = cv2.imread(path)
        if img is None:
            not_valid.append(path[20:-4])
    df = df[~df["Image_Name"].isin(not_valid)]
    train = df[:int(len(df)*density[0])]
    val = df[int(len(df)*density[0]):int(len(df)*sum(density[:2]))]
    test = df[int(len(df)*sum(density[:2])):int(len(df)*sum(density))]
    return train, val, test

def create_embedding(dataset, vector_size=50, window=3, min_count=1, sg=1, workers=4, epochs=300):
    sentences = []
    for caption in dataset:
        caption = caption.lower() # we want lower phrases to reduce the amount of variants words
        caption = tokenizer.tokenize(caption)
        caption = caption + ["<pad>"] * (seq_length - len(caption)) # if the caption does not has the desired length fill the remaining places with "<pad>"
        sentences.append(caption)
    embedding = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=workers, epochs=epochs) # Train a Word2Vec type embedding
    # Add <unk> vector to the embedding as a zero vector
    embedding.wv.add_vector("<unk>", np.full(50, 0.0000001, dtype="float32")) # Due to divisions by zero we replace the absolute 0 to a almost near to 0
    return embedding

def train(model, loader, optimizer, criterion, epoch):
    loss = 0
    model.train()

    for images, captions in loader:
        # load it to the active device
        images = images.to(device)
        captions = captions.to(device)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs = model(images)
        
        # compute training reconstruction loss
        train_loss = criterion(outputs, captions)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(loader)
    print("epoch : {}/{}, Train loss = {:.6f}".format(epoch + 1, epochs, loss))

def test(model, loader, criterion, epoch):
    loss = 0
    model.eval()
    
    for images, captions in loader:
        images = images.to(device)
        captions = captions.to(device)

        with torch.no_grad():
            outputs = model(images)
        
        # compute training reconstruction loss
        test_loss = criterion(outputs, captions)
 
        # add the mini-batch training loss to epoch loss
        loss += test_loss.item()
    
    # compute the epoch test loss
    loss = loss / len(loader)
    
    # display the epoch training loss
    print("epoch : {}/{}, Test loss = {:.6f}".format(epoch + 1, epochs, loss))

class Image_Captioning(Dataset):
    def __init__(self, embedding, dataframe, image_fold, image_resize=(299, 299)):
        dataframe = dataframe.dropna()
        self.img_paths = image_fold + "/Food Images/" + dataframe["Image_Name"] + ".jpg"
        self.img_paths = self.img_paths.to_list()
        self.image_resize = image_resize
        self.captions = [tokenizer.tokenize(caption)[:seq_length] + ["<pad>"] * max(0, seq_length - len(tokenizer.tokenize(caption))) for caption in dataframe["Title"].to_list()]
        for i, caption in enumerate(self.captions):
            phrase = []
            for word in caption:
                word = word.lower()
                # This if works only for val and test data, since the embedding is trained with the train data, otherwise something is going wrong
                if word in embedding.wv:
                    word = embedding.wv[word]
                else:
                    word = embedding.wv["<unk>"] # In case we don't have the word embedded
                phrase.append(word)
            self.captions[i] = phrase     
        self.captions = np.stack(self.captions, 0) # Stack the captions to make it a single array (not a list of arrays)
        self.captions = torch.from_numpy(self.captions).to(dtype=torch.float)
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, key):
        img = cv2.imread(self.img_paths[key])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_resize)
        img = np.transpose(img, (2, 1, 0)) / 255
        return torch.from_numpy(img).to(dtype=torch.float), self.captions[key]

class CNN_GRU(nn.Module):
    def __init__(self, embedding_size):
        super(CNN_GRU, self).__init__()
        self.embedding_size = embedding_size
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, 5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6272, 1024),
            nn.ReLU()
        )
        # decoder
        self.gru = nn.GRU(1024, 1024, num_layers=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(seq_length*1024, seq_length*embedding_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # encoder
        x = self.encoder(x)
        # decoder
        word = []
        hidden = None
        # for each loop save the output, each output would represent a word (but in a higher dimension)
        for i in range(seq_length):
            x, hidden = self.gru(x, hidden)
            word.append(x)
        x = torch.stack(word, 1)
        x = self.relu(x)
        x = torch.reshape(x, (-1, 10240)) # resize to a linear function (to operate over all words)
        x = self.linear(x)
        x = self.tanh(x)
        x = torch.reshape(x, (-1, seq_length, self.embedding_size)) # resize to the expected output (batch_size, seq_length, embedding_size)
        return x
