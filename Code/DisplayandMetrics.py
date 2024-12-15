# This file contains the code needed to visualize 
# and evaluate the model

import evaluate
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from CaptionModel import MAX_SEQ_LENGTH
from CaptionModel import decode_and_resize

### Displaying random images with their captions

def display_random_caption(model, data, vectorization):
    """ Display captions given a model and a random image from the data.
    Args: 
        model : the model which we base our predictions on
        data : a dictionary containing as keys the path to images and
               as values lists containing the captions as one string
    """

    # Load relevant data
    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = MAX_SEQ_LENGTH - 1
    valid_images = list(data.keys())

    # Select a random image from the validation dataset
    sample_img = np.random.choice(valid_images)

    # Store the original caption
    gt_caption = data[sample_img][0]
    gt_caption = re.sub("<start> ", "", gt_caption)
    gt_caption = re.sub(" <end>", "", gt_caption)
    gt_caption = gt_caption.lower()

    # Read the image from the disk
    sample_img = decode_and_resize(sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print(f"Predicted Caption: {decoded_caption}", )
    print(f"Original Caption: {gt_caption}")


### Metrics ###

def load_metrics():
    """ Loads and returns the metrics for evaluating our model

    Returns:
        bleu, rouge and meteor metrics in that order
    """

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    cer = evaluate.load("cer")
    #nist_mt = evaluate.load("nist_mt")
    chrf = evaluate.load("chrf")

    return bleu, rouge, meteor, cer, chrf 


def metrics(model, data, bleu, rouge, meteor, cer, chrf, vectorization):
    """ Generates the metrics data to be read with other functions
    based on the data given
    
    Args: 
        Caption model : the transformer model
        data : a dictionary with file path as keys and a list with the caption as value. 
        bleu : the bleu metric
        rouge :  the rouge metric
        meteor : the meteor metric
        cer : the cer metric
        chrf : the chrf metric
        vectorization : TextVectorization object, obtained from get_vectorization()
    Returns:
        A dictionary with as keys the metrics name and as value a 
        list of results for each image in the data
    """
    
    # Load relevant data
    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = MAX_SEQ_LENGTH - 1
    valid_images = list(data.keys())

    # Create somewhere to store the metrics
    metrics_data = {"BLEU-1" : [], "BLEU-2" : [], "ROUGE-L" : [], "METEOR" : [], "CER" : [], "CHRF" : []}
    ref, pred = [], []

    # For each image in the data provided
    for sample_img in valid_images:

        # Store the original caption
        gt_caption = data[sample_img][0]
        gt_caption = re.sub("<start> ", "", gt_caption)
        gt_caption = re.sub(" <end>", "", gt_caption)
        gt_caption = gt_caption.lower()

        # Read the image from the disk
        sample_img = decode_and_resize(sample_img)
        img = sample_img.numpy().clip(0, 255).astype(np.uint8)

        # Pass the image to the CNN
        img = tf.expand_dims(sample_img, 0)
        img = model.cnn_model(img)

        # Pass the image features to the Transformer encoder
        encoded_img = model.encoder(img, training=False)

        # Generate the caption using the Transformer decoder
        decoded_caption = "<start> "
        for i in range(max_decoded_sentence_length):
            tokenized_caption = vectorization([decoded_caption])[:, :-1]
            mask = tf.math.not_equal(tokenized_caption, 0)
            predictions = model.decoder(
                tokenized_caption, encoded_img, training=False, mask=mask
            )
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = index_lookup[sampled_token_index]
            if sampled_token == "<end>":
                break
            decoded_caption += " " + sampled_token

        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()

        # Store references and predictions for later 
        ref, pred = [[gt_caption]], [decoded_caption]
        
        # Transform data and make predictions
        metrics_data["BLEU-1"].append(bleu.compute(predictions=pred, references=ref, max_order=1)["bleu"])
        metrics_data["BLEU-2"].append(bleu.compute(predictions=pred, references=ref, max_order=2)["bleu"])
        metrics_data["ROUGE-L"].append(rouge.compute(predictions=pred, references=ref)["rougeL"])
        metrics_data["METEOR"].append(meteor.compute(predictions=pred, references=ref)["meteor"])
        metrics_data["CER"].append(cer.compute(predictions=pred, references=ref[0]))
        #metrics_data["NIST-MT"].append(nist_mt.compute(predictions=pred, references=ref[0])["nist_mt"])
        metrics_data["CHRF"].append(chrf.compute(predictions=pred, references=ref)["score"])

    # Return after all calculations
    return metrics_data



def read_metrics(metrics_data):
    """ Prints statsitics of the metrics_data

    Args:
        metrics_data obtained from the metrics function
    """

    for key, values in metrics_data.items():
        print(f"Reading {key} data")
        print(f"Average acc: {sum(values)/len(values)}")
        print(f"Standard deviation: {np.std(values)}")
        print(f"Max accuracy: {max(values)}")
        print(f"Min accuracy: {min(values)}")
        print(f"Quantile information:")
        for x in range(0, 11, 1):
            print(f"Quantile {x} average: {np.quantile(values, x/10)}")
        print("")
