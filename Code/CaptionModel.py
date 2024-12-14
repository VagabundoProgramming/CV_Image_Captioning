# This file contains everything related with the captioning model loading
# training as well as image and text loading and preprocessing

import csv
import evaluate
import keras
from keras import layers
from keras.applications import efficientnet
from keras.layers import TextVectorization
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import re
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"


### Data Loading ###

def load_captions_data(filename):
    """Loads captions (text) data and maps them to corresponding images.

    Args:
        filename: Path to the text file containing caption data.

    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """

    filename = f"archive/{filename}"

    with open(filename, newline='', encoding='utf-8') as csvfile:

        caption_mapping = {}
        text_data = []
        images_to_skip = set()
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row

        for row in reader:
            # Assuming columns: ..., title (col 2), ..., ..., image_name (col 5)
            caption = row[1].strip()  # Second column: title
            img_name = row[4].strip()  # Fifth column: image_name

            # Adjust the path for the images if needed
            img_name = os.path.join(IMAGES_PATH, img_name+".jpg")

            # Tokenize caption and apply length constraints
            tokens = caption.split()
            if len(tokens) < MIN_SEQ_LENGTH or len(tokens) > MAX_SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue
            images_to_skip.add("pan-seared-salmon-on-baby-arugula-242445.jpg")
            # Add start and end tokens to the caption
            

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                caption = f"<start> {caption} <end>"
                text_data.append(caption)
                if img_name in caption_mapping:
                    #caption_mapping[img_name].append(caption)
                    pass
                else:
                    caption_mapping[img_name] = [caption]
                

         # Remove any images marked for skipping
    for img_name in images_to_skip:
        if img_name in caption_mapping:
            del caption_mapping[img_name]

    return caption_mapping, text_data

# Divide the dataset into training, test and validation
def data_split(caption_data, train_size=0.8, test_size = 0.1, shuffle=True):
    assert train_size+test_size<=1, "The fraction of test and train must not surpass 1"
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        test_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning, test and validation datasets as three separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)
    test_size = int(len(caption_data) * test_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    test_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:train_size+test_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size+test_size:]
    }

    # 4. Return the splits
    return training_data, test_data, validation_data


### Text and Image Preprocessing

def custom_standardization(input_string):
    """ Given a string, modify it for proper processing

    Args: 
        A string of characters
    Returns:
        A string in lowercase without ceratin characters
    """

    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")

    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def get_vectorization(text_data):
    """ Create a vectorization layer based on the text data

    Args: 
        text_data: a list with all the preprocessed captions
    Returns:
        A layer capable of converting text into numerical data
    """

    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SEQ_LENGTH,
        standardize=custom_standardization,
    )
    vectorization.adapt(text_data)
    return vectorization

def get_image_aug():
    """ Creates a layer used for image augmentation 

    Returns:
        A sequential layer used for image augmentation
    """
    image_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomContrast(0.3),
        ]
    )
    return image_augmentation

def decode_and_resize(img_path):
    """ Given an image path, obtain the file as a variable

    Args:
        A path to an image file
    Returns:
        The image loaded into a variable
    """

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_path, captions, vectorization):
    """ Given an image path and its captions preprocess them

    Args: 
        Image path and captions
    Returns:
        The processed image and captions
    """
    

    return decode_and_resize(img_path), vectorization(captions)


def make_datasetpre(images, captions, vectorization):
    """ Given a set of images and captions build a dataset
    Args:
        A list of images and a list of captions of same lenght
    Returns:
        The dataset of the provided images and captions
    """

    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

def make_dataset(images, captions, vectorization):
    """ Given a set of images and captions build a dataset
    Args:
        A list of images and a list of captions of same lenght
    Returns:
        The dataset of the provided images and captions
    """

    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map((lambda x, y: process_input(x, y, vectorization)), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

### Model ###
def get_cnn_model():
    """ Creates a CNN model based on imagenet

    Returns:
        CNN model
    """
    
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # We freeze our feature extractor
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


class TransformerEncoderBlock(layers.Layer):
    """ A class holding the encoder of the transformer """

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        """ Initialises the layers of the encoder based on the inputs
        Args: 
            Embedding dimension: size of the word embeddings
            Dense dimension: size of the , number of heads and extra args.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training, mask=None): # May we delete mask here?
        """ Given a batch of inputs returns the result of the encoder
        Args:
            Inputs (data to analize), training (wheter we are training the model or not)
        Returns:
            The inputs after passing through the encoder
        """
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1


class PositionalEmbedding(layers.Layer):
    """ A class holding the positional embeddings for the decoder """
    
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    """ This class holds teh decoder of the transformer """

    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM,
            sequence_length=MAX_SEQ_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.constant([1, 1], dtype=tf.int32),
            ],
            axis=0,
        )
        return tf.tile(mask, mult)


class ImageCaptioningModel(keras.Model):
    """ This class holds the image captioning model
    Which is composed of the cnn, encoder and decoder """
    
    def __init__(
        self,
        cnn_model,
        encoder,
        decoder,
        num_captions_per_image=1,
        image_aug=None,
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_seq[:, i, :], training=True
                )

                # 3. Update loss and accuracy
                batch_loss += loss
                batch_acc += acc

            # 4. Get the list of all the trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # 5. Get the gradients
            grads = tape.gradient(loss, train_vars)

            # 6. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        # 7. Update the trackers
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 8. Return the loss and accuracy values
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_seq[:, i, :], training=False
            )

            # 3. Update batch loss and batch accuracy
            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)

        # 4. Update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 5. Return the loss and accuracy values
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]


class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """ Defines the Learning Rate Scheduler for the optimizer """
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )

### Save & Load Functions

def save_model(model, filepath):
    """ Saves the nodes weights inside a file
    Args: 
        The model to extract the weights from and the filepath to store the data on    
    """
    
    model_weights = model.get_weights()
    with open(filepath, 'wb') as fp:
        pickle.dump(model_weights, fp)
    return

def load_model(model, filepath):
    """ Given a model and a path to a file holding the model's weights load them
    Args: 
        the model to load data into, the filepath containing the weights
    """

    with open(filepath, 'rb') as f:     
        model.set_weights(pickle.load(f))
    return

### Standard model parameters ###
# Path to the images
IMAGES_PATH = "archive\Food Images\Food Images"

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
MAX_SEQ_LENGTH = 25
MIN_SEQ_LENGTH = 1

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE
