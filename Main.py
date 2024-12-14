# This file contains the execution and loading of the model


# Library Imports
import keras

# File Imports
from CaptionModel import *
from DisplayandMetrics import *

# Set random seed
keras.utils.set_random_seed(111)


    ### Set model parameters ###

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
EPOCHS = 2
AUTOTUNE = tf.data.AUTOTUNE

training = True


    ### Load Data ###
# Load the dataset
captions_mapping, text_data = load_captions_data("Food Ingredients and Recipe Dataset with Image Name Mapping.csv")

# Split the dataset into training and validation sets
train_data, test_data, valid_data = data_split(captions_mapping)


    ### Data Preprocessing ###

# Create the vectorization layer
vectorization = get_vectorization(text_data)

# Data augmentation for image data
image_augmentation = get_image_aug()

train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()), vectorization)
test_dataset = make_dataset(list(test_data.keys()), list(test_data.values()))
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))


    ### Create the model ###

cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=image_augmentation,
)

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction=None,
)

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Create a learning rate schedule
num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

# Compile the model
caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

# Fit the model
if training:
    caption_model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        callbacks=[early_stopping],
    )

else:
    load_model(caption_model, "saved_models/caption_model_34.ob")

for x in range(0,3,1):
    display_random_caption(caption_model, valid_dataset, vectorization)