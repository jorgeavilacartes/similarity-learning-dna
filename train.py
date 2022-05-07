import json
from pathlib import Path
from similarity_learning_dna import (
    ModelLoader,
    DataGenerator,    
)
from similarity_learning_dna.callbacks import CSVTimeHistory
from parameters import PARAMETERS
import tensorflow as tf

print(">> train model <<")

# General parameters
KMER = PARAMETERS["KMER"]
CLADES = PARAMETERS["CLADES"]

# Train parameters
BATCH_SIZE = PARAMETERS["BATCH_SIZE"]
EPOCHS     = PARAMETERS["EPOCHS"]
WEIGHTS    = PARAMETERS["WEIGHTS"]
PATIENTE_EARLY_STOPPING = PARAMETERS["PATIENTE_EARLY_STOPPING"]
PATIENTE_EARLY_LR = PARAMETERS["PATIENTE_REDUCE_LR"]

# -1- Model selection
loader = ModelLoader()
model  = loader(
    "cnn_{}mers".format(KMER),
    weights_path=WEIGHTS,
    ) # get compiled model from ./similarity_learning_dna/models

# -2- Datasets
# load list of images for train and validation sets
with open("data/train/datasets.json","r") as f:
    datasets = json.load(f)

list_train = datasets["train"]
list_val   = datasets["val"]

# Instantiate DataGenerator for training set
ds_train = DataGenerator(
    list_train,
    order_output_model = CLADES,
    batch_size = BATCH_SIZE,
    shuffle=True
)

# Instantiate DataGenerator for validation set
ds_val = DataGenerator(
    list_val,
    order_output_model = CLADES,
    batch_size = BATCH_SIZE,
    shuffle=False,
) 

# -3- Training
# - Callbacks
# checkpoint: save best weights
Path("data/train/checkpoints").mkdir(exist_ok=True, parents=True)
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='data/train/checkpoints/weights-{epoch:02d}-{val_loss:.3f}.hdf5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

# reduce learning rate
cb_reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.1,
    patience=PATIENTE_EARLY_LR,
    verbose=1,
    min_lr=0.00001
)

# stop training if
cb_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    min_delta=0.001,
    patience=PATIENTE_EARLY_STOPPING,
    verbose=1
)

# save history of training
Path("data/train").mkdir(exist_ok=True, parents=True)
cb_csvlogger = tf.keras.callbacks.CSVLogger(
    filename='data/train/training_log.csv',
    separator=',',
    append=False
)

cb_csvtime = CSVTimeHistory(
    filename='data/train/time_log.csv',
    separator=',',
    append=False
)

model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=[
        cb_checkpoint,
        cb_reducelr,
        cb_earlystop,
        cb_csvlogger,
        cb_csvtime
        ]
)