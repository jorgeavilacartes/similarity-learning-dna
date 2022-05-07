import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
from parameters import PARAMETERS
from similarity_learning_dna import (
    ModelLoader,
    DataGenerator,
)

print(">> test model <<")
BATCH_SIZE = PARAMETERS["BATCH_SIZE"]
KMER = PARAMETERS["KMER"]

CLADES = PARAMETERS["CLADES"]
MODEL_NAME  = f"cnn_{KMER}mers"

# get best weights
CHECKPOINTS  = [str(path) for path in Path("data/train/checkpoints").rglob("*.hdf5")]
epoch_from_chkp = lambda chkp: int(chkp.split("/")[-1].split("-")[1])
CHECKPOINTS.sort(key = epoch_from_chkp)
BEST_WEIGHTS =  CHECKPOINTS[-1]
print(f"using weights {BEST_WEIGHTS} to test")


# -1- Load model
loader = ModelLoader()
model  = loader(
            model_name=MODEL_NAME, 
            weights_path=BEST_WEIGHTS) # get compiled model from ./simimilarity_learning_dna/models

# -2- Datasets
# load list of images for train and validation sets
with open("data/train/datasets.json","r") as f:
    datasets = json.load(f)
list_test = datasets["test"]


ds_test = DataGenerator(
    list_test,
    order_output_model = CLADES,
    shuffle=False
)

# Save test embeddings for visualization in projector
results = model.predict(ds_test)
np.savetxt("data/test/vecs.tsv", results, delimiter='\t')

# Incluir clades aqui
out_m = io.open('data/test/meta.tsv', 'w', encoding='utf-8')
for img, label in iter(ds_test):
    for l in label:
        clade = CLADES[int(l)]
        [out_m.write(str(clade) + "\n")]
out_m.close()
