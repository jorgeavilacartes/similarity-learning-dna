import io
import json
import numpy as np
import pandas as pd
from parameters import PARAMETERS
from similarity_learning_dna import (
    ModelLoader,
    DatasetLoader,
)

KMER = PARAMETERS["KMER"]
BATCH_SIZE = 8

# -1- Load model
loader = ModelLoader()
model  = loader("cnn_{}mers".format(KMER), weights_path="checkpoint/cp.ckpt") # get compiled model from ./supervised_dna/models

# -2- Datasets
# load list of images for train and validation sets
with open("datasets.json","r") as f:
    datasets = json.load(f)
list_test = datasets["val"]


ds_loader = DatasetLoader(batch_size=BATCH_SIZE, 
                            kmer=KMER, 
                            order_output_model=["1","2","3","4"],
                            shuffle=False,
                            )

ds_test = ds_loader(list_img = list_test)

# Save test embeddings for visualization in projector
results = model.predict(ds_test)
np.savetxt("vecs.tsv", results, delimiter='\t')

out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for img, labels in iter(ds_test):
    [out_m.write(str(label.parent.stem) + "\n") for label in list_test]
out_m.close()

# Evaluate model and save metrics
result = model.evaluate(ds_test)
pd.DataFrame(
    dict(zip(model.metrics_names, result)), index=[0]) \
        .to_csv("metrics_test.csv")