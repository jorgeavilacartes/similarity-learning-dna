import json
import pandas as pd
from parameters import PARAMETERS
from pathlib import Path
from similarity_learning_dna import DataSelector

print(">> train val test split <<")

# Select all data
KMER = PARAMETERS["KMER"]
SPECIE = PARAMETERS["SPECIE"]
FOLDER_FCGR = Path(f"data/fcgr-{KMER}-mer/{SPECIE}")
LIST_FASTA   = list(FOLDER_FCGR.rglob("*npy"))
TRAIN_SIZE   = float(PARAMETERS["TRAIN_SIZE"]) 
PATH_SAVE = Path("data/train/")
PATH_SAVE.mkdir(exist_ok=True, parents=True)

# Input for DataSelector
id_labels = [str(path) for path in LIST_FASTA]
labels    = [path.parent.stem for path in LIST_FASTA]

# Instantiate DataSelector
ds = DataSelector(id_labels, labels)

# Get train, test and val sets
ds(train_size=TRAIN_SIZE, balanced_on=labels)

with open(PATH_SAVE.joinpath("datasets.json"), "w", encoding="utf-8") as f: 
    json.dump(ds.datasets["id_labels"], f, ensure_ascii=False, indent=4)

# Summary of data selected 
summary_labels =  pd.DataFrame(ds.get_summary_labels())
summary_labels["Total"] = summary_labels.sum(axis=1)
summary_labels.to_csv(PATH_SAVE.joinpath("summary_labels.csv"))