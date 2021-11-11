import random
from collections import namedtuple
from tqdm import tqdm
import pandas as pd

from parameters import PARAMETERS, SAMPLES_PER_CLADE

tqdm.pandas()
random.seed(42)

PATH_METADATA = PARAMETERS["PATH_METADATA"]
CLADES = PARAMETERS["CLADES"]
SAMPLES_PER_CLADE = PARAMETERS["SAMPLES_PER_CLADE"]

# Load metadata
COLS = ["Virus name", "Accession ID", "Collection date", "Clade", "Host", "Is complete?"]
data = pd.read_csv(PATH_METADATA,sep="\t")

# Remove NaN in Clades and not-complete sequences
data.dropna(axis="rows",
            how="any",
            subset=["Is complete?", "Clade"], 
            inplace=True,
            )

# Filter by Clades and Host
CLADES = tuple(clade for clade in CLADES)
data.query(f"`Clade` in {CLADES} and `Host`=='Human'", inplace=True)

## Randomly select a subset of sequences
# Generate id of sequences in fasta file: "Virus name|Accession ID|Collection date"
data["fasta_id"] = data.progress_apply(lambda row: "|".join([row["Virus name"],row["Accession ID"], row["Collection date"]]), axis=1)

# subsample 
SampleClade = namedtuple("SampleClade", ["fasta_id","clade"])
list_fasta_selected = []
for clade in tqdm(CLADES):
    samples_clade = random.choices(population=data.query(f"`Clade` == '{clade}'")["fasta_id"].tolist(), k=SAMPLES_PER_CLADE)
    list_fasta_selected.extend([SampleClade(fasta_id, clade) for fasta_id in samples_clade])

pd.DataFrame(list_fasta_selected).to_csv("undersample_by_clade.csv")