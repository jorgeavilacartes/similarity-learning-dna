"""
SET PARAMETERS FOR ALL STEPS
"""
# -- Define parameters

# Undersample sequences
PATH_METADATA = "/data/GISAID/metadata.tsv"
CLADES = ['S','L','G','V','GR','GH','GV','GK']
SAMPLES_PER_CLADE = 1600
PATH_FASTA_GISAID = "/data/GISAID/sequences.fasta"

# General
KMER = 8
SPECIE = "hCoV-19"
# For training
TRAIN_SIZE = 0.8 # size for val and test sets = (1-TRAIN_SIZE)/2
BATCH_SIZE = 8
EPOCHS = 20

# ---------------------
# Load to a Dictionary
PARAMETERS = dict(
    KMER = KMER,
    PATH_METADATA = PATH_METADATA,
    CLADES = CLADES,
    SAMPLES_PER_CLADE = SAMPLES_PER_CLADE,
    PATH_FASTA_GISAID=PATH_FASTA_GISAID,
    SPECIE = SPECIE,
    FOLDER_FASTA = f"data/{SPECIE}",
    FOLDER_IMG = f"img-{KMER}-mer/{SPECIE}",
    TRAIN_SIZE = TRAIN_SIZE,
    BATCH_SIZE = BATCH_SIZE,
    EPOCHS = EPOCHS,       
)