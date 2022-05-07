"""
SET PARAMETERS FOR ALL STEPS
"""
import yaml

with open("parameters.yaml") as fp: 
    PARAMETERS = yaml.load(fp, Loader=yaml.FullLoader)

# # -- Define parameters

# # Undersample sequences
# PATH_METADATA = "/data/GISAID/metadata.tsv"
# CLADES = ['S','L','G','V','GR','GH','GV','GK','GRY','O','GRA']
# SAMPLES_PER_CLADE = 5000
# PATH_FASTA_GISAID = "/data/GISAID/sequences.fasta"

# # General
# KMER = 6
# SPECIE = "hCoV-19"
# # For training
# TRAIN_SIZE = 0.8 # size for val and test sets = (1-TRAIN_SIZE)/2
# BATCH_SIZE = 32
# EPOCHS = 50

# # ---------------------
# # Load to a Dictionary
# PARAMETERS = dict(
#     KMER = KMER,
#     PATH_METADATA = PATH_METADATA,
#     CLADES = CLADES,
#     SAMPLES_PER_CLADE = SAMPLES_PER_CLADE,
#     PATH_FASTA_GISAID=PATH_FASTA_GISAID,
#     SPECIE = SPECIE,
#     FOLDER_FASTA = f"data/{SPECIE}",
#     FOLDER_FCGR = f"data/fcgr-{KMER}-mer/{SPECIE}",
#     TRAIN_SIZE = TRAIN_SIZE,
#     BATCH_SIZE = BATCH_SIZE,
#     EPOCHS = EPOCHS,       
# )