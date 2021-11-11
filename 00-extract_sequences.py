# After undersample the sequences, we need to extract each one in a separated fasta file
# this is just to not modify the rest of the code that generates de FCGR, but eventually
# we can just read the sequences from the fasta from GISAID that contains all the sequences
from Bio import SeqIO
from pathlib import Path
import pandas as pd

from parameters import PARAMETERS

PATH_FASTA_GISAID = PARAMETERS["PATH_FASTA_GISAID"]
SPECIE = PARAMETERS["SPECIE"]
FOLDER_FASTA = Path(PARAMETERS["FOLDER_FASTA"]) # here will be saved the selected sequences by clade
FOLDER_FASTA.mkdir(parents=True, exist_ok=True)

# load fasta_id to save
undersample = pd.read_csv("undersample_by_clade.csv").to_dict("records")
set_fasta_id = set([record.get("fasta_id") for record in undersample])

# Read fasta with all sequences from GISAID
with open(PATH_FASTA_GISAID) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        
        # save sequence if it was selected
        if record.id in set_fasta_id:
            # save sequence in a fasta file "<accession_id>.fasta"
            filename = record.id.replace("/","_")
            path_save = FOLDER_FASTA.joinpath(f"{filename}.fasta")
            SeqIO.write(record, path_save, "fasta") 
            # remove from the set to be saved   
            set_fasta_id.remove(record.id)
        
        # if all sequences has been saved, break the loop
        if not set_fasta_id:
            break