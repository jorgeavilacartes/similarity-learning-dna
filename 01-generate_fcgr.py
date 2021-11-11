from pathlib import Path
from parameters import PARAMETERS
from similarity_learning_dna import (
    GenerateFCGR,
)

print(">> generate fcgr <<")

FOLDER_FASTA = Path(PARAMETERS["FOLDER_FASTA"]) 
LIST_FASTA   = list(FOLDER_FASTA.rglob("*fas"))
KMER = PARAMETERS["KMER"] 
generate_fcgr = GenerateFCGR(destination_folder="img-{}-mer".format(KMER),kmer=KMER)
generate_fcgr(list_fasta=LIST_FASTA,)