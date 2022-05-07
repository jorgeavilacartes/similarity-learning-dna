configfile: "parameters.yaml" 

rule all:
    input:
        "data/test/meta.tsv",
        "data/test/vecs.tsv"


## 1. Generate FCGR from fasta files
rule generate_fcgr:
    input: 
        expand("data/{specie}/extracted_sequences.txt", specie=config["SPECIE"])
    output:
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"]) 
    script: "fasta2fcgr.py"

## 2. split train, val, test sets
rule split_data:
    input:
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"]), 
    output: 
        "data/train/datasets.json",
        "data/train/summary_labels.csv"
    script: 
        "split_data.py"

## 3. train model
rule train_model:
    input: 
        "data/train/datasets.json",
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    output: 
        "data/train/training_log.csv"
    script: 
        "train.py"
    # shell:
    #     "python3 train.py > data/train/train.out 2 > data/train/train.err &" 

## 4. test model
rule test_model:
    input: 
        "data/train/datasets.json",
        "data/train/training_log.csv"
    output: 
        "data/test/meta.tsv",
        "data/test/vecs.tsv"
    script: "test.py"