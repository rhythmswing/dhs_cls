
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from argparse import ArgumentParser
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)



argparse = ArgumentParser()
argparse.add_argument("--input", type=str, required=True)
argparse.add_argument("--output", type=str, required=True)
argparse.add_argument("--batch_size", type=int, default=32)
argparse.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
argparse.add_argument("--max_length", type=int, default=50)
argparse.add_argument("--index_file", type=str, default=None)


argparse = argparse.parse_args()

master_dataset = pd.read_feather(argparse.input)
output_handle = open(argparse.output, "w")


if argparse.index_file:
    with open(argparse.index_file, "r") as index_file:
        indices = np.array([int(line) for line in index_file])
else:
    indices = np.arange(len(master_dataset))

sequences = master_dataset["sequence"].values[indices].tolist()

from tqdm import tqdm

model = model.to(argparse.device)

for sequence_batch in tqdm(range(0, len(sequences), argparse.batch_size)):
    batch = sequences[sequence_batch:sequence_batch + argparse.batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=argparse.max_length)
    inputs = {key: value.to(argparse.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs[0].mean(dim=1).detach().cpu().numpy()
    for embedding in embeddings:
        output_handle.write(" ".join(map(str, embedding)) + "\n")