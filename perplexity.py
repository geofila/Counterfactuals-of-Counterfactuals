# calculate perplexity 
import sys
sys.path.append("disentanglement-vae")
sys.path.append("disentanglement-vae/scripts/evaluation")
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from compute_ppl import *
import numpy as np
from tqdm import tqdm 
import pickle
import math
import os


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    
dataset = sys.argv[1]
model_id = sys.argv[2]
data_path = sys.argv[3]
result_path = sys.argv[4]

tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id).to(DEVICE)


filenames = [os.path.join(data_path, dataset, f"{dataset}_{editor}.pickle") for editor in ["mice", "textfooler", "polyjuice"]]

for filename in filenames:
    print (f"Compute PPL for: {filename}")
    with open(filename, "rb") as handle:
        edits = pickle.load(handle)

    ppls = np.zeros((len(edits), 10))
    for phase in range (10):
        for i, e in tqdm(enumerate(edits)):
            if len(e) > phase:
                try:
                    ppls[i][phase] = compute_ppl(e[phase][0], tokenizer, model, stride=512, verbose=False)
                except:
                    pass
    
    
    out_filename = filename.split("/")[-1].split(".")[0]
    m = model_id.split("/")[-1]
    np.save(os.path.join(result_path, f"ppl_{out_filename}_{m}.npy"), ppls)
