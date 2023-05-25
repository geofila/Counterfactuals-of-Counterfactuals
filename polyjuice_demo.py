import os 
from polyjuice import Polyjuice
import sys
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
import Levenshtein


sys.path.append("mice") # code of mice from github for the predictor
from src.utils import load_predictor

def create_counterfactual(text):
        petrubations = pj.perturb(text, num_perturbations = 50)
        data = {
            "original_text": text,
            "original_pred": predictor.predict(text)["probs"],
            "perturbations": [{"text": p, "pred": predictor.predict(p)["probs"]}  for p in petrubations]
        }

        return data
    
predictor = load_predictor("newsgroups", predictor_folder = "../../mice/trained_predictors/") # or use your predictor

pj = Polyjuice(model_path="uw-hai/polyjuice", is_cuda=True) # load editor

with open("Edits/imdb_polyjuice.pickle", "rb") as handle:
    edits = pickle.load(handle) # load source texts of edits, must be in the same form as the output

for i in tqdm(range(len(edits))):
    row = edits[i]
    text = row[-1][0]

    for j in range (10): # 10 is the number of feedforward steps
        
        try:
            x = create_counterfactual(text)

            orig_input = x["original_text"]
            orig_label = np.argmax(x["original_pred"])

            min_dist = 100000000
            min_label = []
            counter_text = ""
            for petr in x["perturbations"]:
                pred_label = np.argmax(petr["pred"])
                if orig_label != pred_label:
                    cand_text = petr["text"]
                    dist = Levenshtein.distance(cand_text, orig_input) # the distance metric that we want to minimise e.g. Levenshtein
                    if dist < min_dist:
                        counter_text = cand_text
                        min_dist = dist
                        min_label = petr["pred"]

            edits[i].append([counter_text, min_label])
            text = counter_text
            print (j)
        except Exception as e:
            print (e)
            break
            
    with open("output_file_with_source_texts_and_text_for_10_steps.pickle", "wb") as handle: # output file 
        pickle.dump(edits, handle)
        
        