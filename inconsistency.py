import pickle 
from tqdm import tqdm
import numpy as np
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
import nltk
import pandas as pd
import os 
import sys 
from utils import *


# define token based minimality as it is defined in MiCE and Polyjuice
# from the code of MiCE:
# https://github.com/allenai/mice/blob/d70440b0fa8f8ded145abe3a99ce7cfe785c4e7f/src/edit_finder.py#L85
def score_minimality(orig_sent, edited_sent, normalized=False):
    spacy = SpacyTokenizer()
    tokenized_original = [t.text for t in spacy.tokenize(orig_sent)]
    tokenized_edited = [t.text for t in spacy.tokenize(edited_sent)]
    lev = nltk.edit_distance(tokenized_original, tokenized_edited)
    if normalized:
        if len(tokenized_original)==0:
            return 0 
        return lev/len(tokenized_original)
    else:
        return lev
    
    
def inconsistency(all_edits, filename = None):
    inco = np.ones((len(all_edits),10 - 2)) * -1
    for es in tqdm(range(len(all_edits))): 
        edit_sequence = all_edits[es]
        for step in range(0, 10-2):
            if len(edit_sequence) > step + 2:
                    t1 = edit_sequence[step][0]
                    t2 = edit_sequence[step+1][0]
                    t3 = edit_sequence[step+2][0]
                    

                    d12 = score_minimality(t1,t2)
                    d23 = score_minimality(t2,t3)
                    
                    c = d23 - d12
                    if c >0.0:
                        inco[es,step] = c
                    else:
                        inco[es,step]=0.0

    inconsistencies = {(step+1): [i for i in inco[:, step] if i != -1] for step in range(10-2)}
    if filename:
        save_pickle(inconsistencies, filename)
    return inconsistencies


if __name__ == "__main__":
    data_path = sys.argv[1]
    result_path = sys.argv[2]

    filenames = [os.path.join(data_path, dataset, f"{dataset}_{editor}.pickle") for editor in ["polyjuice", "mice", "textfooler"] for dataset in ["imdb", "newsgroups"]]
    
    for filename in filenames:
        print (f"Calculates Inconsistency for: {filename}")
        edits = load_pickle(filename)
        
        out_filename = filename.split("/")[-1].split(".")[0]
        out_filename = os.path.join(result_path, f"{out_filename}_inconsistency.pickle")
        minins = inconsistency(edits, out_filename)



    