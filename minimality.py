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
    
    
# minimality for every step of an editor
def minimality(all_edits, filename = None):
    minimalities = {n: minimalityAt(all_edits,n) for n in tqdm(range(1,len(all_edits[0]))) }
    if filename:
        save_pickle(minimalities, filename)
    return minimalities 


# minimality for a single step 
def minimalityAt(all_edits, n=1):
    minis = list()
    for es in range(len(all_edits)):
        edit_sequence = all_edits[es]
        
        if len(edit_sequence) <= n:
            continue
        t1 = edit_sequence[n-1][0]
        t2 = edit_sequence[n][0]
#         if np.argmax(edit_sequence[n][1]) != np.argmax(edit_sequence[n-1][1]): # if the class is not the same 
        minis.append(score_minimality(t1,t2))
    return minis

    
if __name__ == "__main__":
    data_path = sys.argv[1]
    result_path = sys.argv[2]

    filenames = [os.path.join(data_path, dataset, f"{dataset}_{editor}.pickle") for editor in ["mice", "textfooler", "polyjuice"] for dataset in ["imdb", "newsgroups"]]
    
    for filename in filenames:
        print (f"Calculates Minimality for: {filename}")
        edits = load_pickle(filename)
        
        out_filename = filename.split("/")[-1].split(".")[0]
        out_filename = os.path.join(result_path, f"{out_filename}_minimality.pickle")
        minins = minimality(edits, out_filename)



    