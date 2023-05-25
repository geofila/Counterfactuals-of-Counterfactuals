import pickle


# utility functions for loading and saving objects
def save_pickle(obj, filename):
    with open(filename,'wb') as f:
        pickle.dump(obj,f)
  
def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)