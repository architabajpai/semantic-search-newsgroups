import numpy as np
import pickle

def get_gmm():
    with open("data/gmm.pkl", "rb") as f:
        return pickle.load(f)

def get_cluster_probs():
    return np.load("data/cluster_probs.npy")
 
