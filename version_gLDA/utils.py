# utils
# write some helper functions for Gaussian LDA


# import numpy as np
# import matplotlib.pyplot as plt
import pickle
# import random
# from sklearn.manifold import TSNE
# from numpy import linalg as LA

# helper function to write any type of data to output file
def obj_writer(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

# helper function to load data from the output file generated from obj_writer
def obj_reader(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding="bytes")

