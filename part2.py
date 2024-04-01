from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters):
    # Unpack the dataset into data and labels (even though labels are not used here)
    data, _ = dataset
    
    #Standardize scale with StandardScaler()
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    #Initialize kmeans
    kmeans = cluster.KMeans(n_clusters=n_clusters, init='random', random_state=42)
    
    #Fit data.
    kmeans.fit(data_std)
    #Use centroids and labels to calculate SSE. 
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    sse = 0
    for i in range(len(data_std)):
        dist = np.sum((data_std[i] - centroids[labels[i]])**2)
        sse += dist
    
    return sse


def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    blob = make_blobs(center_box=(-20,20), n_samples=20, centers=5,random_state=12)
    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = list(blob) 

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    k_values = [1,2,3,4,5,6,7,8]
    sse = []
    for value in k_values:
        sse.append(fit_kmeans(blob,value))
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse, marker='o')
    plt.title('Elbow Method (sse)')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('(SSE)')
    plt.xticks(k_values)
    plt.grid(True)
    #plt.show()
    #print(sse)
    optimal_k_sse = sse.index(min(sse))
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = [list(a) for a in zip(k_values,sse)]
    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    k_values = [1,2,3,4,5,6,7,8]
    sse = []
    for value in k_values:
        sse.append(u.fit_kmeans_inertia(blob,value))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse, marker='o')
    plt.title('Elbow Method (inertia)')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('(SSE)')
    plt.xticks(k_values)
    plt.grid(True)
    #plt.show()

    optimal_k_inertia = sse.index(min(sse)) 
    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = [list(a) for a in zip(k_values,sse)]
    # print(f"Are the two graph's optimal k equal? {True if optimal_k_sse == optimal_k_inertia else False}")
    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes" if optimal_k_sse == optimal_k_inertia else "no"
    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
