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

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset,linkage,n_clusters):
    
    data,_ = dataset

    # standardize scale
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)

    hierarchical_cluster.fit(data_std)
    return hierarchical_cluster.labels_

def fit_modified(dataset,linkage_type):
    data,_ = dataset

    #standardize scale
    scale = StandardScaler()
    data_std = scale.fit_transform(data)

    # calculating linkage matrix
    Z = linkage(data_std,method=linkage_type)

    distances = Z[:,2]

    # first derivative (slope)
    slope = np.diff(distances)


    # maximum index
    max_slope_idx = np.argmax(slope)

    # Defining cut-off distance with max
    slope_threshold = distances[max_slope_idx]

    # set number of clusters with cutoff
    num_clusters_slope = np.count_nonzero(distances > slope_threshold)+1

    # create model with slope cut-off
    model_slope = AgglomerativeClustering(n_clusters=num_clusters_slope,linkage=linkage_type)    
    model_slope.fit(data_std)
    return model_slope.labels_


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    n_samples = 500
    seed = 42

    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
    )

    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)

    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)

    #rng = np.random.RandomState(seed)
    #no_structure = rng.rand(n_samples, 2), None

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state= seed
    )

    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)


    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {
        'nc':noisy_circles,
        'nm':noisy_moons,
        'bvv':varied,
        'add':aniso,
        'b':blobs
    }

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    linkage_types = ['single','complete','ward','average']
    fig,axes = plt.subplots(nrows=4,ncols=5,figsize=(20,16)) 
    dataset_group = [noisy_circles,noisy_moons,varied,aniso,blobs]
    for i, linkage in enumerate(linkage_types):
        for j, dataset in enumerate(dataset_group):
            labels = fit_hierarchical_cluster(dataset,linkage,2)
            
            scaler = StandardScaler()
            data_std = scaler.fit_transform(dataset[0])

            #plotting

            axes[i,j].scatter(data_std[:,0],data_std[:,1], c=labels, cmap ='viridis', marker= '.')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            axes[i,j].set_title(f"Linkage Type = {linkage}, Dataset {j+1}")

    plt.tight_layout()
    #plt.show()
        
    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"] # These are the ones I've observed being correctly clustered here, and not in k-means.

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    fig,axes = plt.subplots(nrows=4,ncols=5,figsize=(20,16)) 
    for i, linkage in enumerate(linkage_types):
        for j, dataset in enumerate(dataset_group):
            labels = fit_modified(dataset,linkage)
            
            scaler = StandardScaler()
            data_std = scaler.fit_transform(dataset[0])

            #plotting

            axes[i,j].scatter(data_std[:,0],data_std[:,1], c=labels, cmap ='viridis', marker= '.')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            axes[i,j].set_title(f"Linkage Type = {linkage}, Dataset {j+1}")

    plt.tight_layout()
    #plt.show()

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
