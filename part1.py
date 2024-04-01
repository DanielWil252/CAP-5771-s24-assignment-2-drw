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
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""



# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters):
    #Filter data
    data, _ = dataset
    
    #Standardize scale with StandardScaler()
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    #Initialize kmeans
    kmeans = cluster.KMeans(n_clusters=n_clusters, init='random', random_state=42)
    
    #Fit data.
    kmeans.fit(data_std)
    
    # Return the predicted labels from k-means clustering
    return kmeans.labels_


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
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
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["1A: datasets"] = {
        'nc': noisy_circles,
        'nm': noisy_moons,
        'bvv': varied,
        'add': aniso,
        'b': blobs
    }

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """
    cluster_numbers = [2,3,5,10]
    dataset_group = [noisy_circles,noisy_moons,varied,aniso,blobs]
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 16))
    for i, clusters in enumerate(cluster_numbers):
        for j, dataset in enumerate(dataset_group):
            labels = fit_kmeans(dataset, clusters)

            scaler = StandardScaler()
            data_std = scaler.fit_transform(dataset[0])

            #Plot the scatterplot figure.
            

            axes[i,j].scatter(data_std[:,0],data_std[:,1], c=labels, cmap ='viridis', marker= '.')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            axes[i,j].set_title(f"K = {clusters}, Dataset {j+1}")

    plt.tight_layout()
    #plt.show() 
    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct = answers["1C: cluster successes"] = {"bvv": [3],"add": [3],"b": [3]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["nc","nm"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = ["I did not notice any apparent/noticably large changes in the patterns of the clusters, even when removing the set seed in kmeans."]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
