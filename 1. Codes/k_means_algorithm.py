import numpy as np
import math


def k_means(feature_sample, test_sample, k, iter_conv):
    """
    Input:
    feature_sample: The training features without the label.
    test_sample: The test features without the label.
    k: number of clusters needed.
    iter_conv: the limit of iteration for convergence. The number of iteration will not pass this limit. When the limit is reached
    the result is produced.
    
    Output:
    centroids: The cluster centroids.
    number of iteration done: number of iteration done to reach the results.
    cluster index: the cluster indices, which will be corresponding to the cluster controids.
    
    Solution Structure:
    To keep track of the distances a matrix X is prepared with k number of columns(required number of clusters).
    The number of rows in X is the number of items in the feature_sample(training data).
    
    Then X is filled with the distances from each of the initial cluster centroids.
    
    The initial centroids are taken considering items with minimum distances.
    
    The means of each of the attributes is the modified centroids. 
    
    """
    # necessary modules
    import random

    cluster_indices = None       # to store the cluster indices of the test data.


    fdata = feature_sample
    data = test_sample
    
    n, p = data.shape
    
    
    X = np.zeros([n, k])        # matrix X record the distances from centroids.
    X.fill(np.nan)

    # randomly choose k centroids
    random_centroids_row = random.sample(range(n), k)
    initial_centroids = fdata[random_centroids_row,]

    
    
    convergence = True
    
    iteration = 0


    make_vector = lambda X : np.squeeze(np.asarray(X)) # this function makes an two dimension n*1 matrix to n item vector.

    while(iteration < iter_conv and convergence):
        
        # Measure distance from each attribute of the feature and centroid and keep it to the matrix X                            

        for i in range(n):
            for j in range(k):
                X[i,j] = euclidean_distance(make_vector(fdata[i]), make_vector(initial_centroids[j,:])) 





        initial_cluster = [np.argmin(X[i]) for i in range(n)] # takes the index of the item which is closer to the centroids(the 
                                                              # column is the mark of initial centroids).
        
        
        # matrix modified_centroid to update the centroid.
        modified_centroid = np.zeros([k, p])
        modified_centroid.fill(np.nan)
        
        # take the mean of the initial clusters and update as modified centroids.

        for j in range(k):
            initial_cluster_features = [initial_cluster[i] == j for i in range(n)]
            modified_centroid[j, ] = np.mean(fdata[initial_cluster_features,], axis = 0)
            
        
        # check the convergenc.
        if (initial_centroids == modified_centroid).all():
            convergence = False
        else:
            initial_centroids = modified_centroid
            iteration += 1

    cluster_indices = initial_cluster

    return modified_centroid, iteration, cluster_indices


import math   

## Function for euclidean Distance
def euclidean_distance(x, y):
    # import math
    if len(x) != len(y):
        print('Length of two features/vectors are not same.')
    else:
        n = len(x)
        d2 = 0
        for i in range(n):
            d2 = d2 + (x[i] - y[i])**2
        d = math.sqrt(d2)
    
    return d
