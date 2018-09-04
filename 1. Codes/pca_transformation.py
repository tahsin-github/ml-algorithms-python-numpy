def pca_transformaiton(data, k):
    """
    Input:
    1. feature_data: the data without the label.
    2. k : number of features needed in the final transformation. k must be less than the total number of features in the main 
    dataset.
    
    Output:
    1. The transformed data with k features.
    
    
    """
    
    import numpy as np
        
    f_means = np.mean(data, axis = 0) # means of each of the features(by column)
    adjusted_data = data - f_means # the adjusted data. main feature data - feature means
    V = np.transpose(adjusted_data) # the transpose of the adjusted_data
    C = np.cov(V) # the covariance matrix
    eig_value, eig_vectors = np.linalg.eig(C) # the eigen valus and eigen vectors of the covariance matrix.
    eig_value_indices_sort = np.argsort(eig_value) # find indices of the sorted values. 
    P = (eig_value_indices_sort < len(eig_value_indices_sort) - k) != True # find the indices of first k maximum
    A = eig_vectors[:, P] # eigen vectors of the maximum eigen values
    
    # pca transformation
    pca_transformation = np.transpose(np.matmul(np.transpose(A), np.transpose(adjusted_data)))
    
    return pca_transformation
    
    
    
    
    
