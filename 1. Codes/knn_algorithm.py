import numpy as np
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



def K_nearest_neighbour(feature_sample, feature_label, test_sample, k):
    
    """
    Input: 
    feature_sample: feature matrix, returned from train test split. in numpy matrix
    feature_label : feature label matrix, returned from train test split. in numpy matrix
    test_sample :   test feature matrix, returned from train test split. in numpy matrix
    k : number of items to be considered as "nearest neighbours." Data type: intiger.
    
    Output:
    Predicted label: A list of predicted lables corresponding to the given feature.
    
    In the implementation "np.squeeze(np.asarray(MATRIX))" was used to convert matrix to array for easy of calculation.
    """
    
    m = test_sample.shape[0]                        # number of samples in the test data.
    n = feature_sample.shape[0]                     # number of samples in the train data
    
    predicted_label_list = [] 
    
    for i in range(m):
        X = test_sample[i,:]
        
        D = np.full([n,2], np.nan)                  # make an empty matrix to store feature label and corresponding distance. 1st Column 
                                                    # is for label and 2nd column is for corresponding distance from the test feature.
        D[:, 0] = np.squeeze(np.asarray(feature_label))
        
        for j in range(n):
            D[j,1] = euclidean_distance(np.squeeze(np.asarray(X)), np.squeeze(np.asarray(feature_sample[j,:])))
        
        Z = D[np.argsort(D[:, 1]),][:, 0]
        Z = Z[0:k]

        lab, count = np.unique(Z, return_counts = True)
        
        predicted_label = lab[np.argsort(count)[0],]
        
        predicted_label_list.append(predicted_label)
        
    return predicted_label_list
