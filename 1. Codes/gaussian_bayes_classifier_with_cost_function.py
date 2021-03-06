# Training of the model.

def train_naive_bayes(train_data, label):
    
    """
    Input:
    1. train_data: the train data with the label column. The data must be in 2d array(not matrix).
    2. label: the column index of the label in the train_data.
    
    Output:
    1. mean: the mean vector of the features in dictionary.
    2. covariance: the covariance of the features in dictionary.
    3. prior_prob : the prior probability of each of the labels dictionary.
    4. class_label : label of the classes.
    
    In output dictionary the keys are the label.
    """

    import numpy as np

    # Storing mean and variances of each of the features of each of the classes in dictionary.
    
    # Train Model.

    # Storing mean and variances of each of the features of each of the classes in dictionary.
    mean = {}
    covariance  = {}
    prior_prob = {}


    
    N = train_data.shape[0]
    class_labels = np.unique(train_data[:, label])
    for i in class_labels:
        X = train_data[train_data[:, label] == i, :] # make seperate matrix for each of the classes.
        X_features = np.delete(X, label, axis = 1)       # delete the class column and take only the features.
        mean[i] = X_features.mean(0)                 # means of the features
        covariance[i]  = np.cov(X_features.T)        # covariances of the features
        prior_prob[i] = X_features.shape[0]/N


            
    return mean, covariance, prior_prob, class_labels
    


# Testing

# Normal Likelihood

def multivariate_gaussian_likelihood(x, mu, sigma):
    
    import numpy as np
    import math
    
    p = len(x)
    
    p1 =  1 / (pow(2 * (math.pi), p/2) * np.linalg.det(sigma))

    X = np.asmatrix(x - mu)
    Y = np.asmatrix(np.linalg.inv(sigma))
    p2 = math.exp(-0.5 * np.dot(np.dot(X,Y), X.T))

    return p1*p2

    

    

# Bayes Numerator

def bayes_numerator(likelihood, prior):
    return likelihood * prior




def gaussian_bayes_classification_with_cost(test_feature, mean, covariance, prior_prob, class_labels, cost_function):
    """
    Input:
    1. test_feature: the test features in 2D array(not matrix).
    2. mean: the mean vector of the features in dictionary.(from the training step)
    3. covariance: the covariance of the features in dictionary.(from the training step)
    4. prior_prob : the prior probability of each of the labels dictionary.(from the training step)
    5. cost_function: a 2d array associated with corresponding cost of misclassification.

    Output:
    The predicted label in list.
    """

    import numpy as np

    pred_label = []

    # Test the data
    for i in range(len(test_feature)):
        x = test_feature[i]

        class_compare = np.empty([len(class_labels), 4])
        class_compare[:] = np.nan
        class_compare[:, 0] = class_labels

        bayes_numerator_all_class = [multivariate_gaussian_likelihood(x, mean[i], covariance[i]) for i in class_labels]

        normalizing_factor = sum(bayes_numerator_all_class)

        posterior_porb = [bayes_numerator_all_class[i]/ normalizing_factor for i in range(len(bayes_numerator_all_class))]

        class_compare[:, 1] = bayes_numerator_all_class
        class_compare[:, 2] = posterior_porb

        risk_function = np.dot(cost_function, posterior_porb)
    
        class_compare[:, 3] = risk_function  
        

        
        minimum_cost      =  min(class_compare[:, 3])
        
        
        

        c = class_compare[class_compare[:,3] == minimum_cost,][0][0]

        pred_label.append(c)
    
    return pred_label