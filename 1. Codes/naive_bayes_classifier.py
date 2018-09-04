# Training of the model.

def train_naive_bayes(train_data, label):
    
    """
    Input:
    1. train_data: the train data with the label column. The data must be in 2d array(not matrix).
    2. label: the column index of the label in the train_data
    
    Output:
    1. mean: the mean vector of the features in dictionary.
    2. var: the variance vector of the features in dictionary.
    3. prior_prob : the prior probability of each of the labels dictionary.
    4. class_label : label of the classes.
    
    In output dictionary the keys are the label.
    """

    import numpy as np

    # Storing mean and variances of each of the features of each of the classes in dictionary.
    
    mean = {}
    var  = {}
    prior_prob = {}
    
    

    N = train_data.shape[0]
    class_labels = np.unique(train_data[:, label])
    for i in class_labels:
        X = train_data[train_data[:, label] == i, :] # make seperate matrix for each of the classes.
        X_features = np.delete(X, label, axis = 1)       # delete the class column and take only the features.
        mean[i] = X_features.mean(0)                 # means of the features
        var[i]  = X_features.var(0)                  # variances of the features
        prior_prob[i] = X_features.shape[0]/N
        
    return mean, var, prior_prob, class_labels
    



# Tesing the model

def naive_bayes_classification(test_feature, mean, var, prior_prob, class_labels):
    
    """
    Input:
    1. test_feature: the test data in 2d array.
    2. mean : means of each of the class in dictionary. [returned from the training step]
    3. var  : variance of each of the class in dictionary.[returned from the training step]
    4. prior_prob: prior probariliy of each of the class in dictionary.[returned from the training step]
    5. class_labels : the class labels of the training data.[returned from the training step]
    
        
    Output:
    1. Predicted Label in a list.
    """
    
    import math
    import numpy as np
    
    
    pred_label = []
    
    
    for i in range(len(test_feature)):
        x = test_feature[i]

        class_compare = np.empty([len(class_labels), 3])
        class_compare[:] = np.nan
        class_compare[:, 0] = class_labels

        bayes_numerator_all_class = []

        bayes_numerator_all_class = [bayes_numerator(x, mu = mean[i], variance = var[i], prior = prior_prob[i]) for i in class_labels]

        normalizing_factor = sum(bayes_numerator_all_class)

        posterior_porb = [bayes_numerator_all_class[i]/ normalizing_factor for i in range(len(bayes_numerator_all_class))]

        class_compare[:, 1] = bayes_numerator_all_class
        class_compare[:, 2] = posterior_porb
        maximum_posterior =  max(class_compare[:, 2])
        c = class_compare[class_compare[:,2] == maximum_posterior,][0][0]

        pred_label.append(c)
        
        
    return pred_label
    
    


# Testing
# The likelihood function.
def normal_likelihood(x, mu, var):
    import math
    
    p1 = 1/ math.sqrt(var * 2 * math.pi)
    p2 = math.exp(-0.5 * pow((x - mu), 2)/var)
    
    return (p1*p2)
    

# The Bayes Numerator
def bayes_numerator(x, mu, variance, prior):
    import numpy as np
    likelihood = [normal_likelihood(x[i], mu[i], variance[i]) for i in range(len(x))]
    bayes_numerator = np.prod(likelihood) * prior
    
    return bayes_numerator
   
