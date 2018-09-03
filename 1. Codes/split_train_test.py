# train-test data
class split_train_test:

    
    
    def __init__(self, data, percentage, label):
        '''
        Input:
        data: a numpy matrix.
        percentage: percentage of samples for the training. percentage must be greater than 0 and less than 100.
        label: column index of the label.
        
        
        '''

        # import necessary module

        import numpy as np

        # make an one dimensional matrix n*1 to a vecto

        make_vector = lambda X : np.squeeze(np.asarray(X))


        if percentage <= 0 or percentage >= 100:
            print("The percentage of training data must be greater than 0 and less than 100.")
        else:
            import random;
        
            n = data.shape[0]  # take the number of rows in original data.
            l = int(n*percentage/100)
        
        
            # Take random integers to slice the data
            train_rows = random.sample(range(n), l)
            test_rows  = [i for i in range(n) if i not in train_rows]
            
            # Split the data to train and test.
            self.train = data[train_rows,:]
            self.test  = data[test_rows, :]
            
            # Split both train and test data to feature and label.
            self.train_feature = np.delete(self.train, label, 1)
            self.train_label = make_vector(self.train[:, label])
            
            # Split both train and test data to feature and label.
            self.test_feature = np.delete(self.test, label, 1)
            self.test_label = make_vector(self.test[:, label])
        
 