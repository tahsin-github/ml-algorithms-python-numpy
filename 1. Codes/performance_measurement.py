def confusion_matrix(true_label, pred_label, percentage = True):
    """
    Input: 
    1. true_label: The True label.
    2. pred_label: The predicted label.
    3. percentage: If the result wanted in percentage. True/False
    
    Output:
    1. total accuracy
    2. confusion table. True label * Predicted label
    """


    import numpy as np

        


    if len(pred_label) == len(true_label):
        n = len(pred_label)
        accuracy = round((sum(np.asarray(pred_label) == np.asarray(true_label))/n) * 100, 2)
        
        p_l = np.unique(pred_label)
        t_l = np.unique(true_label)
        n = len(p_l)
        confusion_matrix = np.zeros([n, n])

        for i in range(n):
            t = t_l[i]
            for j in range(n):
                p = p_l[j]
                c = sum([pred_label[k] == p and true_label[k] == t for k in range(len(pred_label))])
                confusion_matrix[i,j] = c
        
        if percentage:
            confusion_matrix = np.around((confusion_matrix/len(pred_label)) * 100, decimals = 2)
        
        
        
        return accuracy, confusion_matrix



    else:
        print("Predicted label and True label lengths are not same.")
        
