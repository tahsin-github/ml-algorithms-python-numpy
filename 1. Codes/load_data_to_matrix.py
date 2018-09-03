# Load the file and convert it to numpy matrix.

# Load necessary modules
import numpy as np
import csv



def load_text_data(file, delimiter, header, label_col_is_str , label_col):
    """
    Loading the text data and convert it to numpy matrix.
    If there is a label column with string data type, it will convert the label to Intiger and replace the string. And also return
    the mapping of the string mapping in a dictionary.
    
    Input:
    file: file name
    delimiter: "," or "\t" etc.
    header : True / False
    label_col_is_str: True/ False
    label_col : the column index of the label.
    
    Output:
    
    Data: in numpy matrix.
    label_map: in dictionary.
    
    """
    
    text_file = open(file, "rt")
    reader = csv.reader(text_file, delimiter = delimiter)
    x = list(reader)
    text_file.close()
    
    # if there is a header, remove it.
    if header:
        x.pop(0)
    Data = np.asmatrix(x)
    
    if label_col_is_str:
        try:
            Data, Label_Map = map_label(Data, label_col)
            return Data, Label_Map
        except ValueError:
            print("The header is string data type. Please specify the header = True in the function.")
            Data = None
    else:
        return Data
            




# mapping of the string labels.


# mapping the label
def map_label(Data, label_column):
    """
    If the label column is string, create a map for the label and convert it to intiger.
    
    Input:
    1. Data: in numpy matrix.
    2. label_column: the column number containing the label.
    
    Output:
    1. Data: in numpy matrix in float data type.
    """
    Label_Map = {}
    
    Label = np.unique(np.array(Data[:, label_column]))
    
    for i in range(len(Label)):
        Label_Map[Label[i]] = i
    
    mapping = lambda label, label_map : label_map[label]
    
    Data[:, label_column] = np.vectorize(mapping)(Data[:, label_column], Label_Map)
    
    Data = Data.astype(float) 
    
    return Data, Label_Map
    
    
