# convert intiger label to string label, according to the map from the funcion load data

def label_to_string(label_int, d_map):
    """
    convert intiger label to string label, according to the map from the funcion load data.
    
    Input:
    label_int: the label in integer. data type: list or np.array
    d_map: the data map from the load data module.

    Output:
    label : a list of string labels.

    """
    lab_map = dict(zip(d_map.values(), d_map.keys()))
    pred_label = [lab_map[i] for i in label_int]
    return pred_label


