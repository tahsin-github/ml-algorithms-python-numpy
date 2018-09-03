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
