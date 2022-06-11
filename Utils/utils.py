import math
import numpy as np

def rgb2hex(rgbcolor):
    """ transfrom the rgba color format to hex format\\
    ususally used for matplotlib.pyplot.cm

    Parameter
    ----------
    rgbcolor : tuple(r, g, b, a)
        4 channel color format
    """
    r, g, b, a = rgbcolor
    r = math.floor(r * 255)
    g = math.floor(g * 255)
    b = math.floor(b * 255)

    res = str(hex(r)[2:])
    if r < 16:
        res = '0' + res
    
    if g < 16:
         res += '0' + str(hex(g)[2:])
    else:
        res += str(hex(g)[2:])
    
    if b < 16:
         res += '0' + str(hex(b)[2:])
    else:
        res += str(hex(b)[2:])

    return '#' + res

def same_point_num(l1, l2):
    """ amounts of same elements in l1 and l2
    
    Parameters
    ----------
    l1 : list or array-like
    l2 : list or array-like
    """

    same_en = set(l1).intersection(set(l2))
    return len(same_en)
    
def norm_by_row(data):
    """ normalize data by row

    Parameters
    ----------
    data : 2 dimentional array
    """
    # m = np.mean(data, axis = 1)
    res = (data.copy()).astype(np.float64)
    for index in range(0, len(data)):
        mx = np.max(data, axis=1)
        mn = np.min(data, axis=1)
        res[index] = (res[index] - mn[index]) / (mx[index] - mn[index])
    return res

def norm_by_col(data):
    """ normalize data by colunm
    
    Parameters
    ----------
    data : 2 dimentional array

    """
    # m = np.mean(data, axis = 1)
    res = (data.copy()).astype(np.float64)
    mx = np.max(data, axis=0)
    mn = np.min(data, axis=0)
    res = (res - mn) / (mx - mn)
    return res

def jaccard(p,q):
    """
    calculate jaccard similarity of 2 list
    
    Parameters
    -----
    p : list or array-like
    q: list or array-like

    Returns
    -----
        float, jaccard similarity
    
    """
    c = [v for v in p if v in q]
    return float(len(c))/(len(q)+len(p)-len(c))

def unique_list(l):
    """
    trasnform the data value to [0, len(unique(l)))
    
    Parameters
    -----
    l : list or array-like
        list data

    Returns
    -----
        array 
    
    """
    unique_l, unique_index = np.unique(l, return_inverse=True)
    return np.array(range(len(unique_l)))[unique_index]

def cal_label_param(labels, data):
    """
    calculate parameters corresdponding to data labels, especially the 'error_means' and 'error_stds'
    don't mean the mean and std of original data but the attribute of the sse(sum of squared error), 
    which really makes some sense

    Parameters 
    ------
    labels: list or array-like
        label list of original data

    data: array-like
        original data, 2-dimentional

    Returns
    ------
    centers: array
        center of every label
    error_means: array
        average of sse index for every label
    error_stds: array
        standard deviation of sse index for every label
    """
    unique_labels = np.unique(labels)

    centers = []
    error_means = []
    error_stds = []

    for label in unique_labels:
        label_cor_data = data[np.where(labels == label)[0]]
        centers.append(np.mean(label_cor_data, axis=0))
        label_errors = np.sqrt(np.sum((label_cor_data - centers[-1])**2, axis=1))
        error_means.append(np.mean(label_errors))
        error_stds.append(np.std(label_errors, ddof=1))

    return np.array(centers), np.array(error_means), np.array(error_stds)