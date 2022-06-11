import h5py
import numpy as np

def get_h5data(fname=''):
    hp = h5py.File(fname, 'r')
    data = np.array(hp['data'])
    return data

def get_npydata(fname=''):
    return np.load(fname)