import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from Utils.utils import rgb2hex


def show_mass(data, labels, u_labels=None, colors=[], reduction = 'tsne'):
    fig = plt.figure(figsize=(15, 10))
    
    if u_labels.all() == None:
        u_labels = np.unique(labels)

    draw_data = data
    if data.shape[1] > 3:
        if reduction == 'tsne':
            draw_data = TSNE(n_components=3, learning_rate=50).fit_transform(data)
        elif reduction == 'pca':
            draw_data = PCA(n_components=3).fit_transform(data)

    if colors == []:
        grad = np.linspace(0, 1, len(u_labels) + 1)
        for i in range(len(u_labels)):
            colors.append(rgb2hex(plt.cm.rainbow(grad[i])))
    
    ax = None
    if draw_data.shape[1] == 2:
        ax = fig.add_subplot(111)
    elif draw_data.shape[1] == 3:
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    for i in range(len(u_labels)):
        label = u_labels[i]
        data_cor_label = draw_data[np.where(labels == label)[0]]
        if draw_data.shape[1] == 2:
            ax.scatter(data_cor_label[:, 0], data_cor_label[:, 1], c=colors[i], alpha=0.6)
        elif draw_data.shape[1] == 3:
            ax.scatter3D(data_cor_label[:, 0], data_cor_label[:, 1], data_cor_label[:, 2], c=colors[i], alpha=0.6)
        else:
            print("unexpected data dimention : ", draw_data.shape)
    
    return fig

def show_series(data, labels, u_labels=None, colors=[]):
    fig = plt.figure(figsize=(15, 30))
    
    if u_labels.all() == None:
        u_labels = np.unique(labels)

    if colors == []:
        grad = np.linspace(0, 1, len(u_labels) + 1)
        for i in range(len(u_labels)):
            colors.append(rgb2hex(plt.cm.rainbow(grad[i])))

    n_row = math.ceil(len(u_labels) / 2)
    n_col = 2
    
    print("row: ", n_row, ", col : ", n_col)
    
    for i in range(n_row):
        for j in  range(n_col):
            if i*n_col + j >= len(u_labels):
                continue
            ax = fig.add_subplot(n_row, n_col, i*n_col + j + 1)
            label = u_labels[i*n_col + j]
            data_cor_label = data[np.where(labels == label)[0]]
            for k in range(len(data_cor_label)):
                ax.plot(range(data.shape[1]), data_cor_label[k], c='#f3f3f3')
            mean_cor_data = np.mean(data_cor_label, axis=0)
            ax.plot(range(data.shape[1]), mean_cor_data, c=colors[i*n_col + j])
            
    return fig
    
def visiualize_cluster(data, labels, method):
    if data.shape[1] < 2:
        print("required data dimention >= 2")
        return None

    u_labels = np.unique(labels)
    
    grad = np.linspace(0, 1, len(u_labels) + 1)

    colors = []
    for i in range(len(u_labels)):
        colors.append(rgb2hex(plt.cm.rainbow(grad[i])))
    
    fig = None
    if method == 'mass':
        fig = show_mass(data, labels, u_labels, colors)
    elif method == 'series':
        fig = show_series(data, labels, u_labels, colors)

    return fig