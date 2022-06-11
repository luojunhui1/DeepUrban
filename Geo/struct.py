import numpy as np
import struct
from shapely.geometry import Polygon

import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from Geo.geo import *
from Utils.utils import *

class AdjMatrix:
    """ Ajacency matrix object

    Generate an object customised for processing adjacency matrix, for different region
    segmentation model, such as 'grid' or 'polygon' providing various functions
    
    Parameters
    ----------
    matrix : np.array
        initialized adjacent matrix

    region_seg_model : string
        name of region segmentation method, default 'polygon', can be 'grid'
    
    grid_h : int
        only for 'grid', regions' row amounts

    grid_w : int
        only for 'grid', regions' colunm amounts 

    polygons : list
        list of region polygons, usually the first dimention claims the a whole region,\\
        the second dimention refers the partition of region, last dimention stores the\\ 
        latitude and longitude of every vertical, like \\
        [   region 1 : 
                [   partition 1 : [[lon, lat], [lon, lat]], \\
                    partition 2 : [[lon, lat], [lon, lat]], ... \\
                ], ...\\
        ]
    """

    def flatten_poygons(self, polygons):
        for i in range(len(polygons)):
            l1 = polygons[i]
            la = [] # all verticals of polygon
            for l in l1:
                if len(la) == 0:
                    la = l.copy()
                else:
                    la.extend(l)
            self.poygons_flatten.append(la)
        return self.poygons_flatten

    def cal_eucl_dist(self, center, indexs, l):
        dists = []
        for index in indexs:
            dists.append(np.sqrt(sum(np.power(center - l[index], 2))))
        return dists

    def cal_guass_weight(self, center, indexs, l, k = 0, theta = 0):
        weights = []
        dists = self.cal_eucl_dist(center, indexs, l)
        for i in range(len(indexs)):
            w = np.exp(-1.0*(dists[i]**2)/(2*(theta**2)))
            if w <= k:
                weights.append(w)
            else:
                weights.append(0)
        return weights

    def get_mainland(self):
        l = []
        for pl in self.polygons:
            if isinstance(pl[0], list) or isinstance(pl[0], np.array):
                areas = []
                for p in pl:
                    areas.append(Polygon(p).area)
                l.append(pl[np.argmax(areas)])  
            else:
                l.append(pl)
        return l

    def gen_grid_adjmatrix(self, grid_h, grid_w):
        region_num = grid_w*grid_h
        adj_matrix = np.array([[0]*region_num]*region_num)
        for i in range(region_num):
            adjrs = adjacency_regions(i, grid_h, grid_w)
            adjrs.append(i)
            adj_matrix[i, adjrs] = 1
        return adj_matrix

    def gen_polygon_adjmatrix(self, polygons):
        hash_l = []
        self.flatten_poygons(polygons)

        for i in range(len(self.flatten_poygons)):
            # change latitude and longitude to string with high data precision
            hash_l.append(np.array([struct.pack('<f', x[0]) + struct.pack('>f', x[1]) for x in self.flatten_poygons[i]]))

        adj_matrix = np.array([[0]*len(polygons)]*len(polygons))
        for i in range(len(hash_l)):
            for j in range(i, len(hash_l)):
                # adjacent polygons share two verticals
                if same_point_num(hash_l[i], hash_l[j]) > 2:
                    adj_matrix[i, j] = 1
        # adjacency matrix is symmetric 
        adj_matrix = np.logical_or(adj_matrix, adj_matrix.T).astype(int)   

        return adj_matrix

    def weight_center_dist(self, mainland = True, k = 0, theta = 0):
        l = []
        if mainland:
            l = self.get_mainland()
        else:
            l = self.polygons
        
        if self.poygons_flatten == []:
            self.flatten_poygons(l)
        
        centers = []
        for pl in self.poygons_flatten:
            centers.append(np.mean(pl, axis=0))
        
        if k == 0 and theta == 0:
            for i in range(self.adjmatrix.shape[0]):
                adj_index = np.where(self.adjmatrix[i] > 0)[0]
                dists = self.cal_eucl_dist(i, adj_index, centers)
                self.adjmatrix[i, adj_index] = norm_by_col(dists) 
        else:
            # weight the edge like : Sun J, Zhang J, Li Q, et al. 
            # Predicting citywide crowd flows in irregular regions using multi-view graph convolutional networks[J]. 
            # IEEE Transactions on Knowledge and Data Engineering, 2020.
            for i in range(self.adjmatrix.shape[0]):
                adj_index = np.where(self.adjmatrix[i] > 0)[0]
                weights = self.cal_guass_weight(i, adj_index, centers, k, theta)
                self.adjmatrix[i, adj_index] = weights

        return None
    
    def weight_share_length(self, mainland):
        l = []
        if mainland:
            l = self.get_mainland() 
            for i in range(len(l)):
                share_lens = []
                adj_index = np.where(self.adjmatrix[i] > 0)[0]
                for j in adj_index:
                    polygon_i = Polygon(l[i])
                    polygon_j = Polygon(l[j])  
                    share_lens.append(polygon_i.intersection(polygon_j).length)
                self.adjmatrix[i, adj_index] = norm_by_col(share_lens)
        else:   
            l = self.polygons
            for i in range(len(l)):
                share_lens = []
                adj_index = np.where(self.adjmatrix[i] > 0)[0]
                for j in adj_index:
                    length = 0
                    for m in range(len(l[i])):
                        polygon_i = Polygon(l[i][m])
                        for n in range(len(l[j])):
                            polygon_j = Polygon(l[j][n])
                            length += polygon_i.intersection(polygon_j).length
                    share_lens.append(length)
                self.adjmatrix[i, adj_index] = norm_by_col(share_lens)
                
        return None

    def __init__(self, matrix, region_seg_model, grid_h = None, grid_w = None, polygons = None):
        self.region_seg_model = self.region_seg_model
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.polygons = polygons
        self.poygons_flatten = []

        if np.sum(matrix) == 0:
            print('matrix is not initialized, try to generate one ......')
            if region_seg_model == 'grid':
                self.adjmatrix = self.gen_grid_adjmatrix(grid_h, grid_w)
            else:
                self.adjmatrix = self.gen_polygon_adjmatrix(polygons)
        else:
            self.adjmatrix = matrix

    def weight_edge(self, by = None, mainland = True, k = 0, theta = 0):
        if self.region_seg_model == 'grid':
            print('there is no corresponding function to weight the edge of grid map ')
            return None

        if by == 'dist':
            self.weight_center_dist(mainland, k, theta)
        else:
            self.weight_share_length(mainland)
