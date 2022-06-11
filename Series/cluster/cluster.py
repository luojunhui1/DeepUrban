from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from scipy.spatial import KDTree

def dist_eucl(vecA, vecB):
    """
    @brief      the similarity function
    @param      vecA  The vector a
    @param      vecB  The vector b
    @return     the euclidean distance
    """
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def rand_cent(data_mat, k):
    """
    @brief      select random centroid
    @param      data_mat  The data matrix
    @param      k
    @return     centroids
    """
    n = np.shape(data_mat)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(data_mat[:,j]) 
        rangeJ = float(max(data_mat[:,j]) - minJ)
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids

def kmeans(data_mat, k, dist = "dist_eucl", create_cent = "rand_cent"):
    """
    @brief      kMeans algorithm
    @param      data_mat     The data matrix
    @param      k            num of cluster
    @param      dist         The distance funtion
    @param      create_cent  The create centroid function
    @return     the cluster
    """
    m = np.shape(data_mat)[0]
    # 初始化点的簇
    cluster_assment = np.mat(np.zeros((m, 2)))  # 类别，距离
    # 随机初始化聚类初始点
    centroid = eval(create_cent)(data_mat, k)
    cluster_changed = True
    # 遍历每个点
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_index = -1
            min_dist = np.inf
            for j in range(k):
                distance = eval(dist)(data_mat[i, :], centroid[j, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
                cluster_assment[i, :] = min_index, min_dist**2
        # 计算簇中所有点的均值并重新将均值作为质心
        for j in range(k):
            per_data_set = data_mat[np.nonzero(cluster_assment[:,0].A == j)[0]]
            centroid[j, :] = np.mean(per_data_set, axis = 0)
    return centroid, cluster_assment

def get_closest_dist(point, centroid):
    """
    @brief      Gets the closest distance.
    @param      point     The point
    @param      centroid  The centroid
    @return     The closest distance.
    """
    # 计算与已有质心最近的距离
    min_dist = np.inf
    for j in range(len(centroid)):
        distance = dist_eucl(point, centroid[j])
        if distance < min_dist:
            min_dist = distance
    return min_dist

def kpp_cent(data_mat, k):
    """
    @brief      kmeans++ init centor
    @param      data_mat  The data matrix
    @param      k   num of cluster      
    @return     init centroid
    """
    data_set = data_mat.getA()
    # 随机初始化第一个中心点
    centroid = list()
    centroid.append(data_set[np.random.randint(0,len(data_set))])
    d = [0 for i in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i in range(len(data_set)):
            d[i] = get_closest_dist(data_set[i], centroid)
            total += d[i]
        # 轮盘法选择下一个聚类中心，d[i]越大被选择的概率就越大
        total *= np.random.rand()
        # 选取下一个中心点
        for j in range(len(d)):
            total -= d[j]
            if total > 0:
                continue
            centroid.append(data_set[j])
            break
    return np.mat(centroid)

def kpp_means(data_mat, k, dist = "dist_eucl", create_cent = "kpp_cent"):
	"""
	@brief      kpp means algorithm
	@param      data_mat     The data matrix
	@param      k            num of cluster
	@param      dist         The distance funtion
	@param      create_cent  The create centroid function
	@return     the cluster
	"""
	m = np.shape(data_mat)[0]
	# 初始化点的簇
	cluste_assment = np.mat(np.zeros((m, 2)))  # 类别，距离
	# 随机初始化聚类初始点
	centroid = eval(create_cent)(data_mat, k)
	cluster_changed = True
	# 遍历每个点
	while cluster_changed:
		cluster_changed = False
		for i in range(m):
			min_index = -1
			min_dist = np.inf
			for j in range(k):
				distance = eval(dist)(data_mat[i, :], centroid[j, :])
				if distance < min_dist:
					min_dist = distance
					min_index = j
			if cluste_assment[i, 0] != min_index:
				cluster_changed = True
				cluste_assment[i, :] = min_index, min_dist**2
		# 计算簇中所有点的均值并重新将均值作为质心
		for j in range(k):
			per_data_set = data_mat[np.nonzero(cluste_assment[:,0].A == j)[0]]
			centroid[j, :] = np.mean(per_data_set, axis = 0)
	return centroid, cluste_assment


def bi_kmeans(data_mat, k, dist = "dist_eucl"):
    """
    @brief      kMeans algorithm
    @param      data_mat     The data matrix
    @param      k            num of cluster
    @param      dist         The distance funtion
    @return     the cluster
    """
    m = np.shape(data_mat)[0]

    # 初始化点的簇
    cluster_assment = np.mat(np.zeros((m, 2)))  # 类别，距离

    # 初始化聚类初始点
    centroid0 = np.mean(data_mat, axis = 0).tolist()[0]
    cent_list = [centroid0]

    # 初始化SSE
    for j in range(m):
        cluster_assment[j, 1] = eval(dist)(np.mat(centroid0), data_mat[j, :]) ** 2 

    while (len(cent_list) < k):
        lowest_sse = np.inf 
        for i in range(len(cent_list)):
            # 尝试在每一类簇中进行k=2的kmeans划分
            row_indexes = np.nonzero(cluster_assment[:, 0].A == i)[0]
            if len(row_indexes) < 2:
                continue
            ptsin_cur_cluster = data_mat[row_indexes,:]
            centroid_mat, split_cluster_ass = kmeans(ptsin_cur_cluster,k = 2)
            # 计算分类之后的SSE值
            sse_split = sum(split_cluster_ass[:, 1])
            sse_nonsplit = sum(cluster_assment[np.nonzero(cluster_assment[:, 0].A != i)[0], 1])
            # 记录最好的划分位置
            if sse_split + sse_nonsplit < lowest_sse:
                best_cent_tosplit = i
                best_new_cents = centroid_mat
                best_cluster_ass = split_cluster_ass.copy()
                lowest_sse = sse_split + sse_nonsplit
        # 更新簇的分配结果
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0].A == 0)[0], 0] = best_cent_tosplit
        cent_list[best_cent_tosplit] = best_new_cents[0, :].tolist()[0]
        cent_list.append(best_new_cents[1, :].tolist()[0])
        cluster_assment[np.nonzero(cluster_assment[:, 0].A == best_cent_tosplit)[0],:] = best_cluster_ass
    return np.array(cent_list), cluster_assment

class visitlist:
    """
        visitlist类用于记录访问列表
        unvisitedlist记录未访问过的点
        visitedlist记录已访问过的点
        unvisitednum记录访问过的点数量
    """
    def __init__(self, count=0):
        self.unvisitedlist=[i for i in range(count)]
        self.visitedlist=list()
        self.unvisitednum=count

    def visit(self, pointId):
        self.visitedlist.append(pointId)
        self.unvisitedlist.remove(pointId)
        self.unvisitednum -= 1

def dbscan(dataSet, eps, minPts):
    """
    @brief      基于kd-tree的DBScan算法
    @param      dataSet  输入数据集，numpy格式
    @param      eps      最短距离
    @param      minPts   最小簇点数
    @return     分类标签
    """
    nPoints = dataSet.shape[0]
    vPoints = visitlist(count=nPoints)
    # 初始化簇标记列表C，簇标记为 k
    k = -1
    C = [-1 for i in range(nPoints)]
    # 构建KD-Tree，并生成所有距离<=eps的点集合
    kd = KDTree(dataSet)
    while(vPoints.unvisitednum>0):
        p = np.random.choice(vPoints.unvisitedlist)
        vPoints.visit(p)
        N = kd.query_ball_point(dataSet[p], eps)
        if len(N) >= minPts:
            k += 1
            C[p] = k
            for p1 in N:
                if p1 in vPoints.unvisitedlist:
                    vPoints.visit(p1)
                    M = kd.query_ball_point(dataSet[p1], eps)
                    if len(M) >= minPts:
                        for i in M:
                            if i not in N:
                                N.append(i)
                    if C[p1] == -1:
                        C[p1] = k
        else:
            C[p] = -1
    return C

class Optics(object):
    """Optics算法"""
    def __init__(self, dataset):
        self.dataset = dataset 
        self.n = len(dataset)
        self.unvisited = [i for i in range(self.n)]
        self.visited = list()
        self.order_list = list()

    def visit(self, index):
        self.visited.append(index)
        self.unvisited.remove(index)
        self.order_list.append(index)

    def cal_core_dist(self, point, point_neighbors, min_pts):
        # 按照离points点的距离排序
        sorted_dist = sorted([dist_eucl(self.dataset[point], self.dataset[item]) for item in point_neighbors])
        return sorted_dist[min_pts - 1]

    def optics(self, eps = 0.1, min_pts = 5):
        self.eps = eps
        self.reach_dist = [np.inf for i in range(self.n)]      # 可达距离
        self.core_dist = [np.inf for i in range(self.n)]     # 核心距离
        kd = KDTree(self.dataset)
        while(self.unvisited):
            # 随机选取一个点
            i = np.random.choice(self.unvisited)
            self.visit(i)
            # 获取i的邻域
            neighbors_i = kd.query_ball_point(self.dataset[i], eps)
            # 如果i是核心点
            if len(neighbors_i) >= min_pts:
                # 计算核心距离
                self.core_dist[i] = self.cal_core_dist(i, neighbors_i, min_pts)
                seed_list = list()
                self.insert_list(i, neighbors_i,seed_list)
                while(seed_list):
                    seed_list.sort(key=lambda x:self.reach_dist[x])
                    j = seed_list.pop(0)
                    self.visit(j)
                    neighbors_j = kd.query_ball_point(self.dataset[j], eps)
                    if len(neighbors_j) >= min_pts:
                        self.core_dist[j] = self.cal_core_dist(j, neighbors_j, min_pts)
                        self.insert_list(j, neighbors_j,seed_list)
        return self.order_list, self.reach_dist

    def insert_list(self, point, point_neighbors, seed_list):
        for i in point_neighbors:
            if i in self.unvisited:
                rd = max(self.core_dist[point], dist_eucl(self.dataset[i], self.dataset[point]))
                if self.reach_dist[i] == np.inf or rd < self.reach_dist[i]:
                    self.reach_dist[i] = rd
                    seed_list.append(i)
                # elif rd < self.reach_dist[i]:
                #     self.reach_dist[i] = rd

    def extract(self, cluster_threshold):
        clsuter_id = -1
        label = [-1 for i in range(self.n)]
        k = 0
        for i in range(self.n):
            j = self.order_list[i]
            if self.reach_dist[j] > cluster_threshold:
                if self.core_dist[j] <= cluster_threshold:
                    clsuter_id = k
                    k += 1 
                    label[j] = clsuter_id
                else:
                    label[j] = -1
            else:
                label[j] = clsuter_id
        return label


def kmeans_lib(data_mat, k, dist = "dist_eucl", create_cent = "rand_cent"):
    estimator = KMeans(n_clusters=k)#构造聚类器
    estimator.fit(data_mat)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    sses = [np.sqrt(sum((data_mat[i] - centroids[label_pred[i]])**2)) for i in range(0, data_mat.shape[0])]
    return centroids, np.stack([label_pred, sses]).T

def kpp_means_lib(data_mat, k, dist = "dist_eucl", create_cent = "kpp_cent"):
    estimator = KMeans(n_clusters=k)#构造聚类器
    estimator.init('k-means++')
    estimator.fit(data_mat)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和
    return centroids, np.stack([label_pred, inertia]).T

def bi_kmeans_lib(data_mat, k, dist = "dist_eucl", create_cent = "kpp_cent"):
    m = np.shape(data_mat)[0]

    # 初始化点的簇
    cluster_assment = np.mat(np.zeros((m, 2)))  # 类别，距离

    # 初始化聚类初始点
    centroid0 = np.mean(data_mat, axis = 0).tolist()[0]
    cent_list = [centroid0]

    # 初始化SSE
    for j in range(m):
        cluster_assment[j, 1] = eval(dist)(np.mat(centroid0), data_mat[j, :]) ** 2 

    while (len(cent_list) < k):
        lowest_sse = np.inf 
        for i in range(len(cent_list)):
            # 尝试在每一类簇中进行k=2的kmeans划分
            row_indexes = np.nonzero(cluster_assment[:, 0] == i)[0]
            if len(row_indexes) < 2:
                continue
            ptsin_cur_cluster = data_mat[row_indexes,:]
            centroid_mat, split_cluster_ass = kmeans_lib(ptsin_cur_cluster,k = 2)
            # 计算分类之后的SSE值
            sse_split = sum(split_cluster_ass[:, 1])
            sse_nonsplit = sum(cluster_assment[np.nonzero(cluster_assment[:, 0] != i)[0], 1])
            print("sse_split, sse_nonsplit", sse_split, sse_nonsplit)
            # 记录最好的划分位置
            if sse_split + sse_nonsplit < lowest_sse:
                best_cent_tosplit = i
                best_new_cents = centroid_mat
                best_cluster_ass = split_cluster_ass.copy()
                lowest_sse = sse_split + sse_nonsplit
        # 更新簇的分配结果
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0] == 1)[0], 0] = len(cent_list)
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0] == 0)[0], 0] = best_cent_tosplit
        cent_list[best_cent_tosplit] = best_new_cents[0, :].tolist()[0]
        cent_list.append(best_new_cents[1, :].tolist()[0])
        cluster_assment[np.nonzero(cluster_assment[:, 0] == best_cent_tosplit)[0],:] = best_cluster_ass
    return np.mat(cent_list), cluster_assment

def dbscan_lib(dataSet, eps, minPts):
    clustering = DBSCAN(eps = eps, min_samples=minPts).fit(dataSet)
    labels = clustering.labels_
    centers = []
    dists = np.array([0]*dataSet.shape[0])
    unique_classes, indices = np.unique(labels, return_inverse=True)
    print('unique class num: ', len(unique_classes))
    for c in range(len(unique_classes)):
        indexes = list(indices == c)
        center = np.mean(dataSet[indexes ,:], axis=0)
        centers.append(center)
        dists[indexes] = sum((dataSet[indexes ,:] - np.repeat([center], sum(indexes), axis=0))**2, axis=1)
    return np.array(centers), np.stack([labels, dists]).T

def optics_lib(dataSet, eps, minPts):
    clustering = OPTICS(eps = eps, min_samples=minPts).fit(dataSet)
    labels = clustering.labels_
    centers = []
    dists = np.array([0]*dataSet.shape[0])
    unique_classes, indices = np.unique(labels, return_inverse=True)
    print('unique class num: ', len(unique_classes))
    for c in range(len(unique_classes)):
        indexes = list(indices == c)
        center = np.mean(dataSet[indexes ,:], axis=0)
        centers.append(center)
        dists[indexes] = sum((dataSet[indexes ,:] - np.repeat([center], sum(indexes), axis=0))**2, axis=1)
    return np.array(centers), np.stack([labels, dists]).T

def plot_cluster(data_mat, cluster_assment, centroid):
    """
    @brief      plot cluster and centroid
    @param      data_mat        The data matrix
    @param      cluster_assment  The cluste assment
    @param      centroid        The centroid
    @return     
    """
    plt.figure(figsize=(15, 6), dpi=80)
    plt.subplot(121)
    plt.plot(data_mat[:, 0], data_mat[:, 1], 'o')
    plt.title("source data", fontsize=15)
    plt.subplot(122)
    k = np.shape(centroid)[0]
    colors = [plt.cm.get_cmap("Spectral")(each) for each in np.linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = data_mat[np.nonzero(cluster_assment[:,0].A == i)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)
    for i in range(k):
        plt.plot(centroid[:,0], centroid[:,1], '+', color = 'k', markersize=18)
    plt.title("K-Means Cluster, k = 3", fontsize=15)
    plt.show()

