# _*_ coding: utf-8 _*_
'''
时间:      2025/7/6 13:30
@author:  andinm
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans


def loadData():
    data = np.loadtxt(r"Data/cluster_data.csv", delimiter=",")
    return data

def trainModel(X, k, maxIters):
    model = KMeans(n_clusters=k, max_iter=maxIters)
    model.fit(X)
    '''KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=10,
       n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)'''
    return model.cluster_centers_, model.labels_

def draw(X, centroids):
    # 定义映射空间
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.scatter(X[:, 0], X[0:, 1], cmap=cm_dark, c=np.ravel(idx), s=20)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(len(centroids)), cmap=cm_dark, marker='*', s=500)
    plt.show()


if __name__ == "__main__":
    X = loadData()
    # print(X.shape)
    # print(data)
    centroids, idx = trainModel(X, 3, 100)
    print('聚类中心:', centroids)
    print('每个样本所属的簇:', idx)
    draw(X, centroids)










