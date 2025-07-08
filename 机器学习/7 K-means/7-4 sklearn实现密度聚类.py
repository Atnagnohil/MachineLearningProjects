# _*_ coding: utf-8 _*_
'''
时间:      2025/7/8 19:42
@author:  andinm
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.cluster import DBSCAN

def loadData():
    data = np.loadtxt('data/cluster_data.csv',delimiter=',')
    return data

def trainModel(X):
    # 注意sklearn 0.20以上版本才支持linkage='single'
    '''### linkage可取值：
    - ward： 最小方差
    - complete 最大距离
    - average 平均距离
    - single 最小距离'''
    model = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
    model.fit(X)
    print('每个样本所属的簇:\n', model.labels_)
    return  model.labels_

def draw(labels):
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cm_dark, s=20)
    plt.show()


if __name__ == "__main__":
    X = loadData()
    # print(X.shape)
    labels = trainModel(X)
    draw(labels)

