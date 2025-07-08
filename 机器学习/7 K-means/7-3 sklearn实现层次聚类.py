# _*_ coding: utf-8 _*_
'''
时间:      2025/7/8 19:38
@author:  andinm
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.cluster import AgglomerativeClustering


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
    model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')
    model.fit(X)
    return  model.labels_

def draw(labels):
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cm_dark, s=20)
    plt.show()


if __name__ == "__main__":
    X = loadData()
    # print(X.shape)
    labels = trainModel(X)
    print(labels)
    draw(labels)






