# _*_ coding: utf-8 _*_
'''
时间:      2025/7/6 12:40
@author:  andinm
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# 加载数据
def loadData():
    data = np.loadtxt(r"Data/cluster_data.csv", delimiter=",")
    return data

# 欧式距离
def euclideanMetric(x1, x2):
    '''计算x1和x2的欧式距离'''
    return np.sqrt(np.sum(np.pow(x1 -x2, 2)))

# 初始化选择K个质点
# 不考虑随机选取 收敛太慢， 可以再已经存在的点中随机抽取K个点
def initCentroids(X, k):
    '''随机从
        X从选择k个点'''
    idx = np.random.randint(0, len(X)-1, k)
    return X[idx]

# kmeans算法第一步
# 对每个样本点计算得到距离最近的质心， 将标签设置为对应质心
def findClosestCentroids(X, centroids):
    # 初始化一个下标组合 存储每个数据对应的质心
    idx = np.zeros(len(X)).reshape(X.shape[0], -1)
    for i in range(X.shape[0]):
        # 遍历每一个数据点
        minDis = float("inf")   # 初始化最小距离
        minIdx = -1             # 最小距离对应的下标
        for k in range(centroids.shape[0]):
            # 遍历每一个质点
            # 计算欧式距离
            curDis = euclideanMetric(X[i], centroids[k])
            if curDis < minDis:
                minDis = curDis
                minIdx = k
        idx[i] = minIdx
    return idx
# kmeans算法第一步
# 重新计算k个cluser对应的质心
def computeCentroids(X, idx):
    # idx为大小 0到k-1的 二维数组
    # 每一个数据在idx有对应标签
    kIdx = set(np.ravel(idx).tolist())  # 找到所有聚类中心索引
    # np.ravel展平为一维数组 set去重
    kIdx = list(kIdx)
    centroids = np.zeros(shape=(len(kIdx), X.shape[1]))    # X.shape[1]为特征数
    for i in range(len(kIdx)):
        smapleOfCur = X[np.where(idx==kIdx[i])[0]]
        centroids[i] = np.sum(smapleOfCur, axis=0) / len(smapleOfCur)   # 沿列方向求和
    return centroids

def kmeans(X, k, maxIters):
    centroids = np.zeros(shape=(k, X.shape[1]))
    idx = np.zeros(shape=(X.shape[0], 1))
    for i in range(maxIters):
        if i == 0:
            centroids = initCentroids(X, k)
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx)
    return idx, centroids

def draw(X, centroids, idx):
    # 定义映射空间
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.scatter(X[:, 0], X[0:, 1], cmap=cm_dark, c=np.ravel(idx), s=20)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(len(centroids)), cmap=cm_dark, marker='*', s=500)
    plt.show()

if __name__ == "__main__":
    X = loadData()
    # print(X.shape)
    # print(data)
    idx, centroids = kmeans(X, 3, 100)
    print('聚类中心:',centroids)
    print('每个样本所属的簇:', np.ravel(idx))
    draw(X, centroids, idx)

'''
聚类中心: [[3.04367119 1.01541041]
 [1.95399466 5.02557006]
 [6.03366736 3.00052511]]
每个样本所属的簇: [1. 2. 2. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 2. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.
 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 0.
 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.
 2. 0. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2.
 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 2. 1.]

'''







