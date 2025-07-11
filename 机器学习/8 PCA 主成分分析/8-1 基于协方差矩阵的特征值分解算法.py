# _*_ coding: utf-8 _*_
'''
时间:      2025/7/11 17:20
@author:  andinm
'''


import numpy as np
import matplotlib.pyplot as plt
# 加载数据
def loadData():
    data = np.loadtxt("Data/pca_data.csv", delimiter=",")
    return data
# 特征归一化
def featureNormal(X):
    mu = np.mean(X, axis=0).reshape(1, -1)  # 按照列方向求均值
    sigma = np.std(X,axis=0,ddof=1).reshape(1,-1)   # sigame是方差
    '''ddof = 1
    表示使用样本标准差（分母为m - 1）'''
    X = (X-mu)/sigma
    return X, mu, sigma
# 计算协方差矩阵
# 计算特征向量
def pca(X, K):
    ''' K 代表需要压缩到的维度
    '''
    # sigma 是协方差矩阵
    m, n = X.shape
    sigma = (X.T @ X) / (m - 1)
    # 计算协方差矩阵的特征向量 特征值
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    # 将特征值按照大小排序
    '''np.argsort() 的作用：返回数组值从小到大排序的索引位置'''
    index = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, index]
    u_reduce = eigenvectors[:, 0:K]
    return X @ u_reduce, eigenvectors
# 还原（已经降低维度的）数据
def recoverData(Z, U, K):
    U_reduce = U[:,0:K]
    print(U_reduce)
    X_rec = Z @ U_reduce.T
    return X_rec
# 绘制图像
def plotData(X_orgin,X_rec):
    plt.scatter(X_orgin[:,0],X_orgin[:,1])
    plt.scatter(X_rec[:, 0], X_rec[:, 1],c='red')
    plt.show()

if __name__ == "__main__":
    X = loadData()
    X, mu, sigma = featureNormal(X)
    K = 1
    Z, U = pca(X, K)
    '''Z是压缩后的数据'''
    X_rec = recoverData(Z, U, K)
    '''X_rec是通过压缩后的数据展开到二维进行可视化'''
    plotData(X, X_rec)












