# _*_ coding: utf-8 _*_
'''
时间:      2025/7/11 20:28
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
def pca(X, K):
    # 对原始数据进行SVD分解
    u,s,vTranspose = np.linalg.svd(X)
    ''' 
        需要完整分解	full_matrices=True (默认)
        大型矩阵，节省内存	full_matrices=False
        只需要奇异值	compute_uv=False
        对称/Hermitian矩阵	hermitian=True
        一般矩阵	使用默认参数
    '''
    return X @ (vTranspose.T[:, 0:K]), u, s, vTranspose

# 还原（已经降低维度的）数据
def recoverData(Z, vTranspose, K):
    V_reduce = vTranspose[:,0:K]
    X_rec = Z @ V_reduce.T
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
    Z, u, s, vTranspose = pca(X, K)
    '''Z是压缩后的数据'''
    X_rec = recoverData(Z, vTranspose, K)
    '''X_rec是通过压缩后的数据展开到二维进行可视化'''
    plotData(X, X_rec)























