# _*_ coding: utf-8 _*_
'''
时间:      2025/7/11 20:41
@author:  andinm
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def loadData():
    data = np.loadtxt("Data/pca_data.csv", delimiter=",")
    return data
def trainModel(X, K):
    model = PCA(n_components=K)
    Z = model.fit_transform(X)  #生成降维后数据
    return Z, model
def recoverData(Z, model):
    XRec = model.inverse_transform(Z)
    return XRec
def plotData(XOrignal, XRec):
    plt.scatter(XOrignal[:, 0], XOrignal[:, 1], label="orign")
    plt.scatter(XRec[:, 0], XRec[:, 1], c="r", label="recover")
    plt.xlabel("X1", fontsize=15, color="black")
    plt.ylabel("X2", fontsize=15, color="black")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, fontsize=15, loc="upper left")
    plt.show()

if __name__ == "__main__":
    X = loadData()
    K = 1
    Z, model = trainModel(X, K)
    XRec = recoverData(Z, model)
    plotData(X, XRec)
    print(f"主成分个数 = {model.n_components}")
    print(f"贡献比 = {np.sum(model.explained_variance_ratio_):0.3%}")
    print(f"特征的方差 = {np.sum(model.explained_variance_):0.4f}")






