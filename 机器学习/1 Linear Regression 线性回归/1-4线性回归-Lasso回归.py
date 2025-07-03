# _*_ coding: utf-8 _*_
'''
时间:      2025/5/26 18:20
@author:  andinm
'''
import numpy as np
import matplotlib.pyplot as plt


'''predict_ans 和 h_theta 相等'''
# 读取数据
def loadData():
    data = np.loadtxt(r"Data/data1.txt", delimiter=',')
    n = data.shape[1] - 1
    X = data[:, 0:n]
    y = data[:, -1].reshape(-1, 1)   # 转换为列向量
    return X, y

# 特征归一化
def featureNormalize(X):
    mu = np.average(X, axis=0)  # 平均值
    sigma = np.std(X, axis=0, ddof=1)
    X = (X - mu) / sigma
    return X, mu, sigma

# 计算损失函数
def computeCost(X, y, theta, lamda=0.02):
    m = X.shape[0]
    return np.sum(np.pow(np.dot(X, theta) - y, 2)) / (2*m) + (lamda * np.sum(np.fabs(theta)))

# 预测数据
def predictData(X, theta, mu, sigma):
    X = (X - mu) / sigma
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X, 0, values=c, axis=1)
    return np.dot(X, theta)

# 带有L1正则化的线性回归
def gradientDescentLasso(X, y, theta, iteratinos, lamda=0.02):
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X, 0, values=c, axis=1)
    m, n = X.shape
    costs = np.zeros(iteratinos)
    for num in range(iteratinos):
        for k in range(n):
            # 计算z_k
            z_k = np.sum(X[:, k] ** 2)
            # p_k = 0
            # for i in range(m):
            #     p_k += X[i, k] * (y[i] - np.sum([theta[j] * X[i, j] for j in range(n) if j != k]))
            # 向量化计算 p_k
            residual = y - X @ theta + theta[k] * X[:, k].reshape(-1, 1)
            p_k = X[:, k].T @ residual
            if p_k < -(lamda / 2):
                theta[k] = (p_k + lamda/2) / z_k
            elif p_k > (lamda / 2):
                theta[k] = (p_k - lamda/2) / z_k
            else:
                theta[k] = 0
        costs[num] = computeCost(X, y, theta)
    return theta, costs

# 计算均方误差
def calMSE(y_true, y_predict):
    m = y_true.shape[0]
    return np.sum(np.pow(y_true - y_predict,2)) / m
# 计算均方根误差
def calRMSE(y_true, y_predict):
    m = y_true.shape[0]
    return np.sqrt(np.sum(np.pow(y_true - y_predict, 2)) / m)
# 计算平均绝对误差
def calMAE(y_true, y_predict):
    m = y_true.shape[0]
    return np.sum(np.fabs(y_true - y_predict)) / m

XOrignal, y = loadData()
X, mu, sigma = featureNormalize(XOrignal)
theta = np.zeros(X.shape[1] + 1).reshape(-1,1)
iterations = 400
theta, costs = gradientDescentLasso(X, y, theta, iterations)

f_x = theta[0] + theta[1]*X[:, 0]
plt.scatter(X, y)
plt.plot(X, f_x)
plt.show()

# 画损失函数
x_axis = np.linspace(1, iterations, iterations)
plt.plot(x_axis, costs[0: iterations], color="black")
plt.show()



predict_ans = predictData(XOrignal, theta, mu, sigma)  # 预测结果
print(f"MSE = {calMSE(y, predict_ans)}")
print(f"RMSE = {calRMSE(y, predict_ans)}")
print(f"MAE = {calMAE(y, predict_ans)}")









