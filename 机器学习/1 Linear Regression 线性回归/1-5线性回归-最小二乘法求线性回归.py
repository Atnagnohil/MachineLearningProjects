# _*_ coding: utf-8 _*_
'''
时间:      2025/5/26 19:28
@author:  andinm
'''




import numpy as np
import matplotlib.pyplot as plt

def loadData():
    data = np.loadtxt(r"Data/data1.txt", delimiter=',')
    n = data.shape[1] - 1
    X = data[:, 0:n]
    y = data[:, -1].reshape(-1,1)
    return X, y


# 预测数据
def predictData(X, theta):
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X, 0, values=c, axis=1)  # 在数据前插入第一行
    return np.dot(X, theta)

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


Xorignal, y = loadData()
X = np.insert(Xorignal, 0, 1, axis=1)
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

plt.scatter(Xorignal,y)
h_theta = theta[0]+theta[1]*Xorignal
plt.plot(Xorignal,h_theta)
plt.show()

'''predict_ans 和 h_theta 相等'''
predict_ans = predictData(Xorignal, theta)  # 预测结果
print(f"MSE = {calMSE(y, h_theta)}")
print(f"RMSE = {calRMSE(y, h_theta)}")
print(f"MAE = {calMAE(y, h_theta)}")


