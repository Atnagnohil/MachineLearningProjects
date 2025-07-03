# _*_ coding: utf-8 _*_
'''
时间:      2025/5/20 19:47
@author:  andinm
'''



'''
处理数据
'''
import numpy as np
import matplotlib.pyplot as plt
'''predict_ans 和 h_theta 相等'''
# 加载数据
def loadData():
    data = np.loadtxt(r"Data/data1.txt", delimiter=',')  # 分隔符为逗号
    n = data.shape[1] - 1  # n代表特征数 减去1表示把实际的函数值减去
    X = data[:,0:n]  # 所有行 第0到n-1列
    y = data[:, -1].reshape(-1,1)  # 将y转换为列向量
    return X, y

# 特征归一化
def featureNormalize(X):
    '''归一化有很多种方式，我们这里采取的方式是：对每一个特征，
    这列中的每个数据分别减去这列的均值，然后再除以这列的方差'''
    mu = np.average(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X = (X - mu) / sigma
    return X, mu, sigma


# 计算损失函数
def computeCost(X, y, theta):
    m = X.shape[0]
    return np.sum(np.pow(np.dot(X, theta) - y, 2)) / (2*m)

#预测数据
def predict(X):
    X = (X-mu)/sigma
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X, 0, values=c, axis=1)  # 在数据前插入第一行
    return np.dot(X,theta)

# 梯度下降
def gradientDescent(X, y, theta, iterations, alpha):
    c = np.ones(X.shape[0]).transpose() # 创建全为1的一个列向量
    X = np.insert(X, 0, values=c, axis=1)  # 在X最前中插入一个全为1的列向量
    m = X.shape[0]  # m行
    n = X.shape[1]
    costs = np.zeros(iterations)  # 创建和迭代次数一样的0矩阵
    for num in range(iterations):
        for j in range(n):
            theta[j] = theta[j] + (alpha / m) * np.sum((y - np.dot(X, theta)) * X[:, j].reshape(-1, 1))
        costs[num] = computeCost(X, y, theta)
    return theta,costs

# 计算均方误差
def calMSE(y_true, y_predict):
    '''计算均方误差'''
    m = y_true.shape[0]
    return np.sum(np.pow(y_true - y_predict, 2)) / m

# 计算均方根误差
def calRMSE(y_true, y_predict):
    '''计算均方误差'''
    m = y_true.shape[0]
    return np.sqrt(np.sum(np.pow(y_true - y_predict, 2)) / m)

# 计算平均绝对误差
def calMAE(y_true, y_predict):
    '''计算平均绝对误差'''
    m = y_true[0]
    return np.sum(np.abs(y_true - y_predict)) / m


X_roignal, y = loadData()
X, mu, sigma = featureNormalize(X_roignal)
theta = np.zeros(X.shape[1]+1).reshape(-1,1)  # 转换为列向量
iterations = 400
alpha = 0.1
theta,costs = gradientDescent(X, y, theta, iterations, alpha)

plt.scatter(X,y)
f = theta[0] + theta[1] * X[:,0]
# print(theta)
plt.plot(X,f)

plt.show()


# 画损失函数图
x_axis = np.linspace(1,iterations,iterations)
plt.plot(x_axis, costs[0:iterations], color="black")
plt.show()

predict_ans = predict(X_roignal)  # 预测结果
print(f"MSE = {calMSE(y, predict_ans)}")
print(f"RMSE = {calRMSE(y, predict_ans)}")
print(f"MAE = {calMAE(y, predict_ans)}")



