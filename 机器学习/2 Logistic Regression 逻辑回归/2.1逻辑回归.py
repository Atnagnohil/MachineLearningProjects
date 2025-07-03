# _*_ coding: utf-8 _*_
'''
时间:      2025/6/1 21:08
@author:  andinm
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score

# 加载数据
def loadData():
    data = np.loadtxt(r"Data/data1.txt",delimiter=',')
    n = data.shape[1] - 1
    X = data[:, 0:n]
    y = data[:, -1].reshape(-1, 1)   # -1为占位符，表示行数待定计算，，，1列数，，表示只有一行
    return X, y

#归一化
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# 画散点图
def plotPoints(X, y):
    pos = np.where(y==1) # positive表示通过考核的索引
    neg = np.where(y==0) # negetive表示未通过考核的索引
    plt.scatter(X[pos, 0], X[pos, 1], marker="x") # X[pos, 0]表示所有通过考核的行的第一科成绩， X[pos, 1]表示所有通过考核的行的第二科成绩
    plt.scatter(X[neg, 0], X[neg, 1], marker='o')
    plt.xlabel("Score 1")
    plt.ylabel("Score 2")
    plt.show()

# 实现sigmod函数
def sigmod(z):
    # 为了防止np.exp溢出或下溢，对z进行裁剪
    # 将z的值限制在-500到500之间，超出此范围的值将导致exp(-z)过大或过小
    z = np.clip(z, -500, 500)
    return 1/(1+np.exp(-z))

# 实现假设函数
def hypothesis(X, theta):
    return sigmod(X @ theta)  # @是显示乘号

# 实现损失函数
def computeCost(X, y, theta):
    m = X.shape[0]
    # 添加一个非常小的常数epsilon，以防止np.log(0)的错误
    epsilon = 1e-8
    return np.sum(-y*np.log(hypothesis(X, theta) + epsilon) - (1 - y) * np.log(1 - hypothesis(X, theta) + epsilon)) / m



# 梯度下降法求解
def gradientDescent(X,y,theta,iterations,alpha):
    X = np.insert(X, 0, values=1, axis=1)  # 在X的第0行前面 插入数字1（因为有广播机制） axis=1 表示插入一列
    m = X.shape[0]
    n = X.shape[1]
    costs = np.zeros(iterations)
    for num in range(iterations):
        for j in range(n):
            theta[j] = theta[j] - (alpha / m)*np.sum((hypothesis(X, theta) - y)*X[:, j].reshape(-1, 1))
        costs[num] = computeCost(X, y, theta)
    return theta, costs


# 预测函数

def predictData(X, theta):
    X_norm = (X - mu) / sigma
    X_norm = np.insert(X_norm, 0, 1, 1)
    h = hypothesis(X_norm, theta)
    # 根据概率值决定最终的分类,>=0.5为1类，<0.5为0类
    h[h >= 0.5] = 1
    h[h < 0.5] = 0
    return h

X_Orign, y = loadData()
m = X_Orign.shape[0]
n = X_Orign.shape[1] + 1
theta = np.zeros((n,1))
iterations = 10000
alpha = 0.004
X_std, mu, sigma = featureNormalize(X_Orign)
theta ,costs = gradientDescent(X_std, y, theta, iterations, alpha)
print(theta)
# 化决策边界
def plotDecisionBoundary(X, y, theta, mu, sigma):
    # 绘制原始图像
    plotPoints(X, y)

    # 将数据标准化
    X= (X - mu) / sigma
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.scatter(X[:, 0], X[:, 1], c=np.array(y).squeeze(), cmap=cm_dark, s=30)

    # 化分类决策面 theta0+theta1*x1+theta2*x2 = 0
    # x1 = np.arange(20,110,0.1)
    x1 = np.arange(X[:, 0].min(), X[:, 0].max(), 0.1)
    x2 = -(theta[1] * x1 + theta[0]) / theta[2]
    if theta[2] == 0:
        print("警告: theta[2]为零，无法将决策边界绘制为直线。")
        plt.show()
        return
    plt.plot(x1, x2)
    plt.show()
plotDecisionBoundary(X_Orign, y, theta, mu, sigma)

# 绘制损失函数图像
def plotCost(iterations, costs):
    X_axis = np.linspace(1, iterations, iterations)
    plt.plot(X_axis, costs, color="black")
    plt.show()
plotCost(iterations, costs)

p = predictData(X_Orign,theta)
print('准确度：',np.mean(p==y))
print('准确度：',accuracy_score(y,p))