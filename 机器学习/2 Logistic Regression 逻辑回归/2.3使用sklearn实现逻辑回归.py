# _*_ coding: utf-8 _*_
'''
时间:      2025/6/2 22:32
@author:  andinm
'''

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def loadData():
    data = np.loadtxt("Data/data1.txt", delimiter=',')
    n = data.shape[1] - 1
    X = data[:, 0:n]
    y = data[:, -1]
    return X, y
X_Orign, y = loadData()

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X_Orign)

'''
### 使用逻辑回归模型
常用参数含义：
- C
正则化参数lambda的倒数, 
C越大，惩罚越小，易过拟合，泛化能力差。
C越小，惩罚越大，不易过拟合，泛化能力好。
- multi_class: ovr, multinomial    # multi_class分类模型 2分类默认使用OvR   多分类使用MvM
'''
model = linear_model.LogisticRegression(C=50, max_iter=20000)
model.fit(X, y)
'''LogisticRegression(C=50, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=2000,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)'''
print(model.intercept_)  # 截距
print(model.coef_)       # 斜率
y_hat = model.predict(X)
print(f"模型准确率{accuracy_score(y_hat, y)}")

def plotDescisionBoundary(X,y,theta):
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.scatter(X[:, 0], X[:, 1], c=np.array(y).squeeze(), cmap=cm_dark, s=30)

    # 化分类决策面 theta0+theta1*x1+theta2*x2 = 0
    # x1 = np.arange(20,110,0.1)
    x1 = np.arange(min(X[:, 0]), max(X[:, 0]), 0.1)
    x2 = -(theta[1] * x1 + theta[0]) / theta[2]
    plt.plot(x1, x2)
    plt.show()
theta = np.append(model.intercept_, model.coef_)
plotDescisionBoundary(X, y, theta)







