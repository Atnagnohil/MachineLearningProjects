# _*_ coding: utf-8 _*_
'''
时间:      2025/7/16 23:06
@author:  andinm
'''

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
def loadData():
    data = np.loadtxt('Data/data.txt', delimiter=',')
    X = data[:, 0:-1]
    y = data[:, -1].reshape(-1, 1)
    return X, y
# 模型训练
def trainModel(X, y):
    model = GradientBoostingRegressor(max_depth=5, n_estimators=3, learning_rate=1.0)
    model.fit(X, y.ravel())
    return model
def predict(model, X, y, k):
    # 测试前k条数据
    XPre = X[:k, :]
    yPre = model.predict(XPre)
    print(f"{y[:k, ].reshape(1, -1)[0]}")
    print(f"{yPre[:k, ]}")
    # 打印预测产生的误差
    print(f"残差{yPre[:k, ] - y[:k, ].reshape(1, -1)[0]}")

'''[17.592   9.1302 13.662  11.854   6.8233]
[17.61560196  9.15380196 12.831       4.57199973  6.68971688]
残差[ 0.02360196  0.02360196 -0.831      -7.28200027 -0.13358312]

'''
# 绘制原图
def plot(X, y):
    plt.scatter(X, y)
    plt.show()

if __name__ == '__main__':
    X, y = loadData()
    # print(X.shape, y.shape)
    model = trainModel(X, y)
    predict(model, X, y, 5)
    plot(X, y)


