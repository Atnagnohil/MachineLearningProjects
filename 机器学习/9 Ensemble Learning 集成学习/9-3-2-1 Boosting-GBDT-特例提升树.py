# _*_ coding: utf-8 _*_
'''
时间:      2025/7/16 22:37
@author:  andinm
'''

from sklearn.tree import DecisionTreeRegressor
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
    #### 1、定义第一课树(最大深度设定为5)，并进行训练
    tree1 = DecisionTreeRegressor(max_depth=5)
    tree1.fit(X, y)
    # 损失函数为平方函数
    # 残差即为对应的负梯度
    # 计算残差 将残差带入tree2
    y2 = y-tree1.predict(X).reshape(-1, 1)
    tree2 = DecisionTreeRegressor(max_depth=5)
    tree2.fit(X, y2)

    y3 = y2 - tree2.predict(X).reshape(-1, 1)
    tree3 = DecisionTreeRegressor(max_depth=5)
    tree3.fit(X, y3)
    return tree1, tree2, tree3
def predict(tree1, tree2, tree3, X, y, k):
    # 测试前k条数据
    XPre = X[:k, :]
    yPre = sum(tree.predict(XPre) for tree in (tree1, tree2, tree3))
    print(f"{y[:k, ].reshape(1, -1)[0]}")
    print(f"{yPre[:k, ]}")
    # 打印预测产生的误差
    print(f"残差{yPre[:k, ] - y[:k, ].reshape(1, -1)[0]}")
'''[[17.592   9.1302 13.662  11.854   6.8233]]
[17.61560196  9.15380196 12.831       4.57199973  6.68971688]
残差[[ 0.02360196  0.02360196 -0.831      -7.28200027 -0.13358312]]
'''
# 绘制原图
def plot(X, y):
    plt.scatter(X, y)
    plt.show()

if __name__ == '__main__':
    X, y = loadData()
    # print(X.shape, y.shape)
    tree1, tree2, tree3 = trainModel(X, y)
    predict(tree1, tree2, tree3, X, y, 5)
    plot(X, y)

