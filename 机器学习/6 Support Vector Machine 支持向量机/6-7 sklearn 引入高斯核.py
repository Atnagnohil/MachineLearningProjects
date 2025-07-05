# _*_ coding: utf-8 _*_
'''
时间:      2025/7/5 21:16
@author:  andinm
'''


import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.patches import Circle

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# 数据分割
def dataSplit(X, y, proportion=0.2):
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=proportion, random_state=42, stratify=y
    )
    return XTrain, XTest, yTrain, yTest

# 模型训练
def trainModel(X, y):
    # 传入训练数据
    # 定义参数网格
    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100],  # 正则化参数
        'gamma': [0.01, 0.1, 1, 10],  # 核函数系数
    }

    # 创建SVM分类器
    svc = SVC(probability=True, kernel="rbf")

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='accuracy',  # 评估指标
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1  # 输出详细日志
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# 在测试集测试精确度
def accessModel(X, y, model):
    yhat = model.predict(X)
    return accuracy_score(y, yhat)

def plot(X, y, model):
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 画决策边界
    h = .01
    x_min, x_max = np.array(X)[:, 0].min() - .5, np.array(X)[:, 0].max() + .5
    y_min, y_max = np.array(X)[:, 1].min() - .5, np.array(X)[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, z, cmap=plt.cm.Paired)

    # 画数据点
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=np.array(y).squeeze(), cmap=cm_dark, s=30)
    # 画支持向量
    alphas_non_zeros_index = model.support_
    for i in alphas_non_zeros_index:
        circle = Circle((X[i][0], X[i][1]), 0.05, facecolor='none', edgecolor=(0.7, 0.7, 0.7), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    plt.show()

if __name__ == "__main__":
    X, y = loadDataSet('Data/svm2.txt')
    XTrain, XTest, yTrain, yTest = dataSplit(X, y)
    model, param, trainScore = trainModel(XTrain, yTrain)
    b = model.intercept_
    testScore = accessModel(XTest, yTest, model)
    print(f"最佳参数: {param}")
    print(f"最佳交叉验证正确率: {trainScore:.4%}")
    print(f"在测试集模型的正确率: {testScore: .4%}")
    print(f"b = {b}")
    print('各类别各有多少个支持向量', model.n_support_)
    print('各类别的支持向量在训练样本中的索引', model.support_)
    print('各类所有的支持向量', model.support_vectors_)
    print('支持向量的alpha值', model.dual_coef_)
    plot(X, y, model)
'''
最佳参数: {'C': 1, 'gamma': 1}
最佳交叉验证正确率: 100.0000%
在测试集模型的正确率:  95.0000%
b = [-2.18653362]
各类别各有多少个支持向量 [18 16]
各类别的支持向量在训练样本中的索引 [ 0  5  8 21 22 23 35 36 43 44 45 47 53 59 64 69 74 78  3 12 13 16 19 20
 33 37 42 48 51 56 58 63 68 77]
各类所有的支持向量 [[-0.307768  0.503038]
 [-0.403483  0.474466]
 [-0.557789  0.375797]
 [-0.257542 -0.753885]
 [ 0.523902 -0.436156]
 [ 0.595222  0.20957 ]
 [ 0.249281 -0.71184 ]
 [ 0.61908  -0.088188]
 [-0.689946 -0.508674]
 [-0.356048  0.53796 ]
 [ 0.541359 -0.205969]
 [ 0.153738  0.491531]
 [-0.755431  0.096711]
 [-0.7016    0.190983]
 [ 0.577939  0.403784]
 [ 0.193449  0.574598]
 [ 0.297885 -0.632874]
 [ 0.334204  0.381237]
 [-0.448939  0.176725]
 [ 0.268066 -0.071621]
 [-0.392868 -0.125261]
 [ 0.068286  0.392741]
 [ 0.3843   -0.17657 ]
 [-0.015165  0.359326]
 [-0.293382 -0.139778]
 [-0.250001  0.141621]
 [-0.045677  0.172907]
 [ 0.353588 -0.070617]
 [ 0.199894 -0.199381]
 [-0.203272  0.286855]
 [ 0.394164 -0.058217]
 [ 0.253289 -0.285861]
 [ 0.035398 -0.333682]
 [ 0.229465  0.250409]]
支持向量的alpha值 [[-1.         -1.         -1.         -1.         -1.         -1.
  -0.96274778 -1.         -0.63607763 -0.37425501 -1.         -1.
  -0.29080443 -1.         -0.19949299 -1.         -1.         -1.
   1.          1.          1.          1.          1.          1.
   1.          1.          0.46337783  1.          1.          1.
   1.          1.          1.          1.        ]]
'''




