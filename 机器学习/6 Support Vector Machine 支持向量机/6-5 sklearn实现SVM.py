# _*_ coding: utf-8 _*_
'''
时间:      2025/7/5 18:30
@author:  andinm
'''
'''本项目当前只考虑线性核'''
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

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
    param_grid = {
        'C': np.arange(10, 1000, 100)  # 正则化参数
    }
    # 创建SVM分类器
    grid_search = GridSearchCV(
        estimator=LinearSVC(),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1   # 输出详细的工作日志
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# 在测试集测试精确度
def accessModel(X, y, model):
    yhat = model.predict(X)
    return accuracy_score(y, yhat)

def plot(w, b):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=np.array(y).squeeze(), cmap=cm_dark, s=30)
    # 画决策平面
    x0 = np.arange(-2.0, 12.0, 0.1)
    x1 = (-w[0][0] * x0 - b) / w[0][1]
    ax.plot(x0, x1.reshape(-1, 1))

    # 画间隔平面
    pos0 = np.arange(-2.0, 12.0, 0.1)
    pos1 = (1 - w[0][0] * pos0 - b) / w[0][1]
    ax.plot(pos0, pos1.reshape(-1, 1), color="green")

    neg0 = np.arange(-2.0, 12.0, 0.1)
    neg1 = (-1 - w[0][0] * neg0 - b) / w[0][1]
    ax.plot(neg0, neg1.reshape(-1, 1), color="green")

    ax.axis([-2, 12, -8, 6])
    plt.show()

if __name__ == "__main__":
    X, y = loadDataSet('Data/svm1.txt')
    XTrain, XTest, yTrain, yTest = dataSplit(X, y)
    model, param, trainScore = trainModel(XTrain, yTrain)
    b, omega = model.intercept_, model.coef_
    testScore = accessModel(XTest, yTest, model)
    print(f"最佳参数: {param}")
    print(f"最佳交叉验证正确率: {trainScore:.4%}")
    print(f"在测试集模型的正确率: {testScore: .4%}")
    print(f"b = {b}")
    print(f"omega = {omega}")
    plot(omega, b)
