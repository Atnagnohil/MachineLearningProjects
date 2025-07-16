# _*_ coding: utf-8 _*_
'''
时间:      2025/7/15 17:02
@author:  andinm
'''
'''
软投票分类器是基于概率 因此对应的基分类器要有概率
'''

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 得到数据
def loadData():
    X, y = make_moons(n_samples=7000, noise=0.1)
    return X, y
# 数据集分割
def splitModel(X, y):
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    return XTrain, XTest, yTrain, yTest

# 模型训练
def trainModel(XTrain, XTest, yTrain, yTest):
    # 创建基分类器
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    svm = SVC(probability=True)
    voting = VotingClassifier(
        estimators=[('lr', lr), ('dt', dt), ('svm', svm)],
        voting="soft"
    )
    for clf in (lr, dt, svm, voting):
        clf.fit(XTrain, yTrain)
        y_hat = clf.predict(XTest)
        ''''''
        print(clf.__class__.__name__, '=', accuracy_score(yTest, y_hat))
'''
LogisticRegression = 0.8835714285714286
DecisionTreeClassifier = 0.9985714285714286
SVC = 0.9992857142857143
VotingClassifier = 0.9992857142857143
'''
# 绘制图像
def plot(X):
    plt.scatter(X[:, 0], X[:,1]), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    X, y = loadData()
    # plot(X)
    XTrain, XTest, yTrain, yTest = splitModel(X, y)
    trainModel(XTrain, XTest, yTrain, yTest)
