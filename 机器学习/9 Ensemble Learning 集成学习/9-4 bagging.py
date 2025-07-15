# _*_ coding: utf-8 _*_
'''
时间:      2025/7/15 17:04
@author:  andinm
'''

'''
不同与Voting在相同的数据集上使用不同的分类器
Bagging是使用相同的分类器 使用不同的模型
'''

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris


# 加载数据
def loadData():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y
'''
    bootstrap = True 为bagging，
    bootstrap = False为pasting
    max_samples设置为整数表示的就是采样的样本数，
    设置为浮点数表示的是max_samples*x.shape[0]
'''
def trainModel(X, y):
    bag_clf = BaggingClassifier(
        SVC(),
        n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
        ,oob_score=True
    )
    # 因为考虑到概率问题
    # 大约有37.8%的样本没法选中 oob即为out of bagging     因此oob_score表示没有选择的样本的分数
    bag_clf.fit(X, y)

    print(f"在oob上的预测精准度{bag_clf.oob_score_:0.4%}")
'''在oob上的预测精准度96.6667%'''
if __name__ == "__main__":
    X, y = loadData()
    trainModel(X, y)
