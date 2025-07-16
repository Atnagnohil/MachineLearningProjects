# _*_ coding: utf-8 _*_
'''
时间:      2025/7/15 18:10
@author:  andinm
'''

''' 如果基分类器是决策树，那么就是随机森林
'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
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
def trainModel1(X, y):
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
        n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1,
        oob_score=True
    )
    # 因为考虑到概率问题
    # 大约有37.8%的样本没法选中 oob即为out of bagging     因此oob_score表示没有选择的样本的分数
    bag_clf.fit(X, y)
    print(f"在oob上的预测精准度{bag_clf.oob_score_:0.4%}")

    '''# 如果oob_score=False
    则可以
    y_hat = bag_clf.predict(X)
    print(bag_clf.__class__.__name__,'=',accuracy_score(y,y_hat))
    '''

'''在oob上的预测精准度96.0000%
'''
def trainModel2(X, y):
    '''调用随机森林的API实现'''
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, oob_score=True)
    rnd_clf.fit(X, y)
    print(f"在oob上的预测精准度{rnd_clf.oob_score_:0.4%}")
if __name__ == "__main__":
    X, y = loadData()
    trainModel1(X, y)
    trainModel2(X, y)
