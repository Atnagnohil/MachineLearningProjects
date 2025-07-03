# _*_ coding: utf-8 _*_
'''
时间:      2025/6/16 12:21
@author:  andinm
'''
from sklearn import tree
import numpy as np
from sklearn.tree import export_graphviz
import graphviz

def loadData():
    dataSet = [[0, 0,0,0, 'no'],
               [0, 0,0,1,'no'],
               [0, 1,0,1, 'yes'],
               [0, 1,1,0, 'yes'],
               [0, 0,0,0, 'no'],
               [1, 0,0,0, 'no'],
               [1, 0,0,1, 'no'],
               [1, 1,1,1, 'yes'],
               [1, 0,1,2, 'yes'],
               [1, 0,1,2, 'yes'],
               [2, 0,1,2, 'yes'],
               [2, 0,1,1, 'yes'],
               [2, 1,0,1, 'yes'],
               [2, 1,0,2, 'yes'],
               [2, 0,0,0,'no']]
    feature_name = ['age','job','house','credit']
    return dataSet, feature_name


'''
Sklearn中实现的决策树都是二叉树

DecisionTreeClassifier的常用参数含义：
- criterion：‘gini’ or ‘entropy’ (default=”gini”)，前者是基尼系数，后者是信息熵。
- max_depth：决策树最大深度。常用来解决过拟合。
- min_impurity_decrease：这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益)小于这个阈值，则该节点不再生成子节点。
- min_samples_split：如果是 int，则取传入值本身作为最小样本数； 如果是 float，则用 ceil(min_samples_split * 样本数量) 的值作为最小样本数，即向上取整。
- min_samples_leaf：如果是 int，则取传入值本身作为最小样本数； 如果是 float，则去 ceil(min_samples_leaf * 样本数量) 的值作为最小样本数，即向上取整。 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。
- max_leaf_nodes：最大叶子节点数。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
- min_impurity_split：决策树在创建分支时，信息增益（基尼系数）必须大于这个阀值，否则不分裂
'''
myData, feature_name = loadData()
X = np.array(myData)[:,0:4]
y = np.array(myData)[:,-1]
model = tree.DecisionTreeClassifier()
model.fit(X,y)
print(model.predict([[1,1,0,1]]))


export_graphviz(
    model,
    out_file="tree.dot",
    feature_names=feature_name,
    class_names=['yes','no'],
    rounded=True,
    filled=True
)

with open("tree.dot") as f:
    dot_grapth = f.read()
dot = graphviz.Source(dot_grapth)
dot.view()

