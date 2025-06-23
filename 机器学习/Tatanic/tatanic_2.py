# _*_ coding: utf-8 _*_
'''
时间:      2025/6/22 16:55
@author:  andinm
'''

import numpy as np
import joblib
from tatanic_1 import loadData, processData
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz
import graphviz

dfTrain, dfTest = loadData()
X, y, XPre, ComIdx, PreIdx = processData(dfTrain)

# 直接预测将Xre补全
# 加载模型
model = joblib.load("tatanic_ridge_model_pipeline.pkl")
yPre = model.predict(XPre)
# print(yPre)
# 接下来将这个年龄根据对应的下标补全到整个数据集中
dfTrain.loc[PreIdx, "Age"] = np.maximum(0, yPre)
# print(dfTrain.to_numpy().tolist())
# print(dfTrain.isna().sum())


# 需要将男性， 女性编码处理
dfTrain["Sex"] = dfTrain["Sex"].map({"female":1, "male":0})

# 这样训练集的数据基本补全完整
# Cabin: 船舱号
# Embarked: 登船港口 (C, Q, S)
# 并且不考虑进行决策时和这个有关
# 使用决策树
# 准备要分裂数据的标签
featuresForClassification = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
targetForClassification = "Survived"
# 提取特征目标和特征变量
X_clf = dfTrain[featuresForClassification].copy()
y_clf = dfTrain[targetForClassification].copy()
# 确保 'Fare' 列没有缺失值（虽然 train.csv 中 'Fare' 没有缺失，但以防万一）
if X_clf['Fare'].isna().any():
     X_clf['Fare'] = X_clf['Fare'].fillna(X_clf['Fare'].median())
# with pd.option_context("display.max_rows", None, "display.max_columns", None):
#     print(dfTrain[featuresForClassification])

'''此时已经完成对数据的编码处理'''
# 经过数据处理和编码 树据都为数值类型

# 数据分割
X_clf_train, X_clf_test ,y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# print(y_clf_train.value_counts()/len(y_clf_train))
# print(y_clf_test.value_counts()/len(y_clf_test))
'''Survived
0    0.616573
1    0.383427
Name: count, dtype: float64
Survived
0    0.614525
1    0.385475
Name: count, dtype: float64
'''
treeModel = DecisionTreeClassifier(random_state=42)

# 定义网格参数
parame_grid_tree = {
    "max_depth": [None, 5, 7, 10, 12],
    "min_samples_leaf": [1, 2, 3, 5, 7, 10],   # 叶子节点最小样本数
}

grid_search_tree = GridSearchCV(
    estimator=treeModel,
    param_grid=parame_grid_tree,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_tree.fit(X_clf_train, y_clf_train)
print(f"\n决策树的最佳参数: {grid_search_tree.best_params_}")
print(f"\n决策树最佳交叉验证的准确率: {grid_search_tree.best_score_}")


bestTreeModel = grid_search_tree.best_estimator_
y_clf_test_pre = bestTreeModel.predict(X_clf_test)
print(f"\n决策树在测试集上面的准确率{accuracy_score(y_clf_test, y_clf_test_pre): 0.4f}")
print(f"\n决策树分类报告: ")
print(classification_report(y_clf_test_pre, y_clf_test))

'''
决策树的最佳参数: {'max_depth': 5, 'min_samples_leaf': 5}

决策树最佳交叉验证的准确率: 0.8090416625627892

决策树在测试集上面的准确率 0.7821

决策树分类报告: 
              precision    recall  f1-score   support

           0       0.85      0.80      0.83       117
           1       0.67      0.74      0.70        62

    accuracy                           0.78       179
   macro avg       0.76      0.77      0.77       179
weighted avg       0.79      0.78      0.78       179

'''



'''
决策树的最佳参数: {'max_depth': 7, 'min_samples_leaf': 1}

决策树最佳交叉验证的准确率: 0.8119078104993598

决策树在测试集上面的准确率 0.7989

决策树分类报告: 
              precision    recall  f1-score   support

           0       0.91      0.79      0.85       126
           1       0.62      0.81      0.70        53

    accuracy                           0.80       179
   macro avg       0.77      0.80      0.78       179
weighted avg       0.82      0.80      0.81       179

'''

export_graphviz(
    bestTreeModel,
    out_file="tree.dot",
    feature_names=featuresForClassification,
    class_names=['yes','no'],
    rounded=True,
    filled=True
)

with open("tree.dot") as f:
    dot_grapth = f.read()
dot = graphviz.Source(dot_grapth)
dot.view()


