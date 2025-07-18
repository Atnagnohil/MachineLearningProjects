# _*_ coding: utf-8 _*_
'''
时间:      2025/7/18 16:27
@author:  andinm
'''
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
import os
from sklearn.metrics import accuracy_score

'''
#### 读取数据

XGBoost中数据形式可以是libsvm的，libsvm作用是对稀疏特征进行优化，看个例子：

```
1 101:1.2 102:0.03 
0 1:2.1 10001:300 10002:400
0 2:1.2 1212:21 7777:2
```
每行表示一个样本，每行开头0，1表示标签，而后面的则是特征索引：数值，其他未表示都是0.

我们以判断蘑菇是否有毒为例子来做后续的训练。数据集来自：http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/ ，
其中蘑菇有22个属性，将这些原始的特征加工后得到126维特征，并保存为libsvm格式，标签是表示蘑菇是否有毒。
'''


# 数据处理
def loadDataSet():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(__file__)

    # 构建完整路径
    train_path = os.path.join(current_dir, 'Data', 'agaricus.txt.train')
    test_path = os.path.join(current_dir, 'Data', 'agaricus.txt.test')

    # 使用 sklearn 加载 LibSVM 文件
    X_train, y_train = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)

    # 转换为 DMatrix
    data_train = xgb.DMatrix(X_train, label=y_train)
    data_test = xgb.DMatrix(X_test, label=y_test)

    return data_train, data_test
# 训练模型
def trainModel():
    param = {'max_depth': 3, 'eta': 0.3, 'objective': 'binary:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 6
    model = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    return model
# 评估
def predict(model, data_test):
    y_hat = model.predict(data_test)
    y_pred = y_hat.copy()
    y_pred[y_hat >= 0.5] = 1
    y_pred[y_hat < 0.5] = 0
    y = data_test.get_label()
    print(f'accuracy_score={accuracy_score(y, y_pred):0.2%}')


# 绘图

def draw(model):
    graph = xgb.to_graphviz(model, tree_idx=0)
    graph.render(filename='xgboost_tree', format='png')

    # 显示图形
    graph.view()


if __name__ == '__main__':
    data_train, data_test = loadDataSet()
    model = trainModel()
    predict(model, data_test)
    draw(model)





