# _*_ coding: utf-8 _*_
'''
时间:      2025/6/1 19:12
@author:  andinm
'''

from sklearn import linear_model
import numpy as np
# from sklearn.datasets import fetch_openml   # 从openml上加载数据的模块
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error  # 计算平均绝对误差
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import pandas as pd

'''直接从sklearn调用'''
# fetch = fetch_openml(name="iris", version=1)
# print(fetch.DESCR)

                                # 1 加载数据

'''从文件中读取'''
df = pd.read_excel('Data/boston.xls')
# print(df)
# print(df.columns)  # 获取所有列名
# print(df.columns[0:-1])

# X = df[df.columns[0:-1]]
# y = df[df.columns[-1]]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
                                # 2 数据预处理
                                # 3 数据可视化
                                # 4 模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 创建带标准化的Ridge回归管道
# make_pipeline 会按顺序创建管道：
# 第一步：StandardScaler() 对数据进行标准化处理 (均值为0，方差为1)
# 第二步：linear_model.Ridge() 使用岭回归模型进行训练和预测
ridge_pipe = make_pipeline(
    StandardScaler(),
    linear_model.Ridge()
)

# 5. 设置参数网格
# 定义要搜索的Ridge回归的alpha参数范围
# np.linspace(14.6, 14.8, 41) 会生成从14.6到14.8之间包含41个均匀分布的数值
param_test = {'ridge__alpha': np.linspace(14.6, 14.8, 41)}

# 6. 网格搜索
# 使用GridSearchCV进行超参数调优
# estimator: 要优化的模型管道
# param_grid: 参数网格
# scoring: 评估指标，'neg_mean_squared_error' 表示负均方误差，GridSearchCV会尝试最大化这个值 (即最小化均方误差)
# cv: 交叉验证的折数，这里是5折交叉验证
# n_jobs: 并行运行的作业数，-1表示使用所有可用的CPU核心以加速计算
gsearch = GridSearchCV(
    estimator=ridge_pipe,
    param_grid=param_test,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)
# 在训练数据上拟合GridSearchCV，执行网格搜索和交叉验证
gsearch.fit(X_train, y_train)

# 打印最佳参数和对应的交叉验证MSE
# gsearch.best_params_ 返回最佳超参数组合
# gsearch.best_score_ 返回使用最佳参数组合时交叉验证的平均得分
# 由于scoring是'neg_mean_squared_error'，所以需要取负值来得到实际的MSE
print(f"最佳参数: {gsearch.best_params_}, 交叉验证MSE = {-gsearch.best_score_:.4f}")
# {'ridge__alpha': np.float64(14.655)},  MSE = 25.76770574347892
'''1. gsearch.best_params_
含义：在参数搜索空间中表现最佳的超参数组合
内容：包含最佳参数名称和值的字典
格式：{参数名: 最佳值}
在代码中：会是类似 {'ridge__alpha': 0.1} 的形式
意义：告知我们哪一个正则化强度（alpha值）在交叉验证中表现最好

2. gsearch.best_score_
含义：使用最佳参数组合时达到的交叉验证分数
内容：一个浮点数
在代码中：由于设置了 scoring='neg_mean_squared_error':
返回的是负的均方误差值
实际意义是：-gsearch.best_score_ 才是真正的均方误差
公式：best_score = -平均(MSE)
意义：衡量模型预测质量（数值越接近0越好）'''


# 使用最佳模型直接评估测试集
# gsearch.best_estimator_ 是在整个训练集上用最佳参数重新训练过的模型管道
best_model = gsearch.best_estimator_
y_test_pred = best_model.predict(X_test)  # 自动处理标准化
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"测试集MSE: {test_mse:.4f}")

import joblib
joblib.dump(best_model, "boston_ridge_model_pipeline.pkl")


# 读取模型
moudle = joblib.load("boston_ridge_model_pipeline.pkl")
print(y_test.tolist())
y_pre = moudle.predict(X_test)  ## 注意：这里传入的是原始的X_test，因为管道内部会处理标准化
print(y_pre)
print('MSE=',mean_squared_error(y_test,y_pre))













