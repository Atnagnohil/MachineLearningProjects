# _*_ coding: utf-8 _*_
'''
时间:      2025/6/16 17:20
@author:  andinm
'''
# 获取数据
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error   # 计算均方误差
from sklearn.model_selection import train_test_split, GridSearchCV # 用来分割数据集
from sklearn.pipeline import make_pipeline
import numpy as np
import joblib


def loadData():
    dfTrain = pd.read_csv("Data/train.csv")
    dfTest = pd.read_csv("Data/test.csv")
    return dfTrain, dfTest

# 接下来考虑通过线性回归 填充年龄信息
# 假设年龄信息和Pclass Sex SibSp parch Fare有一定的线性关系
# 数据处理
def processData(dataSet):
    '''

    :param dataSet: 传入的数据集
    :return: 返回Pclass, Sex, SibSp, Parch, 和 Fare 这5个特征的数据
    '''
    # 需要的特征名
    featureName = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
    # 提取对应的子集
    dataSetSub = dataSet[featureName].copy()
    # 特征编码分类
    dataSetSub["Sex"] = dataSetSub["Sex"].map({"male":0, "female":1})

    # 接下来考虑是否需要补充数据
    for feature in featureName[0:5]:
        if dataSetSub[feature].isna().any().any():  # 存在缺失值
            # 取中位数
            dataSetSub[feature] = dataSetSub[feature].fillna(dataSetSub[feature].median())

    # 接着继续处理数据
    # 将dataSetSub分割为含有Age 和 不含有Age的部分
    dateTrain = dataSetSub[dataSetSub["Age"].notna()].copy()  # age行非空的数据拷贝出来
    dateTest = dataSetSub[dataSetSub["Age"].isna()].copy()     # age行空的数据拷贝出来  未来需要预测的数据

    XTrain = dateTrain[["Pclass", "Sex", "SibSp", "Parch", "Fare"]]
    yTrain = dateTrain["Age"]
    XTest = dateTest[["Pclass", "Sex", "SibSp", "Parch", "Fare"]]

    # 获取索引
    trainIdx = dateTrain.index
    testIdx = dateTest.index

    return XTrain, yTrain, XTest, trainIdx, testIdx

if __name__ == "__main__":
    # dataSetSubTrain = processData(dfTrain)
    # dataSetSubTest = processData(dfTest)
    # 如果需要逐行打印
    # print("\n逐行打印数据：")
    # for index, row in dataSetSubTrain.iterrows():
    #     print(f"行 {index}: {row.to_dict()}")
    # print(dataSetSubTrain.columns)
    '''Index(['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare'], dtype='object')'''
    # 此时Age作为标签
    # print(dataSetSubTrain.isna().sum())

    dfTrain, dfTest = loadData()

    # print(dfTrain)
    # print(dfTest)
    # print(dfTrain.columns)
    '''
    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')

    PassengerId: 乘客ID  
    Survived: 是否幸存 (0 = 否, 1 = 是) **<-- 这是我们的分类目标**  
    Pclass: 船票等级 (1 = 一等舱, 2 = 二等舱, 3 = 三等舱)  
    Name: 姓名  
    Sex: 性别  
    Age: 年龄 **<-- 这是我们的回归目标**  
    SibSp: 同行的兄弟姐妹/配偶数  
    parch: 同行的父母/子女人数  
    Ticket: 船票号  
    Fare: 船票价  
    Cabin: 船舱号  
    Embarked: 登船港口 (C, Q, S)
    '''

    # 检查每列缺值的情况
    '''
    isna().sum()：查看每列缺失值数量（最常用）。
    isna().any().any()：快速确认是否有缺失值。
    isna().mean()：查看缺失值比例。
    可视化：使用 missingno 库直观显示缺失值。
    describe()：间接检查数据完整性。
    '''
    # print(dfTrain.isna().sum())
    # print(dfTest.isna().sum())
    '''
    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
    '''
    # 船舱号缺失太多 简化模型直接删除Cabin特征 后期考虑优化在加入

    # with pd.option_context("display.max_columns", None):
    #     print(dfTrain.describe())
    '''
        原因：省略号是因为 Pandas 默认限制了显示的列数（display.max_columns）。
        解决方法：
            设置 pd.set_option("display.max_columns", None) 显示所有列。
            使用 with pd.option_context("display.max_columns", None) 临时调整。
            调整 display.width 和 display.max_colwidth 优化输出格式。
            保存到 CSV 文件查看完整数据。
    不'''

    X, y, XPre, ComIdx, PreIdx = processData(dfTrain)
    # print(X)
    # 需要将X， y分为训练集和测试集
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    '''数据集的划分过程是随机的，设置 random_state 是为了保证结果的可复现性。
    当你设置一个固定的整数（比如 42），那么无论你运行多少次这段代码，得到的训练集和测试集的划分结果都将是完全相同的。
    这在调试代码、分享结果或与他人协作时非常重要，可以确保每个人得到的数据划分都是一致的。
    如果你不设置这个参数（或者设为 None），那么每次运行代码时，由于随机性的存在，你都会得到一个不同的划分结果。数字 42 本身没有特殊含义，你可以换成任何其他整数。'''

    # 创建带有标准化的Ridge回归管道
    # make_pipeline 会按顺序创建管道：
    # 第一步：StandardScaler() 对数据进行标准化处理 (均值为0，方差为1)
    # 第二步：linear_model.Ridge() 使用岭回归模型进行训练和预测
    ridge_pipe = make_pipeline(
        StandardScaler(),
        linear_model.Ridge()
    )

    # 设置网格参数
    # 定义要搜索的Ridge回归的alpha参数范围
    # np.linspace(14.6, 14.8, 41) 会生成从14.6到14.8之间包含41个均匀分布的数值
    param_test = {'ridge__alpha': np.linspace(13, 13.5, 51)}

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
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1
    )

    # 在训练数据集上面拟合GridSearchCV， 执行网格搜素和交叉验证
    gsearch.fit(XTrain, yTrain)

    # 输出当前的最佳参数
    # gsearch.best_params_ 返回最佳超参数组合
    # gsearch.best_score_ 返回使用最佳参数组合时交叉验证的平均得分
    # 由于scoring是'neg_mean_squared_error'，所以需要取负值来得到实际的MSE
    print(f"最佳参数: {gsearch.best_params_}, 交叉验证MSE = {-gsearch.best_score_:.4f}")
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
    "最佳参数: {'ridge__alpha': np.float64(13.48)}, 交叉验证MSE = 170.2656 误差太大了 不适合 但是走一遍流程"
    best_model = gsearch.best_estimator_
    y_test_pred = best_model.predict(XTest)  # 自动处理标准化
    test_mse = mean_squared_error(yTest, y_test_pred)
    print(f"测试集MSE: {test_mse:.4f}")

    # 将其打包为模型
    joblib.dump(best_model, "tatanic_ridge_model_pipeline.pkl")

    # 接下来通过这个模型预测数据
    # 将在第二个文件中执行



