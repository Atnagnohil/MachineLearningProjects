# _*_ coding: utf-8 _*_
'''
时间:      2025/6/16 17:20
@author:  andinm
'''
# 获取数据
import pandas as pd
from sklearn import linear_model

dfTrain = pd.read_csv("Data/train.csv")
dfTest = pd.read_csv("Data/test.csv")

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

XTrain, yTrain, XTest, trainIdx, testIdx = processData(dfTrain)
# print(XTrain)


