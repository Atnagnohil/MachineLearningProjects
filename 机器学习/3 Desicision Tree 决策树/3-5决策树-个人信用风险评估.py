# _*_ coding: utf-8 _*_
'''
时间:      2025/6/16 13:57
@author:  andinm
'''


# 数据探索与预处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit = pd.read_csv("Data/german_credit.csv")
# print(credit.head(2))

# print(credit.shape)


'''该数据集包含1000个样本和21个变量。其中default表示信用好坏(0表示好，1表示坏)，其余为特征变量

使用value_counts()函数对支票余额变量account_check_status和储蓄账户余额变量savings_balance进行查看。'''
# print(credit.job.value_counts())
'''job
skilled employee / official                                      630
unskilled - resident                                             200
management/ self-employed/ highly qualified employee/ officer    148
unemployed/ unskilled - non-resident                              22
Name: count, dtype: int64
'''
# print('\n'+'-'*100+'\n')
# print(credit.savings.value_counts())


'''上述两个变量的单位都是德国马克（Deutsche Mark, DM）。 直观来看，支票余额和储蓄账户余额越大，贷款违约的可能性越小。

该贷款数据集还有一些数值型变量，例如贷款期限（duration_in_month）和贷款申请额度（credit_amount）。'''

                                            # 将数据中的字符串编码成数值
# 将字符变量转换成数值
# print(credit.info()) # 查看哪些是数字变量

# 查看可取哪些值
# print(credit.account_check_status.value_counts())

# 取字符串的数组
cols = ['account_check_status', 'credit_history', 'purpose', 'savings', 'present_emp_since', 'personal_status_sex',
        'other_debtors', 'property', 'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker']
# 映射字典
col_dicts = {}
col_dicts = {
    'account_check_status': {
        '0 <= ... < 200 DM': 2,
        '< 0 DM': 1,
        '>= 200 DM / salary assignments for at least 1 year': 3,
        'no checking account': 0
    },

    'credit_history': {
        'existing credits paid back duly till now': 0,
        'critical account/ other credits existing (not at this bank)': 1,
        'delay in paying off in the past': 2,
        'all credits at this bank paid back duly': 3,
        'no credits taken/ all credits paid back duly': 4
    },

    'purpose': {
        'domestic appliances': 0,
        'car (new)': 1,
        'radio/television': 2,
        'car (used)': 3,
        'business': 4,
        '(vacation - does not exist?)': 5,
        'education': 6,
        'furniture/equipment': 7,
        'repairs': 8,
        'retraining': 9
    },

    'savings': {'... < 100 DM': 0,
                'unknown/ no savings account': 1,
                '100 <= ... < 500 DM': 2,
                '500 <= ... < 1000 DM': 3,
                '.. >= 1000 DM': 4},

    'present_emp_since': {
        '1 <= ... < 4 years': 0,
        '.. >= 7 years': 1,
        '4 <= ... < 7 years': 2,
        '... < 1 year': 3,
        'unemployed': 4
    },

    'other_installment_plans': {
        'none': 0,
        'bank': 1,
        'stores': 2
    },

    'foreign_worker': {'no': 1, 'yes': 0},

    'housing': {'for free': 1, 'own': 0, 'rent': 2},

    'job': {'skilled employee / official': 0,
            'unskilled - resident': 1,
            'management/ self-employed/ highly qualified employee/ officer': 2,
            'unemployed/ unskilled - non-resident': 3},

    'other_debtors': {'co-applicant': 2, 'guarantor': 1, 'none': 0},

    'personal_status_sex': {'male : single': 0,
                            'female : divorced/separated/married': 1,
                            'male : married/widowed': 2,
                            'male : divorced/separated': 3},

    'property': {'if not A121/A122 : car or other, not in attribute 6': 0,
                 'real estate': 1,
                 'if not A121 : building society savings agreement/ life insurance': 2,
                 'unknown / no property': 3},

    'telephone': {'none': 0, 'yes, registered under the customers name': 1}}

for col in cols:
    credit[col] = credit[col].map(lambda x: x.strip())
    credit[col] = credit[col].map(col_dicts[col])

# print(credit.head(5))
# print(credit.info())


        # 划分数据集
from sklearn import model_selection

y = credit['default']
X  = credit.loc[:,'account_check_status':'foreign_worker']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
print (y_train.value_counts()/len(y_train))
print (y_test.value_counts()/len(y_test))


            # 模型训练
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
credit_model = DecisionTreeClassifier(min_samples_leaf = 6,random_state=1)
credit_model.fit(X_train, y_train)

            # 模型性能评估
credit_pred = credit_model.predict(X_test)


'''现在，我们得到了决策树模型在测试数据上的预测结果，通过将预测结果和真实结果进行对比可以评估模型性能。
可以使用`sklearn.metrics`包中的`classification_report()`和`confusion_matrix()`函数，展示模型分类结果：'''
from sklearn import metrics
print (metrics.classification_report(y_test, credit_pred))
'''
              precision    recall  f1-score   support

           0       0.77      0.83      0.80       214
           1       0.48      0.40      0.43        86

    accuracy                           0.70       300
   macro avg       0.63      0.61      0.62       300
weighted avg       0.69      0.70      0.69       300

'''
print (metrics.confusion_matrix(y_test, credit_pred))

'''在300个贷款申请测试数据中，模型的预测正确率（Accuracy）为70%。 214个未违约贷款中，模型正确预测了83%(recall值)。 
86个违约贷款中，模型正确预测出了40%(recall)。 下面，我们看看是否能够进一步改善模型的性能。
'''




                # 6 模型性能提升

'''
在实际应用中，模型的预测正确率不高，很难将其应用到实时的信贷评审过程。
在本案例中，如果一个模型将所有的贷款都预测为“未违约”，此时模型的正确率将为72%，而该模型是一个完全无用的模型。
上节中我们建立的模型，正确率为70%，但是对于违约贷款的识别性能很差。
我们可以通过创建一个代价矩阵定义模型犯不同错误时的代价。
假设我们认为一个贷款违约者给银行带来的损失是银行错过一个不违约的贷款带来损失的4倍，则未违约和违约的代价权重可以定义为：'''
class_weights = {0:1, 1:4}
credit_model_cost = DecisionTreeClassifier(max_depth=6,class_weight = class_weights)
credit_model_cost.fit(X_train, y_train)
credit_pred_cost = credit_model_cost.predict(X_test)

print (metrics.classification_report(y_test, credit_pred_cost))
'''
              precision    recall  f1-score   support

           0       0.91      0.54      0.67       214
           1       0.43      0.86      0.57        86

    accuracy                           0.63       300
   macro avg       0.67      0.70      0.62       300
weighted avg       0.77      0.63      0.64       300
'''
print (metrics.confusion_matrix(y_test, credit_pred_cost))
'''[[115  99]
 [ 12  74]]
'''


