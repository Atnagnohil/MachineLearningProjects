# _*_ coding: utf-8 _*_
'''
时间:      2025/6/3 16:28
@author:  andinm
'''

# 导入库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from tool.Read_Minist_Tool import *


# 加载数据
print("-----------正在加载数据-----------------")
train_images = load_train_images()
train_labels = load_train_labels()

test_images = load_test_images()
test_labels = load_test_labels()
print("-----------数据加载完成-----------------")

print(train_images.shape)
# 查看一部分数据
def plotPartData(num, Data, labels):
    for i in range(num):
        plt.subplot(3,int(num/3)+1, i+1)
        print(labels[i])
        plt.imshow(Data[i], cmap='gray')
    plt.show()
# plotPartData(10, train_images, train_labels)

# 数据处理（维度转变，归一化处理）
X = train_images.reshape(train_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)
# print(train_images.shape)
# print(test_images.shape)
X = X / 255
X_test = X_test / 255


# 模型训练
model = linear_model.LogisticRegression(C=50, max_iter=2000)
model.fit(X, train_labels)
print(model.intercept_)
print(model.coef_)

# 测试模型精准度
y_train_hat = model.predict(X_test)
print(f"测试集精度{accuracy_score(y_train_hat, test_labels)}")


import joblib  # 保存模型
joblib.dump(model, "digit_recognition_model.pkl")
















