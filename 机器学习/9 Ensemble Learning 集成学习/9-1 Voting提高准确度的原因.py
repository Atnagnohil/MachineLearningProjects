# _*_ coding: utf-8 _*_
'''
时间:      2025/7/15 16:37
@author:  andinm
'''


#### 模拟不同分类器的集成效果，每个分类器的准确率只有51%(只比随机猜好一点)，但是集成后的效果有显著提升
##### 如果用1000个分类器，最终准确率可达75%。如果用10000个分类器，最终准确率高达97%

import numpy as np

n=10000#分类器数量
p = np.array([0.51, 0.49])
# print(p.ravel())
result = []
for i in range(1000):
    num = np.random.choice([0, 1],size=n, p = p.ravel())    # 每次选取个数维n的0/1  判断0和1的数目谁多
    if len(num[num==0])>len(num[num==1]):
        result.append(0)
    else:
        result.append(1)

result = np.array(result)
print(len(result[result==0])/float(len(result)))






