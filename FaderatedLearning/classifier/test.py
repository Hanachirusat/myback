import numpy as np
import pandas as pd
from FaderatedLearning.tool import tool as tool
import classifier as cl

#读取数据
data = pd.read_csv('../Data/data.csv')
data.head()
X = data.iloc[:, :35]
y = data.iloc[:, 35:36]
X = np.array(X)  # X为原始数据，二维数组表示
y = np.array(y)  # Y为原始数据，列向量表示
raw_y = y.reshape(444, )  # 行向量  ps可以认为（444，1）是列向量 （444，0）是行向量
X = np.insert(X, 0, values=1, axis=1)
Y = tool.one_hot_encoder(y)  # Y为热编码之后的数据，是一个二维数组（神经网络特供）

#测试结果e
for i in range(440):
    #random=np.random.randint(0,444)
    #xcv=X.take(random,0)
    xcv=X.take(i,0)
    xcv=xcv.reshape(1,36)
    rate,disease=cl.Classifier(xcv)
    #print('第{0}次测试  选取数据为第{1}行  您有{2}的几率患{3}'.format(i,random,rate,disease))
    print('第{0}次测试    您有{1}的几率患{2}'.format(i, rate, disease))