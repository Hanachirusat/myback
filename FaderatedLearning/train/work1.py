import time

import numpy as np
import pandas as pd
import ray
from FaderatedLearning.tool import tool as tool


@ray.remote
def work1(theta):
    # 数据读取部分
    data = pd.read_csv('../Data/data1_new.csv')
    data.head()
    X = data.iloc[:, :35]
    y=data.iloc[:,35:36]
    X = np.array(X)  #X为原始数据，二维数组表示
    y = np.array(y)  #Y为原始数据，列向量表示
    X = np.insert(X,0,values=1,axis=1)
    Y = tool.one_hot_encoder(y)   #Y为热编码之后的数据，是一个二维数组（神经网络特供）

    # 训练开始
    lamda=1
    cost = tool.reg_cost(theta, X, Y, lamda)
    res = tool.nn_training(theta, X, Y, lamda)
    time.sleep(0.02)   #使得分布式计算缩短的时间更明显
    return res.x,cost



