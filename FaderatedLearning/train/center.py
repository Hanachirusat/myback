import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from FaderatedLearning.tool import tool as tool
import ray
import work1
import work2
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


# 把多分类根据某个类别变为二分类 index代表类别
def MultipleToBinary(Y, index):
    y = Y
    for i in range(len(y)):
        if y[i] != index:
            y[i] = 0
    for i in range(len(y)):
        if y[i] == index:
            y[i] = 1
    return y


# 对最终得到的theta参数进行测试，此时的theta为(n,)类型
def test(theta):
    # 读取数据
    data = pd.read_csv('../Data/data.csv')
    data.head()
    X = data.iloc[:, :35]
    y = data.iloc[:, 35:36]
    X = np.array(X)  # X为原始数据，二维数组表示
    y = np.array(y)  # Y为原始数据，列向量表示
    raw_y = y.reshape(444, )  # 行向量  ps可以认为（444，1）是列向量 （444，0）是行向量
    X = np.insert(X, 0, values=1, axis=1)
    Y = tool.one_hot_encoder(y)  # Y为热编码之后的数据，是一个二维数组（神经网络特供）

    # 准确率计算
    _, _, _, _, h = tool.feed_forwaed(theta, X)
    Y_pred = np.argmax(h, axis=1) + 1  # h里面每一行中最大的索引拿出来 因为索引是从0开始，所以最后要+1
    acc = np.mean(Y_pred == raw_y)
    print(acc)

    # PRF计算
    # 首先是总体情况下的prf
    # P = （真阳性） / （所有预测为真的样本数）= TP / (TP + FP)
    # R = （真阳性） / （实际情况中为真的样本数）= TP / (TP + FN)
    # F = 2 * P * R / (P + R)
    print('    P        R         F1       ')
    targets = raw_y  # 原始y值非热编码
    outputs = Y_pred  # 预测y值
    precision = precision_score(targets, outputs, average='macro')
    recall = recall_score(targets, outputs, average='macro')
    f1 = f1_score(targets, outputs, average='macro')
    print('{0}   {1:.2f}     {2:.2f}     {3:.2f}     '.format(1, precision, recall, f1))

    # 接下来是对每个类别分别进行PRF值计算，对每个类别进行计算时要把多分类转化为二分类，是该类别为1，不是该类别统统为0
    for i in range(7):
        target = raw_y.copy()
        output = Y_pred.copy()
        targets = MultipleToBinary(target, i + 1)
        outputs = MultipleToBinary(output, i + 1)
        precision = precision_score(targets, outputs)
        recall = recall_score(targets, outputs)
        f1 = f1_score(targets, outputs)
        print('{0}   {1:.2f}     {2:.2f}     {3:.2f}     '.format(i + 2, precision, recall, f1))
        time.sleep(0.1)
    print('注：第1行为计算每个标签的度量，并找到它们的未加权平均值。第2-8行为分别为1-7每个标签的度量')
    # 由此可见风寒感冒和风热感冒的无法准确分辨开啦，
    # 并且对于脚气（疾病代码1）因为只有3个症状，因此对每个症状的特征加大一点
    # 其他症状有就是1，没有就是0，脚气有是2没有是0，这样处理后对于脚气的预测就变得更准确了


# 绘制散点图
def costpicture(costs):
    fig, ax = plt.subplots()  # ax是绘图实例
    ax.plot(np.arange(len(costs)), costs)  # range(iters列表0-2000，)第一个参数是横轴数据，第二个函数是纵轴数据
    # 画图的时候 根据数据画图，实际上画的是点，但应用了plot，会自动把点变成曲线
    ax.set(xlabel='iters',
           ylabel='costs',
           title='cost of iters')
    plt.show()


# -------------------初始化阶段--------------------
costs = []  # costs用于保存过程cost
ray.init()  # ray框架启动
# 确定模型
HideLayer = 25
OutPut = 7
LengthInput = 35
FinalTheta = np.zeros(1082)
theta = np.random.uniform(-0.5, 0.5, 1082)

# -------------------学习阶段--------------------
# 开始联邦学习模拟过程，并记录过程所需时间
start_time = time.time()
for i in range(400):
    # 下面work1和work2并行执行
    work1_id = work1.work1.remote(theta)
    work2_id = work2.work2.remote(theta)

    # 获取work1和work2执行后的返回结果，由于数据量不大，因此采用同步方式最终准确率更高。
    theta1 = ray.get(work1_id)[0]
    cost1 = ray.get(work1_id)[1]
    theta2 = ray.get(work2_id)[0]
    cost2 = ray.get(work2_id)[1]

    # 对获取到的结果进行加权平均，并且记录cost
    theta = (theta1 + theta2) / 2
    costs.append((cost1 + cost2) / 2)
print('执行时间为:{0}'.format(time.time() - start_time))

# 下面是非并行计算下所需的时间，过程和并行计算下一致
'''
for i in range(600):
     theta1,cost1 = work3.work3(theta)
     theta2,cost2 = work4.work4(theta)
     theta=theta1+theta2
     theta=theta/2
     costs.append((cost1+cost2)/2)
print('执行时间为:{0}'.format(time.time()-start_time))
'''

# -------------------评测阶段--------------------
# PRF评测
test(theta)
# 绘制学习图像，观察是否收敛
costpicture(costs)
# 训练完成后的模型保存
scipy.io.savemat('../Data/FinalTheta.mat', {'theta': theta})
