import numpy as np
import scipy.io as sio
from FaderatedLearning.tool import tool as tool


#symtom是症状特征数组格式为(1,36)的二维数组
def Classifier(symptom):
    #读取保存的模型，即相应的参数
    theta = sio.loadmat('E:\pythonProject\myback\FaderatedLearning\Data\FinalTheta.mat')['theta']
    theta = theta.reshape(1082, )

    #根据模型，和获得的输入进行预测输出，其中rate是百分数，disease是疾病名称
    _, _, _, _, h = tool.feed_forwaed(theta, symptom)
    disease_code = np.argmax(h, axis=1) + 1  # h里面每一行中最大的索引拿出来 因为索引是从0开始，所以最后要+1
    rate=h.take(disease_code-1,1)  #取出h中Y_pred列的数，第二个参数0表示行取，1表示列取
    disease=['干眼症','脚气','高血压','急性肠胃炎','飞蚊症','风寒感冒','风热感冒']

    r=rate[0][0]  #rate为[[]]类型的数据，要转化为普通float数据
    d=disease_code[0]  #disease_code为[]类型数据，要转化为普通int数据
    #格式化返回结果，如98.34%，风热感冒
    return '{:.2%}'.format(r),disease[d-1]
