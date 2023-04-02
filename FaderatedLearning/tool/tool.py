import numpy as np
from scipy.optimize import minimize

#热编码比如把[1,2,3]变为[[1,0,0,],[0,1,0],[0,0,1]]
def one_hot_encoder(raw_y):
    result = []
    for i in raw_y:
        y_temp = np.zeros(7)
        y_temp[i - 1] = 1
        result.append(y_temp)
    return np.array(result)

# 序列化函数化为(n,)类型的数组
def serialize(a, b):
    return np.append(a.flatten(), b.flatten())

#反序列化函数
def deserialize(theta_serialization):
    theta11 = theta_serialization[:25 * 36].reshape(25, 36)
    theta22 = theta_serialization[25 * 36:].reshape(7, 26)
    return theta11, theta22

# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 前向传播
def feed_forwaed(theta_serialization, X):
    theta1, theta2 = deserialize(theta_serialization)
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=1, axis=1)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

# 损失函数
def cost(theta_serialization, X, Y):
    a1, z2, a2, z3, h = feed_forwaed(theta_serialization, X)
    J = -np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h)) / len(X)
    return J

#正则化损失函数
def reg_cost(theta_serialization, X, Y, lamda):
    t1, t2 = deserialize(theta_serialization)
    sum1 = np.sum(np.power(t1[:, 1:], 2))
    sum2 = np.sum(np.power(t2[:, 1:], 2))
    reg = (sum1 + sum2) * lamda / (2 * len(X))
    current_cost=reg + cost(theta_serialization, X, Y)
    return current_cost

#sigmoid函数求导
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

#反向传播
def gradient(theta_serialization, X, Y):
    t1, t2 = deserialize(theta_serialization)
    a1, z2, a2, z3, h = feed_forwaed(theta_serialization, X)
    d3 = h - Y
    d2 = d3 @ t2[:, 1:] * sigmoid_gradient(z2)
    D2 = (d3.T @ a2) / len(X)
    D1 = (d2.T @ a1) / len(X)
    return serialize(D1, D2)

#正则化反向传播
def reg_gradient(theta_serialization, X, Y, lamda):
    D = gradient(theta_serialization, X, Y)
    D1, D2 = deserialize(D)
    t1, t2 = deserialize(theta_serialization)
    D1[:, 1:] = D1[:, 1:] + t1[:, 1:] * lamda / len(X)
    D2[:, 1:] = D2[:, 1:] + t2[:, 1:] * lamda / len(X)
    return serialize(D1, D2)

#优化训练函数
def nn_training(init_theta,X,Y,lamda):
    res = minimize(fun=reg_cost,
                   x0=init_theta,
                   args=(X, Y,lamda),
                   method='CG',
                   jac=reg_gradient,
                   options = {'maxiter':1}
                   )
    return res

