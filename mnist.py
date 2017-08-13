import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return x_train,t_train

def softmax(a):
    a = a.T   # 각 행의 최대값을 가져온다.
    c = np.max(a, axis=0)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a, axis=0)
    y = exp_a / sum_exp_a
    return  y.T

def init_network():
    with open("C:\data\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

def crossEntropyerror(y,t):
    delta = 1e-7
    print('y', y)   # y에 확률 들어오고
    print(len(y))   # 0~~9 10줄
    print('t', t)   # t에 레이블 들어온다.(원-핫 코딩)
    print(len(t))   # 0~~9 10줄
    return -np.sum(t*np.log(y+delta)) / len(y)   # 각 행에 대한 오차율 평균

train = get_data()
# print('train1111', train)
# print(len(train))

x_train = train[0]
t_train = train[1]

print(x_train.shape)
print(t_train.shape)

train_size = 60000
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

y = predict(init_network(), x_batch)
p = np.argmax(y, axis=1)
print(p)

print('CEE:', crossEntropyerror(y, t_batch))   # 오차율