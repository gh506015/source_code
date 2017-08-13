import numpy as np
import matplotlib.pyplot as plt
import csv

a = np.array([[1,2], [3,4]])
print(a)


a = np.array([[1,2], [3,4]])
print(a + 5)

a = np.array([1,2,4,4,5,5,7,10,13,18,21])
print(a)
print(sum(a)/len(a))
print(np.mean(a))
print(np.median(a))
print(np.min(a))
print(np.max(a))
print(np.var(a))
print(np.std(a))

a = np.array([[1,3,7], [1,1,0]])
b = np.array([[0,0,5], [7,5,0]])
print(a+b)

a = np.array([[1,2,3], [4,10,6], [8,9,20]])
print(a[1][1])
print(a[1,1])

a = np.array([[1,2],[3,4]])
b = np.array([10,20])
a * b

a = np.array([[0], [10], [20], [30]])
b = np.array([0,1,2])
print(a+b)

a = np.array([[51,55], [14,19], [0,4]])
a.flatten()
a[a > 15]

a = [[1,3,7],[1,0,0]]
len(a)
len(a[0])

a = [[1,3,7], [1,1,0]]
b = [[0,0,5], [7,5,0]]
c = [[0,0,0], [0,0,0]]
for i in range(len(a)):
    for j in range(len(a[0])):
        c[i][j] = a[i][j] + b[i][j]
print(c)

a = [[1,3,7], [1,1,0]]
b = [[0,0,5], [7,5,0]]
c = [[0,0,0], [0,0,0]]
for i in range(len(a)):
    for j in range(len(a[0])):
        c[i][j] = a[i][j] + b[i][j]
print(c)


a = [[1,2], [3,4]]
b = [[5,6], [7,8]]
c = [[0,0], [0,0]]
for i in range(len(a)):
    for j in range(len(a[0])):
        for k in range(len(a[0])):
        c[i][j] += a[i][k] * b[k][j]
print(c)

a = np.matrix([[1,2], [3,4]])
b = np.matrix([[5,6], [7,8]])
print(a*b)



a = [[10,20], [30,40]]
b = [[5,6], [7,8]]
c = [[0,0], [0,0]]
for i in range(len(a)):
    for j in range(len(a[0])):
        c[i][j] = a[i][j] - b[i][j]
print(c)

a = np.array([[10,20], [30,40]])
b = np.array([[5,6], [7,8]])
print(a-b)


a = [[1,2], [3,4]]
b = [10,20]
c = [[0,0], [0,0]]
for i in range(len(a)):
    for j in range(len(a[0])):
        c[i][j] = a[i][j] * b[j]
print(c)


plt.figure()   # 객체 선언
plt.plot([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0])
plt.show()

t = np.arange(0, 12, 0.01)
print(t)
plt.figure()
plt.plot(t)
plt.show()

t = np.arange(0, 12, 0.01)
print(t)
plt.figure()
plt.plot(t)
plt.grid()
plt.xlabel('size')
plt.ylabel('cost')
plt.title('size & cost')
plt.show()

x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])
plt.figure()
plt.plot(x,y)
plt.show()



import matplotlib.pyplot as plt
import csv
a = []
b = []
file =open('c:\data\\foundation.csv', 'r')
foundation = csv.reader(file)
foundation
for i in foundation:
    a.append(i[4])
    b.append(i[0])
    print(a)
    print(b)
a = a[1:]
b = b[1:]
plt.plot(b,a)


import numpy as np
chi1 = np.loadtxt('c:\data\\foundation.csv', skiprows=1, unpack=True, delimiter=',')
chi2 = np.loadtxt('c:\data\\shut_down.csv', skiprows=1, unpack=True, delimiter=',')
x1 = chi1[0]
y1 = chi1[4]
x2 = chi2[0]
y2 = chi2[4]
plt.figure(figsize=(6,4))
plt.plot(x1,y1, label='fnd')
plt.plot(x2,y2, linestyle='--', label='sut_dn')
plt.xlabel('YEAR')
plt.ylabel('CNT')
plt.legend()
plt.title('Chicken_foundation and shut_down')
plt.show()


import matplotlib.pyplot as plt
from matplotlib.image import imread
img = imread('c:\data\\하이에나.jpg')
plt.imshow(img)
plt.show()



def percept(w1, w2, bias):
    X = np.array([0,1])
    W = np.array([w1, w2])
    Y = np.sum(X*W)
    if Y + bias > 0:
        return 1
    else:
        return 0
percept(0.5, 0.5, -0.7)


def AND(x1, x2):
    X = np.array([x1, x2])
    W = np.array([0.5, 0.5])
    b = -0.8   # theta값에 - 해서
    Y = np.sum(X*W) + b
    if Y < 0:
        return 0
    else:
        return 1
AND(0,0)


def NAND(x1, x2):
    X = np.array([x1, x2])
    W = np.array([-0.5, -0.5])
    b = 0.8   # theta값에 - 해서
    Y = np.sum(X*W) + b
    if Y < 0:
        return 0
    else:
        return 1
NAND(1,1)


def OR(x1, x2):
    X = np.array([x1, x2])
    W = np.array([1, 1])
    b = -0.5   # theta값에 - 해서
    Y = np.sum(X*W) + b
    if Y < 0:
        return 0
    else:
        return 1
OR(1,1)


def XOR(x1, x2):
    X1 = NAND(x1, x2)
    X2 = OR(x1, x2)
    return AND(X1, X2)

XOR(0,0)

















# AND 게이트
def AND(X1, X2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# NAND 게이트
def NAND(X1, X2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])   # w와 b만 조절해서 NAND 게이트를 만든다!
    b = 0.7                      # w와 b만 조절해서 NAND 게이트를 만든다!
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# OR 게이트
def AND(X1, X2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])    # w와 b만 조절해서 NAND 게이트를 만든다!
    b = -0.2                    # w와 b만 조절해서 NAND 게이트를 만든다!
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 세개 다 같은 구조의 퍼셉트론이고 차이는 가중치 매개변수의 값뿐이다.

import numpy as np
x = np.array([1,2])
y = np.array([3,4])
print(2*x + y)


#######################
import numpy as np

def andPerceptron(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    netInput = x1*w1 + x2*w2
    if netInput <= theta:
        return 0
    elif netInput > theta:
        return 1

def nandPerceptron(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    netInput = x1*w1 + x2*w2
    if netInput <= theta:
        return 0
    elif netInput > theta:
        return 1

def orPerceptron(x1, x2):
    w1, w2, bias = 0.5, 0.5, -0.2
    netInput = x1*w1 + x2*w2 + bias
    if netInput <= 0:
        return 0
    else:
        return 1

def xorPerceptron(x1, x2):
    X1 = nandPerceptron(x1, x2)
    X2 = orPerceptron(x1, x2)
    return andPerceptron(X1, X2)

inputData = np.array([[0,0],[0,1],[1,0],[1,1]])

print("---And Perceptron---")
for xs1 in inputData:
    print(str(xs1) + " ==> " + str(andPerceptron(xs1[0], xs1[1])))

print("---Nand Perceptron---")
for xs2 in inputData:
    print(str(xs2) + " ==> " + str(nandPerceptron(xs2[0], xs2[1])))

print("---Or Perceptron---")
for xs3 in inputData:
    print(str(xs3) + " ==> " + str(orPerceptron(xs3[0], xs3[1])))

print("---Xor Perceptron---")
for xs4 in inputData:
    print(str(xs4) + " ==> " + str(xorPerceptron(xs4[0], xs4[1])))


import numpy as np
# 계단함수 구현
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

x_data = np.array([-1,0,1])
def step_function(x):
    y = x > 0
    return y.astype(np.int)
print(step_function(x_data))


# 계단함수 그래프 그리기
def step_function(x):
    return np.array(x > 0, dtype=np.int)
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()



x = np.array([-1,0,0])       # x0, x1, x2
w = np.array([0.3,0.4,0.1])  # w0, w1, w2
o = np.sum(x*w)
def step_function(x):
    y = x > 0
    return y.astype(np.int)
print(step_function(o))

def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.array([1,2])
print(sigmoid(x))

x = np.arange(-5,5,0.1)
y = sigmoid(x)
y1 = step_function(x)
plt.plot(x,y)
plt.plot(x,y1)
plt.ylim(-0.1, 1.1)
plt.show()


# relu함수
def relu(x):
    return np.maximum(0,x)
relu(0.3)
x = np.arange(-5,5,0.1)
y = relu(x)
plt.plot(x,y)
plt.ylim(-0.1, 5.1)
plt.show()

a = np.array([[1,2], [3,4]])
b = np.array([[5,6], [7,8]])
np.dot(a, b)

a = np.array([[1],[2]])
b = np.array([[5,6],[7,8],[9,10]])
np.dot(b, a)

x = np.array([1,2])
w = np.array([[1,3,5], [2,4,6]])
b = np.array([7,8,9])
o = np.dot(x,w) + b
sigmoid(o)


x = np.matrix([1,2])
w = np.matrix([[1,3,5], [2,4,6]])
b = np.matrix([7,8,9])
o = x*w+b
type(o)
o = list(o)
type(o)



def sigmoid(x):
    return 1 / (1+np.exp(-x))
def identity_function(x):
    return x

x = np.array([4.5,6.2])
w1 = np.array([[0.1,0.3], [0.2,0.4]])
w2 = np.array([[0.5,0.7], [0.6,0.8]])
w3 = np.array([[0.1,0.3], [0.2,0.4]])
b = np.array([0.7,0.8])

o = np.dot(x,w1) + b
o1 = sigmoid(o)
o = np.dot(o1, w2) + b
o1 = sigmoid(o)
o = np.dot(o1, w3) + b
type(o)
y = identity_function(o)
y


# 소프트맥스 함수 만들기
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
softmax(o)

# 항등함수 만들기
def identity_function(x):
    return x


def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)


import numpy as np
a = np.array([[1,1,-1], [4,0,2], [1,0,0]])
b = np.array([[2,-1], [3,-2], [0,1]])
np.dot(a, b)

a = np.matrix([[1,1,-1], [4,0,2], [1,0,0]])
b = np.matrix([[2,-1], [3,-2], [0,1]])
a * b



def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def init_network():
    network = {}
    network['W'] = np.array([[2,4,3], [2,3,5], [2,4,4]])
    network['b'] = np.array([-3,4,9])
    return network

def forward(network, x):
    W = network['W']
    b = network['b']
    a = np.dot(x,W) + b
    return a

network = init_network()
x = np.array([0.2,0.7,0.9])
y = forward(network, x)
z = softmax(y)
print(z)
np.sum(z)


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)


import numpy as np
a = list(range(0,10,3))
print(np.argmax(a))
print(a.index(max(a)))

a = np.array([[.1,.8,.1], [.3,.1,.6], [.2,.5,.3], [.8,.1,.1]])
print(np.argmax(a, axis=1))

type(list(np.argmax(a, axis=1)))

for i in a:
    print(np.argmax(i))

a = np.array([.1,.8,.1,.3,.1,.6,.2,.5,.3,.8,.1,.1]).reshape(4,3)


x = np.array([2,1,3,5,1,4,2,1,1,0])
y = np.array([2,1,3,4,5,4,2,1,1,2])
print(x==y)


# 평균제곱 오차
import numpy as np
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)
t = np.array([0,0,1,0,0,0,0,0,0,0])
y1 = np.array([0.1,0.05,0.6,0,0.05,0.1,0,0,0,0])
y2 = np.array([0.1,0.05,0.1,0,0.05,0.6,0,0,0,0])
mean_squared_error(y1,t)
mean_squared_error(y2,t)

# 교차엔트로피오차
def crossEntropyError(y, t):
    delta = 1e-7 #아주 작은 값 (y가 0인 경우 -inf 값을 예방)
    return -np.sum(t*np.log(y+delta))
t = np.array([0,0,1,0,0,0,0,0,0,0])
y1 = np.array([0.1,0.05,0.6,0,0.05,0.1,0,0,0,0])
y2 = np.array([0.1,0.05,0.1,0,0.05,0.6,0,0,0,0])
crossEntropyError(y1,t)
crossEntropyError(y2,t)

# 여러개의 출력값을 교차엔트로피로 계산하기
t = [0,0,1,0,0,0,0,0,0,0]    # 숫자2
y1 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
y2 = [0.1,0.05,0.2,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
y3 = [0.0,0.05,0.3,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
y4 = [0.0,0.05,0.4,0.0,0.05,0.0,0.0,0.5,0.0,0.0]
y5 = [0.0,0.05,0.5,0.0,0.05,0.0,0.0,0.4,0.0,0.0]
y6 = [0.0,0.05,0.6,0.0,0.05,0.0,0.0,0.3,0.0,0.0]
y7 = [0.0,0.05,0.7,0.0,0.05,0.0,0.0,0.2,0.0,0.0]
y8 = [0.0,0.1,0.8,0.0,0.1,0.0,0.0,0.2,0.0,0.0]
y9 = [0.0,0.05,0.9,0.0,0.05,0.0,0.0,0.0,0.0,0.0]

def crossEntropyError(y, t):
    delta = 1e-7 #아주 작은 값 (y가 0인 경우 -inf 값을 예방)
    return -np.sum(t*np.log(y+delta))

for i in range(1,10):
    print(crossEntropyError(np.array(eval('y'+str(i))), np.array(t)))




import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def  softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return  y

def init_network():
    with open("d://sample_weight.pkl", 'rb') as f:
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
    y = softmax(a3)   # 출력층
    return y

def crossEntropyError(y, t):
    train_size = 60000
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)  # 인덱스로 활용
    t_batch = t_train[batch_mask]
    delta = 1e-7 #아주 작은 값 (y가 0인 경우 -inf 값을 예방)
    return -np.sum(t*np.log(t_batch+delta)) / len(t_batch)

print(1/np.float32(1e-50))


# 수치미분식
def numerical_diff(f,x):
    h = 0.0001
    return (f(x+h)-f(x-h)) / (2*h)

def function_1(x):
    return 3*x**2 + 4*x

# y = 0.01x^2 + 0.1x 미분식 만들어서 수치미분해보기
numerical_diff(function_1, 7)

import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
x = np.random.rand(100,784)
print(x)
a = np.max(x, axis=1)
print(a.shape)
print(x[0])
print(a)

def  softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return  y


import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
def numericalGradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    print(grad)
    for idx in range(x.size):
        tmpVal = x[idx]
        x[idx] = tmpVal + h
        print(idx, x)
        fxh1 = f(x)
        x[idx] = tmpVal - h
        print(idx, x)
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2) / (2*h)
        x[idx] = tmpVal
    return grad

def sampleFunc4(x):
    return x[0]**2 + x[1]**2

print(numericalGradient(sampleFunc4, np.array([3.0,4.0])))
print(numericalGradient(sampleFunc4, np.array([0.0,2.0])))
print(numericalGradient(sampleFunc4, np.array([3.0,0.0])))

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numericalGradient(f,x)
        x -= lr * grad
    return x

def function_2(x):
    return x[0]**2 + x[1]**2
init_x = np.array([-3.0,4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# array([ -6.11110793e-10,   8.14814391e-10]) 거의 0에 가까운 결과
init_x = np.array([-3.0,4.0])
gradient_descent(function_2, init_x=init_x, lr=0.0000000001, step_num=100)   # 학습안됨
init_x = np.array([-3.0,4.0])
gradient_descent(function_2, init_x=init_x, lr=10, step_num=100)             # 발산


#문제88. 위의  2 x 3의 가중치를 랜덤으로 생성하고 간단한 신경망을 구현해서 기울기를 구하는 파이썬 코드를 작성하시오
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    def predict(self, x):
        return np.dot(x, self.W)
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        # print('y:', y)
        loss = cross_entropy_error(y,t)   # y와 t를 교차엔트로피 오류 함수에 넣어 출력
        return loss

net = simpleNet()
print(net.W)



#문제89. 문제88번에서 만든 신경망에 입력값[0.6, 0.9]를 입력하고 target은 [0,0,1]로 해서 정답레이블이 2번이다라고 가정하고서
# 오차가 얼마나 발생하는지 확인하시오
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p)   # 최대값의 인덱스
t = np.array([0, 0, 1])   # 정답레이블
net.loss(x, t)


#문제90. 어제 만든 수치미분함수를 위에서 만든 신경망의 비용함수와 (2x3)의 가중치를 입력해서 기울기를(2x3)를 구하시오
def f(w):
    return net.loss(x,t)

f = lambda w: net.loss(x, t)

def numericalGradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    print(grad)
    for idx in range(x.size):
        tmpVal = x[idx]
        x[idx] = tmpVal + h
        print(idx, x)
        fxh1 = f(x)
        x[idx] = tmpVal - h
        print(idx, x)
        fxh2 = f(x)
        grad[idx] = (fxh1-fxh2) / (2*h)
        x[idx] = tmpVal

dw = numerical_gradient(f, net.W)
print(dw)


#문제91. 아래에서 만든 함수 f를 그냥 lambda식으로 구현해서 f라는 변수에 넣고 아래와 같이 수행하면 기울기가 출력되게 하시오
# def f(w):
#     return net.loss(x,t)
f = lambda w: net.loss(x, t)


#문제92. w1 = np.random.randn(784, 50)를 확인하시오
w1 = np.random.randn(784, 50)
print(w1.shape)

#문제93. b1 = np.zeros(50)를 확인하시오
b1 = np.zeros(50)
print(b1)
print(b1.shape)
type(b1)

#문제97. numerical_gradient() 함수 말고 앞으로 5장에서 배울 오차역전파 (gradient() 함수를 사용해서 정확도를
# 계산하게끔 코드를 수정하시오



# 아래의 스크립에서 out과 x가 같은 객체를 공유하고 있기 때문에 out의 객체를 수정하면 x를 통해 출력되는 객체도 수정된다.
# 같은 객체를 공유하지 않도록 하시오
import numpy as np
x = np.array([[1.0,-.5], [-2.0,3.0]])
print(x)
mask = (x<=0)
print(mask)
out = x.copy()  # 이게 가장 무난하고 좋다.
out = x[:]      # 리스트가 아니라 배열이라서 복사가 되지 않는다.
print(out)
out[mask] = 0
print(out)
print(x)


# 곱셈계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out
    def backward(self,dout):
        dx = dout * self.y   # x와 y를 바꾼다.
        dy = dout * self.x
        return dx, dy

# 문제100. 위에서 만든 곱셈 클래스를 객체화해서 아래의 사과가격의 총 가격을 구하시오
apple = 200
apple_num = 5
tax = 1.2

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)   # 1200


# 문제101. 덧셈계층을 파이썬으로 구현하시오!
class AddLayer:
    def __init__(self):
        pass
    def forward(self,x,y):
        return x + y
    def backward(self, difout):
        dx = difout   # 덧셈은 그냥 그대로
        dy = difout   # 덧셈은 그냥 그대로
        return dx, dy


# 문제102. 사과 2개와 귤 5개를 구입하면 총 가격이 얼마인지 구하시오
apple = 200
apple_num = 2
citrus = 300
citrus_num = 5
tax = 1.5

mul_apple_layer = MulLayer()
mul_citrus_layer = MulLayer()
mul_tax_layer = MulLayer()
plus_layer = AddLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
citrus_price = mul_apple_layer.forward(citrus, citrus_num)
plus_price = plus_layer.forward(apple_price, citrus_price)
price = mul_tax_layer.forward(plus_price, tax)
print(price)


# 문제103. 문제102번 순전파로 출력한 과일 가격 총합의 신경망의 역전파를 파이썬으로 구현하시오
apple = 200
apple_num = 2
citrus = 300
citrus_num = 5
tax = 1.5

mul_apple_layer = MulLayer()
mul_citrus_layer = MulLayer()
mul_tax_layer = MulLayer()
add_apple_citrus_layer = AddLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
citrus_price = mul_citrus_layer.forward(citrus, citrus_num)
plus_price = add_apple_citrus_layer.forward(apple_price, citrus_price)
price = mul_tax_layer.forward(plus_price, tax)

dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dcitrus_price = add_apple_citrus_layer.backward(dall_price)
dcitrus, dcitrus_num = mul_citrus_layer.backward(dcitrus_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple_num, dapple, dcitrus, dcitrus_num, dtax)


# 문제104. 사과, 귤, 배의 총 가격을 구하시오(순전파, 역전파)
# 사과 100원, 4개
# 귤 200원, 3개
# 배 300원, 2개
# 소비세 1.3

apple = 100
citrus = 200
pear = 300
tax = 1.3

mul_apple_layer = MulLayer()
mul_mandarin_layer = MulLayer()
mul_pear_layer = MulLayer()
add_apple_mandarin_layer = AddLayer()
add_all_layer = AddLayer()
mul_tax_layer = MulLayer()

##순전파
apple_price = mul_apple_layer.forward(apple, apple_cnt)
mandarin_price = mul_mandarin_layer.forward(mandarin, mandarin_cnt)
pear_price = mul_pear_layer.forward(pear, pear_cnt)
apple_mandarin_price = add_apple_mandarin_layer.forward(apple_price, mandarin_price)
all_price = add_all_layer.forward(apple_mandarin_price, pear_price)
price = mul_tax_layer.forward(all_price, tax)

## 역전파
d_price = 1
d_all_price, d_tax = mul_tax_layer.backward(d_price) #6번
d_apple_mandarin_price, d_pear_price = add_all_layer.backward(d_all_price) #5번
d_apple_price, d_mandarin_price = add_apple_mandarin_layer.backward(d_apple_mandarin_price) #4번
d_apple, d_apple_cnt = mul_apple_layer.backward(d_apple_price) # 1번
d_mandarin, d_mandarin_cnt = mul_mandarin_layer.backward(d_mandarin_price) #2번
d_pear, d_pear_cnt = mul_pear_layer.backward(d_pear_price) # 3번
print(price)
print(d_apple, d_apple_cnt, d_mandarin, d_mandarin_cnt, d_pear, d_pear_cnt)

# 단순복사
a = [1,2,3,4]
b = a
b[2] = 100
print(a)
print(b)
# 리스트는 변수들이 같은 객체를 공유한다.

a = 10
b = a
print(b)   # 10
b = 'abc'
print(b)   # 'abc'
print(a)   # 10
# 문자열이나 숫자열은 변수들이 같은 객체를 공유하지 않는다.

# 얕은복사
import copy
a = [1,[1,2,3]]
b = copy.copy(a)   # 얕은카피, 모듈사용, 객체공유x
b = a.copy()       # 얕은카피, 모듈사용x, 객체공유x
print(b)   # [1,[1,2,3]]
b[0] = 100
print(b)   # [100,[1,2,3]]
print(a)   # [1,[1,2,3]]
b[1][0] = 200
print(b)   # [100,[200,2,3]]
print(a)   # [1,[200,2,3]]
# 얕은 카피는 밖의 리스트는 공유하지 않지만 안의 리스트는 공유한다.

# 깊은복사
import copy
a = [1,[1,2,3]]
b = copy.deepcopy(a)   # 깊은카피, 모듈사용, 객체공유x
print(b)   # [1,[1,2,3]]
b[0] = 100
print(b)   # [100,[1,2,3]]
print(a)   # [1,[1,2,3]]
b[1][0] = 200
print(b)   # [100,[200,2,3]]
print(a)   # [1,[1,2,3]]
# 깊은 카피는 안과 밖의 리스트 둘 다 공유하지 않는다.

# relu 함수를 만들기 전에 기본적으로 알아야할 문법
import copy
import numpy as np
x = np.array([[1.0,-0.5], [-2.0,3.0]])
print(x)
mask = (x<=0)
print(mask)
out = x.copy()
print(out)
out[mask] = 0
print(out)
print(x)
# 카피가 되었기 때문에 별도의 객체인 out이 만들어졌고 x 객체와는 별도로 out을 변경한 것이다.

# 문제107. Relu함수를 파이썬으로 구현하시오
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# 문제108. 아래의 x변수를 생성하고 x를 Relu객체에 forward 함수에 집어넣으면 무엇이 출력되는지 확인하시오
x = np.array([[1.0,-0.5], [-2.0,3.0]])
relu = Relu()
print(relu.forward(x))

# 문제109. sigmoid함수를 파이썬으로 구현하시오
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        out = 1 / (1+np.exp(-x))
        self.out = out
        return out
    def backward(self, dout):
        dx = dout * (1.0-self.out) * self.out
        return dx

# 문제110. 아래의 식을 파이썬으로 계산하시오
import numpy as np
a = np.array([[1,2,3], [4,5,6]])
b = np.array([[1,2], [3,4], [5,6]])
np.dot(a,b)

###################################아침 시험
class simpleNet:
    def __init__(self):
        self.W = np.array([[2,4,4], [6,3,5]])
    def predict(self, x):
        return np.dot(x, self.W)

net = simpleNet()
print(net.W)
x = np.array([5,6])
p = net.predict(x)
print(p)

# 문제112. 아래의 행렬식을 계산하시오
x = np.array([[1,2], [3,4], [5,6]])
x
y = np.array([7, 8, 9])
y
z = np.array([7,8], ndmin=2)
x.shape
y.shape
z.shape
np.dot(y, x)   # y는 1차원이기 때문에 행렬곱을 유연하게 처리한다. 1차원과 계산을 하면 1차원으로 반환한다.
np.dot(x, z)
#   (3,) * (3,4) * (4,) ==> 이렇게 계산 가능하고, 실제 우리가 손으로 계산할 때는
#   (1,3) * (3,4) * (4,1) ==> 이런 모양이라고 가정하고 계산하면 된다.
#   계산을 완료한 형태가 (1,4)이든, (3,1)이든 무조건 (4,)나 (3,)과 같은 1차원 형태로 반환한다.

# 문제115. 점심시간 문제
import numpy as np

input = np.array([[1,2]])
W1 = ([[1, 3, 5], [2, 4, 6]])
b1 = ([[1, 2, 3]])

L1 = np.dot(input, W1) + b1
print(L1)

W2 = np.array([[1, 4], [2, 5], [3, 6]])
b2 = np.array([[1, 2]])

L2 = np.dot(L1, W2) + b2
print(L2)

W3 = np.array([[1, 3], [2, 4]])
b3 = np.array([[1, 2]])

y = np.dot(L2, W3) + b3
print(y)

# 문제116. 아래의 신경망의 역전파를 구현하시오

# 문제121. 문제120번의 순전파를 구하는 함수를 forward란 이름으로 생성하시오
x = np.array([1,2])
w = np.array([[1,3,5], [2,4,6]])
b = np.array([1,2,3])
def forward(x, w, b):
    return np.dot(x,w) + b
print(forward(x, w, b))

# 문제122. 위의 문제의 역전파를 구하는 함수를 backward라는 이름으로 구현하시오
out = np.array([6,13,20], ndmin=2)
x = np.array([1,2], ndmin=2)
w = np.array([[1,3,5], [2,4,6]])
b = np.array([1,2,3])
def backward(x, w, out):
    dx = np.dot(out, w.T)
    dw = np.dot(x.T, out)
    db = np.sum(out, axis=0)
    return dx, dw, db
print(backward(x, w, out))



# 문제 122. 위에서 만든 forward, backward 함수를 class Affine이라 생성하시오
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, x, dout):
        dx = np.dot(dout, self.W.T)
        dw = np.dot(self.x.T, dout)
        db = dout
        return dx, dw, db

x = np.array([1, 2], ndmin=2)
w = np.array([[1, 3, 5], [2, 4, 6]])
b = np.array([1, 2, 3])
affine = Affine(w, b)
print(affine.forward(x))
print(affine.backward(x, out))


# 문제 123. 아래의 2층 신경망의 순전파를 Affine 클래스를 사용해서 출력하시오
x1 = np.array([1, 2], ndmin=2)
w1 = np.array([[1, 3, 5], [2, 4, 6]])
w2 = np.array([[1,4], [2,5], [3,6]])
b1 = np.array([1, 2, 3])
b2 = np.array([1, 2])
affine1 = Affine(w1, b1)
affine2 = Affine(w2, b2)
out = affine1.forward(x1)
out2 = (affine2.forward(out))

dx2, dw2, db2 = affine2.backward(out, out2)
dx1, dw1, db1 = affine1.backward(x1, dx2)
print(dx1, dw1, db1)

# 문제125. 다시 2층 신경망의 순전파를 구현하는데 은닉층에 활성화 함수로 Relu함수를 추가해서 구현하시오
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, x, dout):
        dx = np.dot(dout, self.W.T)
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        return dx, dw, db

class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


x1 = np.array([1, 2], ndmin=2)
w1 = np.array([[1, 3, 5], [2, 4, 6]])
w2 = np.array([[1,4], [2,5], [3,6]])
b1 = np.array([1, 2, 3])
b2 = np.array([1, 2])


affine1 = Affine(w1, b1)
affine2 = Affine(w2, b2)
relu1 = Relu()
out = affine1.forward(x1)
out1 = relu1.forward(out)
out2 = (affine2.forward(out1))
out2

# 문제126. Relu 함수가 추가된 역전파를 구현하시오

affine1 = Affine(w1, b1)
affine2 = Affine(w2, b2)
relu1 = Relu()
out = affine1.forward(x1)
out1 = relu1.forward(out)
out2 = affine2.forward(out1)
dx2, dw2, db2 = affine2.backward(out, out2)
out1 = relu1.backward(dx2)
dx1, dw1, db1 = affine1.backward(x1, out1)

print(dx1)
print(dw1)
print(db1)


import numpy as np
x = np.array([[1,2], [2,4]])
w = np.array([[1,3,5], [2,4,6]])
b = np.array([1,2,3])
out = np.dot(x, w) + b
dw = np.dot(x.T, out)
dx = np.dot(out, w.T)
print(out)
print(dw)
print(dx)


# 문제127. softmaxwithloss 클래스 파이썬 코드를 구현하시오
class softmaxwithloss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]   # 행의 개수
        dx = (self.y-self.t) / batch_size   # 오차율이 크면 크게 나간다.
        return dx

# 문제128. 위에서 만든 softmaxwithloss클래스를 객체와해서 아래의 x(입력값)과 t(타겟밸류)를 입력해서 순전파 오차율을 확인하시오
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
t = np.array([0,0,1,0,0,0,0,0,0,0])
x1 = np.array([0,.01,.01,.01,.01,.01,.05,.3,.1,.5])
x2 = np.array([0,.01,.9,.01,.01,.01,.01,.03,.01,.01])
np.sum(x)
softmaxwithloss = softmaxwithloss()
print(softmaxwithloss.forward(x1, t))   # 2.40638446467
print(softmaxwithloss.backward())   # -0.09098594   # 오차율이 크다.
print(softmaxwithloss.forward(x2, t))   # 1.54758622721, 오차가 줄어들었다.
print(softmaxwithloss.backward())   # -0.07872391

###OrderDict()확인하기
import collections
print('dict    :')
d1 = {}
d1['a'] = 'A'
d1['b'] = 'B'
d1['c'] = 'C'
d1['d'] = 'D'
d1['e'] = 'E'
d2 = {}
d2['e'] = 'E'
d2['d'] = 'D'
d2['c'] = 'C'
d2['b'] = 'B'
d2['a'] = 'A'
print(d1 == d2)   # True

print('ordereddict    :')
d1 = collections.OrderedDict()
d1['a'] = 'A'
d1['b'] = 'B'
d1['c'] = 'C'
d1['d'] = 'D'
d1['e'] = 'E'
d2 = collections.OrderedDict()
d2['e'] = 'E'
d2['d'] = 'D'
d2['c'] = 'C'
d2['b'] = 'B'
d2['a'] = 'A'
print(d1 == d2)   # True
# orderdDict는 순서가 있는 딕셔너리이다.



###### 수치미분
########################
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.functions import *
from common.gradient import numerical_gradient

# 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1   # 100 x 784   *  784 x 50 + 50 = 100 x 50
        z1 = sigmoid(a1)  # 100 x 50
        a2 = np.dot(z1, W2) + b2  # 100 x 50 * 50 x 10 + 10  = 100 x 10
        y = softmax(a2)
        return y

    def loss(self, x,t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])  # 784 x 50 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])  # 50 개의 bias
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])  # 50 x 10 기울기
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])  # 10개의 bias
        return grads

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
print(x_train.shape[1])  # 784개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch) # 600

for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    print(x_batch.shape) #100 x 784
    t_batch = t_train[batch_mask]
    print(t_batch.shape) # 100 x 10
    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    #grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크
    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        print(x_train.shape) # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

########################



###### 오차역전파
########################
# coding: utf-8
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        print('accuracy', accuracy)
        return accuracy
    # x : 입력 데이터, t : 정답 레이블

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
batch_size = 100  # 미니배치 크기  (100,784)
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch) # 600

for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]


    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]


    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        print(x_train.shape) # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()



#####################################################
#############################################################
##### 오차역전파 이해 코드
# coding: utf-8
import sys,os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = np.array([[1,2,3],[4,5,6]]) #(2,3)
        self.params['b1'] = np.array([1,2,3], ndmin=2) # (2, )
        self.params['W2'] = np.array([[1,2,3],[4,5,6], [7,8,9]]) #(3,3)
        self.params['b2'] = np.array([1,2,3], ndmin=2) #(2, )

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블

    def gradient(self, x, t):
        # forward
        self.loss(x, t)   # 순전파 후 오차율

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)   # 어파인2, 렐루, 어파인1
            print(layer.__class__.__name__, 'dx :\n', dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads


# 문제129. 데이터만 mnist가 아니라 쉽게 하나의 값으로 변경한 코드의 순전파 결과값을 출력하시오
x = np.array([[1,2],[3,4],[5,6]])
t = np.array([[3,4,5], [2,1,4], [2,5,6]])
net = TwoLayerNet()
net.gradient(x, t)


# 문제130. 오차 역전파 2층 신경망 코드의 데이터를 mnist가 아닌 cifar10 데이터로 훈련시켜서 정확도를 확인하시오


# 문제135. 책195쪽에 나오는 모멘텀 공식을 구현하시오
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items()
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


#################################
#################################
#################################
#################################
#################################
# 모멘텀과 sgd 성능비교
# coding: utf-8
import sys,os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.v = None
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    # x : 입력 데이터, t : 정답 레이블

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.01
train_loss_list = []
train_acc_list = []
test_acc_list = []


# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch) # 600

for i in range(iters_num): # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    # SGD @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # for key in ('W1', 'b1', 'W2', 'b2'):
    #     network.params[key] -= learning_rate * grad[key]

    # Momentum @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # v = None
    # if v is None:
    #     v = {}
    #     for key, val in network.params.items():
    #         v[key] = np.zeros_like(val)  # val이랑 같은 shape로 an array of zeros
    # for key in network.params.keys():  # 키만 가지고 옴
    #     v[key] = 0.9 * v[key] - learning_rate * grad[key]   # 러닝레이트
    #     network.params[key] += v[key]

    # AdaGrad @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # v = None
    # if v is None:
    #     v = {}
    #     for key, val in network.params.items():
    #         v[key] = np.zeros_like(val)  # val이랑 같은 shape로 an array of zeros
    # for key in network.params.keys():  # 키만 가지고 옴
    #     v[key] += grad[key] * grad[key]   # 러닝레이트
    #     network.params[key] -= learning_rate * grad[key] / (np.sqrt(v[key]) + 1e-7)

    #Adam @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    v = None
    beta1 = 0.9
    beta2 = 0.999
    iter = 0
    if v is None:
        m, v = {}, {}
        for key, val in network.params.items():
            v[key] = np.zeros_like(val)  # val이랑 같은 shape로 an array of zeros
            m[key] = np.zeros_like(val)  # val이랑 같은 shape로 an array of zeros
    iter += 1
    lr_t = learning_rate * np.sqrt(1.0 - beta2 ** iter) / (1.0 - beta1 ** iter)
    for key in network.params.keys():  # 키만 가지고 옴
        m[key] += (1 - beta1) * (grad[key] - m[key])
        v[key] += (1 - beta2) * (grad[key] ** 2 - v[key])
        network.params[key] -= lr_t * m[key] / (np.sqrt(v[key]) + 1e-7)

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) # cost 가 점점 줄어드는것을 보려고

    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크
    if i % iter_per_epoch == 0: # 600 번마다 정확도 쌓는다.
        print(x_train.shape) # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################


# 문제136. adagrad클래스를 파이썬으로 구현하시오
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items:
                self.h[key] = np.zeros_like(val)
            for key in params.key():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# 문제138. SGD와 AdaGrad의 정확도 차이가 어떻게 나는지 mnist데이터를 16 에폭 경사감소하는 코드를 활용해서 확인하시오
# 위의 코드에 적용시켜놨음

# 문제139. common에 optimizer.py 파일안에 Adam클래스를 가져와서 구현하시오
class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)



# 문제140. 16에폭 mnist 데이터 2층 신경망의 경사감소법을 Adam으로 변경하고 정확도를 확인하시오
# 위에 적용함


# 문제150. 배치정규화 class(common/layers.py)
from common.layers import *


# 문제151. 카페에 배치정규화 하기전 코드 5층 신겅망 코드를 내려받고 돌려보고 정확도를 확인한 수에 배치정규화 코드를 추가하시오
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from collections import OrderedDict
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.4):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] =weight_init_std * np.random.randn(input_size, hidden_size)  # 표준 정규 분포를 따르는 난수 생성
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['W5'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)
        # 계층 생성
        self.layers = OrderedDict()  # forward, backward 시 계층 순서대로 수행하기 위해 순서가 있는 OrderedDict 를 사용.
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['BatchNorm1'] = BatchNormalization(gamma=1.0, beta=0.0)   # 배치정규화
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['BatchNorm2'] = BatchNormalization(gamma=1.0, beta=0.0)   # 배치정규화
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['BatchNorm3'] = BatchNormalization(gamma=1.0, beta=0.0)   # 배치정규화
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['BatchNorm4'] = BatchNormalization(gamma=1.0, beta=0.0)   # 배치정규화
        self.layers['Relu4'] = Relu()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():  # Affine1 -> Relu1 -> Affine2
            x = layer.forward(x)  # 각 계층마다 forward 수행
        return x
    # x : 입력 데이터, t : 정답 레이블

    def loss(self, x, t):  # x : (100, 1024), t : (100, 10)
        y = self.predict(x)  # (100, 10) : 마지막 출력층을 통과한 신경망이 예측한 값
        return self.lastLayer.forward(y, t)  # 마지막 계층인 SoftmaxWithLoss 계층에 대해 forward 수행


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # [[0.1, 0.05, 0.5, 0.05, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1], ....] -> [2, 4, 2, 1, 9, ....]
        if t.ndim != 1: t = np.argmax(t, axis=1)  # t.ndim != 1 이면 one-hot encoding 인 경우이므로, 2차원 배열로 값이 들어온다
        accuracy = np.mean(y == t)
        return accuracy


    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()  # 역전파를 수행하기 위해 기존 layer 순서를 반대로 바꾼다.

        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        grads['W4'], grads['b4'] = self.layers['Affine4'].dW, self.layers['Affine4'].db
        grads['W5'], grads['b5'] = self.layers['Affine5'].dW, self.layers['Affine5'].db


        return grads



(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]  # 60000 개
batch_size = 100  # 미니배치 크기
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch)  # 600


for i in range(iters_num):  # 10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size)  # 100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5'):
        network.params[key] -= learning_rate * grad[key]
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)  # cost 가 점점 줄어드는것을 보려고
    # 1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을때 정확도만 체크

    if i % iter_per_epoch == 0:  # 600 번마다 정확도 쌓는다.
        print(x_train.shape)  # 60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)  # 10000/600 개  16개 # 정확도가 점점 올라감
        test_acc_list.append(test_acc)  # 10000/600 개 16개 # 정확도가 점점 올라감
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 문제151. 바로 위의 코드에 dropout을 추가해서 트레이닝 data의 정확도와 테스트 data의 정확도를 비교해서 오버피팅이 발생이 덜 했는지 확인하시오
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask


import numpy as np
a = np.array([[12,20,30,0], [8,12,2,0], [34,70,37,4], [112,100,25,12]])
b = np.array([[6,7,2,5], [9,21,20,4], [2,5,7,9], [12,14,13,22]])
sum(sum(a*b))


# 문제154. 0~15까지의 (4,4)의 행렬과 0~8까지의 (3,3) 행렬을 만드시오
a = np.array([i for i in range(16)]).reshape(4,4)
a
b = np.array([i for i in range(9)]).reshape(3,3)
b

# 문제155. 위의 코드에서 zero padding 1을 수행하시오
aa = np.pad(a, pad_width=1, mode='constant', constant_values=0)
aa

# 문제156. 다음의 합성곱을 구하시오
data = np.array(range(0,16)).reshape(4,4)
filter = np.array(range(0,9)).reshape(3,3)
result = []
for rn in range(len(data)-2):
    for cn in range(len(data)-2):
        result.append(np.sum(data[rn:rn+3, cn:cn+3] * filter))
np.array(result).reshape(2, 2)


# 문제157. 다음의 합성곱을 구하시오
data = np.array(range(0,36)).reshape(6,6)
filter = np.array(range(0,16)).reshape(4,4)

# 일반화
num = len(data) - len(filter) + 1
result = []
for rn in range(num):
    for cn in range(num):
        result.append(np.sum(data[rn:rn+len(filter), cn:cn+len(filter)] * filter))
np.array(result).reshape(num, num)

# 문제158. 아래와 같이 출력값(OH)와 Straid(S)와 입력값(H)와 필터값(FH)을 입력하면 패딩(P)이 출력되는 함수를 생성하시오
def padding(oh, s, h, fh):
    return (((oh-1)*s)+fh-h) / 2
print(padding(6,1,6,4))

# 문제159. 0~25까지의 원소로 이루어진 (5,5) 행렬을 만들고 0~4까지의 원소로 이루어진 (2,2) 행렬을 이용해서 합성곱을 하시오
# (단, 스트라이드는 1, 출력행렬은 (5,5)가 되도록 패딩을 적용하세요)
import numpy as np
data = np.array(range(0,81)).reshape(9,9)
filter = np.array(range(0,16)).reshape(4,4)

def find_pad(data, filter, s, oh):
    h = len(data)
    fh = len(filter)
    return (((oh-1)*s)+fh-h) / 2

def padding(data, x):
    if x%1 == 0:
        x = int(x)
        return np.pad(data, pad_width=x, mode='constant', constant_values=0)
    else:
        x1 = int(x+0.5)
        x2 = int(x-0.5)
        return np.pad(data, pad_width=((x1,x2), (x1,x2)), mode='constant', constant_values=0)

def output(data, filter):
    num = len(data) - len(filter) + 1
    result = []
    for rn in range(num):
        for cn in range(num):
            result.append(np.sum(data[rn:rn+len(filter), cn:cn+len(filter)] * filter))
    return np.array(result).reshape(num, num)

f_p = find_pad(data, filter, 1, 9)   # Straid(s) / 출력값(oh)
data = padding(data, f_p)
print('q3\n', output(data, filter))
print('q4\n', output(data, filter) * 3)

# 문제160. 3차원 합성곱
x1 = np.array([[1,2,0,0], [0,1,-2,0], [0,0,1,2], [2,0,0,1]])
x2 = np.array([[1,0,0,0], [0,0,-2,-1], [3,0,1,0], [2,0,0,1]])
f1 = np.array([[-1,0,3], [2,0,-1], [0,2,1]])
f2 = np.array([[0,0,0], [2,0,-1], [0,-2,1]])
o1 = output(x1, f1)
o2 = output(x2, f2)
q1 = o1 + o2
print('q1\n',q1)
q2 = q1 + np.array([[0,0], [2,2]])
print('q2\n',q2)


x = np.array([[21,8,8,12], [12,19,9,7], [8,10,4,3], [18,12,9,10]])
def pooling(data):
    result = []
    for i in range(0, len(data), 2):
        for j in range(0, len(data[0]), 2):
            tmp = []
            tmp.append(data[i:i+2, j:j+2])
            a = np.mean(tmp)
            result.append(a)
    result = np.array(result).reshape(2,2)
    print(result)
pooling(x)

#############################
###################################### 용현이형 코드
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.

    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        # out = np.max(col, axis=1)   # 최대풀링
        out = np.mean(col, axis=1)  # 평규풀링
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out


data = np.array(
       [[
         [[1, 2, 3, 0],
          [0, 1, 2, 4],
          [1, 0, 4, 2],
          [3, 2, 0, 1]],
         [[3, 0, 6, 5],
          [4, 2, 4, 3],
          [3, 0, 1, 0],
          [2, 3, 3, 1]],
         [[4, 2, 1, 2],
          [0, 1, 0, 4],
          [3, 0, 6, 2],
          [4, 2, 4, 5]]
       ]])
max_pool = Pooling(2, 2)
forward_max = max_pool.forward(data)
print(data.shape)
print(forward_max.shape)
print(data)
print(forward_max)



#####################
#####################
#####################
#####################
#####################
#####################
import sys, os
sys.path.append(os.pardir)
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h  # y=0일 때 y_max=2,  y=1일 때 y_max=3,  y=2일 때 y_max=4
        for x in range(filter_w):
            x_max = x + stride * out_w  # x=0일 때 x_max=2,  x=1일 때 x_max=3,  x=2일 때 x_max=4
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # 디버깅 variables화면/ col 우측 클릭 => view as Array 클릭
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    # col.shape = (N,C,filter_h,filter_w,out_h,out_w)  =>  col.shape(N,out_h,out_w,C,filter_h,filter_w) 로 transpose 이후
    # (N*out_h*out_w, C*filter_h*filter_w) 의 2차원 행렬로 reshape
    # print(col.shape)
    return col

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)
        col = im2col(x, FH, FW, self.stride, self.pad)  # col.shape = (N*out_h*out_w, C*FH*FH)
        # print(col.shape)
        col_W = self.W.reshape(FN, -1).T  # (FN,C,FH,FW)  reshape=> (FN, C*FH*FW)  transpose=> (C*FH*FW, FN)
        out = np.dot(col, col_W) + self.b  # 결과 차원 (N*out_h*out_w,FN)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # out.reshape의 shape = (N,out_h,out_w,FN)
        # transpose이후 shape = (N,FN,out_h,out_w)
        return out

x1 = np.arange(192).reshape(1, 3, 8, 8)
W1 = np.arange(135).reshape(5, 3, 3, 3)
b1 = 1
conv = Convolution(W1, b1)
f = conv.forward(x1)
print('f = ', f, 'f.shape = ', f.shape)   # N, FN, out_h, out_w


class Pooling:
    def __init__(self,pool_h,pool_w,stride=2,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # 입력 데이터 4차원을 2차원으로 변경
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        # 2차원으로 변경된 행에서 최대값을 출력
        out = np.max(col,axis=1)
        # 다시 2차원을 4차원으로 변경한다.
        out = out.reshape(N,out_h, out_w, C).transpose(0,3,1,2)
        return out

data = np.array([[
                [[1,2,0,1],
                 [3,0,2,4],
                 [1,0,3,2],
                 [4,2,0,1]],
                [[3,0,4,2],
                 [6,5,4,3],
                 [3,0,2,3],
                 [1,0,3,1]],
                [[4,2,0,1],
                 [1,2,0,4],
                 [3,0,4,2],
                 [6,2,4,5]]
               ]])

max_pool = Pooling(2, 2)
forward_max = max_pool.forward(data)
print(data.shape) # (1, 3, 4, 4)
print(forward_max.shape) # (1, 3, 2, 2)

class SimpleNet:
    def __init__(self, input_dim=(1,28,28),
                 conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, stride:1}):
        hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size-filter_size+2*filter_pad) / filter_stride+1



####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)
        col = im2col(x, FH, FW, self.stride, self.pad)  # col.shape = (N*out_h*out_w, C*FH*FH)
        # print(col.shape)
        col_W = self.W.reshape(FN, -1).T  # (FN,C,FH,FW)  reshape=> (FN, C*FH*FW)  transpose=> (C*FH*FW, FN)
        out = np.dot(col, col_W) + self.b  # 결과 차원 (N*out_h*out_w,FN)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # out.reshape의 shape = (N,out_h,out_w,FN)
        # transpose이후 shape = (N,FN,out_h,out_w)
        return out


# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer


class SimpleConvNet:
    """단순한 합성곱 신경망
    conv - relu - pool - affine - relu - affine - softmax
    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """손실 함수를 구한다.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t)
        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])
        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Conv2', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]


