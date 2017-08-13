import tensorflow as tf
import numpy as np
a = tf.placeholder("float") # 공간을 만든다.
b = tf.placeholder("float")
y = tf.multiply(a,b)  #곱을 한다.
with tf.Session() as sess:
    sess.run(y, feed_dict={a:10,b:32})
##########################################
##########################################
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder("float",[None,784])
#todo 입력 노드가 784개이고 앞에 None 은 여기에 어떤 크기가 오던
#todo 가능하다는 뜻이다. 학습과정에 사용될 이미지의 총개수가 될것이다.
W = tf.Variable(tf.zeros([784,10]))
#todo 출력층의 갯수가 10개여서 10으로 줌
# 텐써 이용하지 않았을때 ?
# self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
b = tf.Variable(tf.zeros([10]))
#todo 출력층의 노드의 갯수에 맞춰서 편향도 10개이다.
y = tf.nn.softmax(tf.matmul(x,W) + b)
#todo affine 한 결과를 softmax 함수에 바로 입력해서 한번에 수행되고
#todo  예상값을 리턴하고 있다.
y_ = tf.placeholder("float",[None,10])
#todo 교차엔트로피를 구현하기 위해서 실제 레이블을 담고있는
#todo 새로운 플레이스 홀더를 생성한다.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#todo 비용함수를 구현하는데 여기서 사용되는 reduce_sum 은
#todo 차원축소후 sum 하는 함수
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#todo 학습속도 0.01 과 SGD 경사하강법으로 비용함수의 오차가
#todo 최소화 되겠금 역전파 시킴
sess = tf.Session()
#todo 텐써 플로우 그래프 연산을 시작하겠금 세션객체를 생성
sess.run(tf.global_variables_initializer())
#todo 모든 변수를 초기화 한다.
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #todo 훈련 데이터셋에서 무작위로 100개를 추출
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #todo 100개의 데이터를 SGD 의 경사감소법으로 훈련시킨다.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #todo y 라벨(예상)중 가장 큰 인덱스를 리턴하고
    #todo y_라벨(실제값) 중 가장 큰 인덱스를 리턴해서 같은지 비교
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    # todo [True,False,True,True] 를 [1,0,0,1] 로 변경 될것이고
    # todo 평균 0.75 가 출력된다.ㅏ
    print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # todo 정확도가 출력이 된다.
######################################
######################################

# coding: utf-8
# 문제2번. 단층 신경망을 텐써플로우로 구현하시오!(p112)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # Mnist 데이터를 사용하여 원핫인코딩을 한다.

import tensorflow as tf

x = tf.placeholder("float", [None, 784])
# todo 입력노드가 784개이고 앞의 None은 여기에 어떤 크기가 오든 가능하다는 뜻이다.
# todo 학습과정에 사용될 이미지의 총 갯수가 될 것이다.
# todo placeholder는 비어있는 메모리 공간을 잡아두는 것이다.

W = tf.Variable(tf.zeros([784,10]))
# todo 출력층의 갯수가 10개여서 10으로 지정한다.
# todo 텐써 이용하지 않았을때?
# todo self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)

b = tf.Variable(tf.zeros([10]))
# todo 출력층의 노드의 갯수에 맞춰서 편향도 10개이다.

y = tf.nn.softmax(tf.matmul(x,W) + b)
# todo affine한 결과를 softmax 함수에 바로 입력해서 한번에 수행되고
# todo 예상값을 리턴하고 있다.

y_ = tf.placeholder("float", [None,10])
# todo 교차엔트로피를 구현하기 위해서 실제 레이블을 담고있는 새로운 플레이스 홀더를 생성한다.
# todo placeholder는 비어있는 메모리 공간을 잡아두는 것이다.

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# todo 비용함수를 구현하는데 여기서 사용되는 reduce_sum은 차원축소 후 sum 하는 함수이다. 4차원을 2차원으로 차원 축소한다.

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# todo 학습속도 0.01과 SGD(GradientDescentOptimizer) 경사하강법으로 비용함수의 오차가 최소화 되게 역전파 시킨다.

sess = tf.Session()
# todo 텐써플로우 그래프 연산을 시작하겠금 세션객체를 생성

sess.run(tf.global_variables_initializer())
# todo 모든 변수를 초기화 한다.

for i in range(1000):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    # todo 784,라벨이 들어간다.
    # todo 훈련데이터 셋에서 무작위로 100개를 추출
    sess.run(train_step, feed_dict= {x: batch_xs , y_: batch_ys})
    # todo 100개의 SGD의 경사감소법으로 데이터를 훈련시키는 과정

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # todo y라벨(예상값)중에 가장 큰 index를 리턴하고
    # todo y_라벨(실제값)중 가장 큰 index를 리턴해서 같은지 비교한다.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # todo [True,False,False,True]를 [1,0,0,1]로 변경 될것이고
    # todo 평균 0.75가 출력된다.
    print( sess.run(accuracy,feed_dict={ x : mnist.test.images, y_: mnist.test.labels}))
    # todo 정확도가 출력이 된다.
# ■ 위의 단층 신경망 코드에 중요 문법 3가지
# 1.reduce_sum과 reduce_mean
# reduce_sum
import numpy as np
import tensorflow as tf

x = np.arange(6).reshape(2,3)
print(x)
sess = tf.Session()

print(sess.run(tf.reduce_sum(x)))
print(sess.run(tf.reduce_sum(x,0))) # 열 단위로 sum한다.
print(sess.run(tf.reduce_sum(x,1))) # 행 단위로 sum한다.

# 2.feed_dict
# 3.cast

# 문제5. 숫자 2로 채워진 2행 3열의 행렬을 만들고 숫자 3으로 채워진 2행 3열의 행렬을 만든 후 두 행렬의 합을 출력하시오
a = tf.placeholder('float' ,[2,3])
b = tf.placeholder('float', [2,3])
add = tf.add(a, b)
sess = tf.Session()
print(sess.run(a, feed_dict={a:[[2,2,2], [2,2,2]]}))
print(sess.run(b, feed_dict={b:[[3,3,3], [3,3,3]]}))
print(sess.run(add, feed_dict={a:[[2,2,2], [2,2,2]], b:[[3,3,3], [3,3,3]]}))
#####
x = tf.fill([2,3],2)
y = tf.fill([2,3],3)


# 문제6. 숫자 2로 채워진 2x3행렬과 숫자 3으로 채워진 3x2행렬의 행렬곱을 출력하시오
a = tf.placeholder('float' ,[2,3])
b = tf.placeholder('float', [3,2])
matmul = tf.matmul(a, b)
sess = tf.Session()
print(sess.run(a, feed_dict={a:[[2,2,2], [2,2,2]]}))
print(sess.run(b, feed_dict={b:[[3,3], [3,3], [3,3]]}))
print(sess.run(matmul, feed_dict={a:[[2,2,2], [2,2,2]], b:[[3,3], [3,3], [3,3]]}))
#####
x = tf.fill([2,3],2)
y = tf.fill([3,2],3)


# 문제7. True를 1로 False를 0으로 출력하시오
correct_prediction = [ True, False , True  ,True  ,True  ,True  ,True,  True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True, False , True  ,True, False , True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True,
  True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True ,False , True  ,True  ,True  ,True  ,True
  ,True  ,True, False , True, False , True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
 ,False , True  ,True  ,True]
sess = tf.Session()
accuracy = tf.cast(correct_prediction, 'float')
print(sess.run(accuracy))


# 문제8. 위에서 출력한 100개의 숫자의 평균값을 출력하시오!
mean = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(sess.run(mean))


# 문제10. 무성이 코드 보기## 킹무성@@@킹무성@@@킹무성@@@킹무성@@@킹무성@@@킹무성@@@킹무성@@@킹무성@@@킹무성@@@
import tensorflow as tf
import numpy as np
from dataset.mnist import load_mnist

##### mnist 데이터 불러오기 및 정제 #####

############################################
# mnist 데이터 중 10000개 저장
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True)
input = np.concatenate((x_train, x_test), axis=0)
target = np.concatenate((t_train, t_test), axis=0)
print('input shape :', input.shape, '| target shape :', target.shape)
a = np.concatenate((input, target), axis=1)
np.savetxt('mnist.csv', a[:10000], delimiter=',')
print('mnist.csv saved')
############################################

# 파일 로드 및 변수 설정
save_status = True
load_status = True

mnist = np.loadtxt('mnist.csv', delimiter=',', unpack=False, dtype='float32')
print('mnist.csv loaded')
print('mnist shape :',mnist.shape)

train_num = int(mnist.shape[0] * 0.8)

x_train, x_test = mnist[:train_num,:784], mnist[train_num:,:784]
t_train, t_test = mnist[:train_num,784:], mnist[train_num:,784:]

print('x train shape :',x_train.shape, '| x target shape :',x_test.shape)
print('t train shape :',t_train.shape, '| t target shape :',t_test.shape)

global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.placeholder(tf.float32,[None, 784])
T = tf.placeholder(tf.float32,[None, 10])
W = tf.Variable(tf.random_uniform([784,10], -1e-7, 1e-7)) # [784,10] 형상을 가진 -1e-7 ~ 1e-7 사이의 균등분포 어레이
b = tf.Variable(tf.random_uniform([10], -1e-7, 1e-7))    # [10] 형상을 가진 -1e-7 ~ 1e-7 사이의 균등분포 벡터
Y = tf.add(tf.matmul(X,W), b) # tf.matmul(X,W) + b 와 동일

############################################
# 그외 가중치 초기화 방법
# W = tf.Variable(tf.random_uniform([784,10], -1, 1)) # [784,10] 형상을 가진 -1~1 사이의 균등분포 어레이
# W = tf.get_variable(name="W", shape=[784, 10], initializer=tf.contrib.layers.xavier_initializer()) # xavier 초기값
# W = tf.get_variable(name='W', shape=[784, 10], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 초기값
# b = tf.Variable(tf.zeros([10]))
############################################

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(cost, global_step=global_step)

############################################
# 그외 옵티마이저
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01)
############################################

##### mnist 학습시키기 #####
# 일반 버전
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#todo 로드 버전
# sess = tf.Session()
# saver = tf.train.Saver(tf.global_variables())
#
# cp = tf.train.get_checkpoint_state('./save') # save 폴더를 checkpoint로 설정
# # checkpoint가 설정되고, 폴더가 실제로 존재하는 경우 restore 메소드로 변수, 학습 정보 불러오기
# if cp and tf.train.checkpoint_exists(cp.model_checkpoint_path):
#     saver.restore(sess, cp.model_checkpoint_path)
#     print(sess.run(global_step),'회 학습한 데이터 로드 완료')
# # 그렇지 않은 경우 일반적인 sess.run()으로 tensorflow 실행
# else:
#     sess.run(tf.global_variables_initializer())
#     print('새로운 학습 시작')

# epoch, batch 설정
epoch = 100
total_size = x_train.shape[0]
batch_size = 100
# mini_batch_size = 100
total_batch = int(total_size/batch_size)

# 정확도 계산 함수
correct_prediction = tf.equal(tf.argmax(T, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 설정한 epoch 만큼 루프
for each_epoch in range(epoch):
    total_cost = 0
    # 각 epoch 마다 batch 크기만큼 데이터를 뽑아서 학습
    for idx in range(0, total_size, batch_size):
        batch_x, batch_y = x_train[idx:idx+batch_size], t_train[idx:idx+batch_size]

        _, cost_val = sess.run([optimizer, cost], feed_dict={X : batch_x, T : batch_y})
        total_cost += cost_val

    print('Epoch:', '%04d' % (each_epoch + 1),
          'Avg. cost =', '{:.8f}'.format(total_cost / total_batch),
          )

print('최적화 완료!')

#todo 최적화가 끝난 뒤, 변수와 학습 정보 저장
# saver.save(sess, './save/mnist_dnn.ckpt', global_step=global_step)

##### 학습 결과 확인 #####
print('Train 정확도 :', sess.run(accuracy, feed_dict={X: x_train, T: t_train}))
print('Test 정확도:', sess.run(accuracy, feed_dict={X: x_test, T: t_test}))


# 문제1. 1) 가중치 초기값을 xavier 초기값으로 설정, 2) 옵티마이저를 momentum 옵티마이저로 설정 후, 3) epoch은 200번,
#       4) batch_size 는 200으로 수정하여 학습해보기

# 문제2. 19~20번째 줄의 save_status 와 load_status가 각각 True 인 경우에만 저장/불러오기 되도록 코드 수정

# 문제3. 82번째 줄의 mini_batch_size를 이용하여 200개의 배치 데이터 중 100개만 랜덤으로 뽑아 학습하도록 코드 수정
#       (힌트 : np.random.randint(low=a, high=b, size=c) --> 숫자 a~b 사이의 정수 c개를 랜덤으로 뽑아주는 함수)

# 문제4. 훈련데이터의 10%를 뽑아 만든 검증 데이터로 아래 형식과 같이 50번째 epoch 마다 정확도 출력해보기.
#         (훈련데이터(0.9) + 검증데이터(0.1) = 전체의 80%   /   테스트 데이터 = 전체의 20%)
'''
Epoch: 0048 Avg. cost = 0.21684368
Epoch: 0049 Avg. cost = 0.16016438
Epoch: 0050 Avg. cost = 0.24727620
=================================
50번째 검증 데이터 정확도 : 0.881
=================================
Epoch: 0051 Avg. cost = 0.22011646
Epoch: 0052 Avg. cost = 0.17041421
Epoch: 0053 Avg. cost = 0.13844220
'''

##########################################
###### 텐서플로우로 2층 신경망 구현하기 ######
##########################################
##########################################
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
import tensorflow as tf

x = tf.placeholder('float', shape=[None, 784])  # 입력값을 담을 변수
y_ = tf.placeholder('float', shape=[None, 10])  # 라벨을 담을 변수 (mnist 라벨 10개)

x_image = tf.reshape(x, [-1,28,28,1])   # 입력 이미지를   cnn에 입력하기 위해 reshape
print('x_image=')
print(x_image)

def weight_variable(shape):   # 가중치의 값을 초기화 해주는 함수
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):     # 바이어스를 초기화해주는 함수
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):   # 스트라이드는 1로 하고 패딩은 output 사이즈가 입력과달라지지 않게하겠다고 'SAME'을 사용한다.
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')  # 아주 간단하다.

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# ksize(filter 사이즈) 윈도루 사이즈를 2로 하고 스트라이드도 2로 하겠다.
# 스트라이드도 2로 하기 때문에 기존 사이즈가 절반으로 줄어들 것이다.

W_conv1 = weight_variable([5,5,1,32])
# 가로5, 세로5, inpput채널1, output채널32로 하는 가중치 매개변수 생성
# feature map을 32개 생성하겠다.
b_conv1 = bias_variable([32])   # 편향도 32개 만들어서 더해줌

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 입력값과 가중치의 합성곱에 편향을 더한걸 relu함수에 집어넣음
h_pool1 = max_pool_2x2(h_conv1)   # 위의 결과를 풀링계층에 입력

W_conv2 = weight_variable([5,5,32,64])   # 두번째 conv계층에서 쓰일 가중치인데 가로5, 세로5, 입력값32, 출력값이64인 가중치 매개변수
b_conv2 = bias_variable([64])   # 편향 64개 생성

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 첫번째 conv 계층과 pooling까지 통과한 결과가
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool1.get_shape())

W_fc1 = weight_variable([7*7*64, 1024])   # affine 계층에서 쓰일 가중치 매개변수를 생성, 가로7, 세로7, feature map64, 노드의개수1024
b_fc1 = bias_variable([1024])   # 편향1024


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder('float')   # dropout을 수행할지 말지 결정하기 위한 변수 생성
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)    # 오차함수에 오차를 최소화하는 경사감소법으로 Adam사용
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  # 실제라벨과 예상라벨을 비교한 결과를 correct_predition에 담기
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))      # 불리언을 인티저로 바꿔 평균

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

test_x, test_y = mnist.test.next_batch(1000)
print("test accuracy %g"% sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
# 맨 마지막 test할 때는 dropout하지 않는다.














