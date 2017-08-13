
# coding: utf-8

# ## 2.1 MNIST 데이터 불러오기

# In[ ]:


import tensorflow as tf


# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data


# In[ ]:


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# ## 2.2 그래프 구성

# In[ ]:


x = tf.placeholder('float', shape=[None, 784], name='images_flat')
y_ = tf.placeholder('float', shape=[None, 10], name='labels')


# In[ ]:


x_image = tf.reshape(x, [-1, 28, 28, 1], name='batch_images') # N x 784 -> N x 28 x 28 x 1
print(x_image)


# ### 2.2.1 가중치, 편향 생성 함수 정의
# 각 레이어마다 가중치와 편향을 부여할때 중복되는 코드 작성을 줄이기 위해 가중치, 편향의 형상을 인수로 하는 함수를 작성하여 이용한다. weight_variable 과 bias_variable 은 shape 를 인수로 받아 Variable 을 반환한다.
# - tf.truncated_normal(shape, stddev=.1)    
#     : 표준편차를 .1 로 하는 정규분포를 따라 난수를 생성하는데 평균으로부터 표준편차의 2배를 넘는 범위의 값은 폐기하고 다시 생성한다.
# 
# 
# <img src='1.png'>

# In[ ]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)


# ### 2.2.2 합성곱, pooling 함수 정의
# convolution, pooling 레이어도 마찬가지로 중복되는 코드를 줄이기 위해 함수를 선언해서 기능을 수행한다.
# - tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')    
#     : 4-D tensor 입력에 대해 2-D 합성곱을 수행. 입력 tensor 는 [N, H, W, C] 형상을 갖고, 필터(커널)은 [FH, FW, C, FN] 형상을 갖는다. strides 는 1-D tensor 의 원소들이 순서대로 N, H, W, C 방향의 필터 이동 간격이다. padding 옵션은 'VALID' 인 경우 tensor 에 padding 없이 필터 크기와 stride 에 의해 출력 크기가 결정되고, 'SAME' 인 경우 padding 을 하여 stride 에 의해서만 출력 크기가 결정된다.    
#     
#     
# - tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')    
#     : 4-D tensor 입력에 대해 max pooling 을 수행. 입력 tensor 는 [N, H, W, C] 형상을 갖고, ksize 는 pooling 윈도우 사이즈이며, strides 와 padding 옵션은 conv2d 함수와 같다.

# In[ ]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# ### 2.2.3 계층 연결
# <img src='2.png'>

# #### 2.2.3.1 conv1
# 2.2.2 에서 padding 옵션을 SAME 으로 설정하여 padding 을 하므로 pooling stride 에 의해서 출력의 형상이 결정된다. stride 가 2 이므로 출력은 입력 이미지의 절반 크기가 된다.
# <img src='conv1.png'>

# In[ ]:


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# In[ ]:


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# #### 2.2.3.2 conv2
# <img src='conv2.png'>

# In[ ]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])


# In[ ]:


h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# #### 2.2.3.3 affine1
# 4-D 입력를 2-D 로 평탄화하고 가중치를 곱하고 편향을 더한다. 출력에 dropout 을 적용한다.
# <img src='affine1.png'>
# - tf.nn.dropout(h_fc1, keep_prob)    
#     : 출력되는 tensor 의 각 원소들을 keep_prob 의 확률로 다음 계층으로 전달하고 전달되지 못하는 원소들은 0으로 대체하여 전달한다.

# In[ ]:


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])


# In[ ]:


h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[ ]:


keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# #### 2.2.3.4 affine2 & softmax

# In[ ]:


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


# In[ ]:


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# ## 2.3 세션 실행

# In[ ]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


# In[ ]:


sess = tf.Session()


# In[ ]:


sess.run(tf.global_variables_initializer())


# In[ ]:


for i in range(10000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: .5})


# In[ ]:


print('test accuracy %g' % 
      sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.}))


# In[ ]:


writer = tf.summary.FileWriter('./tb', sess.graph)
writer.close()


# 작성된 그래프 - 위 코드 실행하면 경로에 로그 파일 생성. cmd -> 가상환경 접근 -> tensorboard --logdir= 경로 -> 출력되는 url 로 접속
# <img src='graph.png'>
