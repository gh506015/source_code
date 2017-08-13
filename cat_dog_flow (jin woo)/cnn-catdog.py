
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import os


# In[ ]:


path = 'C:\data\cat_dog_data//'  # 여기 수정해서 쓰세요~
train_file = {0: [], 1: []}
test_file = {0: [], 1: []}

for _, _, files in os.walk(path):
    for file in files:
        if file.startswith('test'):
            if file.endswith('0.csv'):
                test_file[0].append(file)
            else : test_file[1].append(file)
        else :
            if file.endswith('0.csv'):
                train_file[0].append(file)
            else : train_file[1].append(file)
print(train_file, test_file)

train_data = {0: None, 1: None}
test_data = {0: None, 1: None}

for i in ['train', 'test']:
    for key, val in eval(i + '_file').items():
        for file in val :
            print(i, key, file)
            temp = np.loadtxt(path + file, delimiter=',') / 255
            temp = np.hstack((temp, np.ones((temp.shape[0], 1))*key))
            temp_name = eval(i + '_data')
            if temp_name[key] is None :
                temp_name[key] = temp
            else :
                temp_name[key] = np.vstack((temp_name[key], temp))
            print(temp_name[key].shape)


# In[ ]:


training = tf.placeholder(tf.bool, name='training')

def BN(input, training, scale=True, decay=0.99):
    return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale, is_training=training, updates_collections=None)


# In[ ]:


x = tf.placeholder('float', shape=[None, 128 * 128], name='images_flat')
y_ = tf.placeholder(tf.int64, shape=[None], name='labels')


# In[ ]:


x_image = tf.reshape(x, [-1, 128, 128, 1], name='batch_images')
print(x_image)


# In[ ]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)


# In[ ]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[ ]:


W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])


# In[ ]:


cc = conv2d(x_image, W_conv1) + b_conv1
dd = BN(cc, training=training)
h_conv1 = tf.nn.relu(dd)
h_pool1 = max_pool_2x2(h_conv1)


# In[ ]:


W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])


# In[ ]:


aa = conv2d(h_pool1, W_conv2) + b_conv2
bb = BN(aa, training=training)
h_conv2 = tf.nn.relu(bb)
h_pool2 = max_pool_2x2(h_conv2)


# In[ ]:


W_fc1 = weight_variable([32 * 32 * 32, 1024])
b_fc1 = bias_variable([1024])


# In[ ]:


h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[ ]:


keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[ ]:


W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])


# In[ ]:


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# In[ ]:


cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


# In[ ]:


sess = tf.Session()


# In[ ]:


sess.run(tf.global_variables_initializer())


# In[ ]:


batch_0, batch_1 = np.random.choice(train_data[0].shape[0], 50), np.random.choice(train_data[1].shape[0], 50)
batch = np.vstack((train_data[0][batch_0], train_data[1][batch_1]))
np.random.shuffle(batch)
sess.run(tf.to_int32(batch[0, -1]))


# In[ ]:


for i in range(10000):
    batch_0, batch_1 = np.random.choice(train_data[0].shape[0], 50), np.random.choice(train_data[1].shape[0], 50)
    batch = np.vstack((train_data[0][batch_0], train_data[1][batch_1]))
    np.random.shuffle(batch)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[:, :-1], y_: batch[:, -1], keep_prob: 1., training: True})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[:, :-1], y_: batch[:, -1], keep_prob: .5, training: True})


# In[ ]:


test = np.vstack((test_data[0], test_data[1]))
np.random.shuffle(test)
print(test.shape)   # 13000
acc_lst = []
for i in range(0, 13000, 100):
    batch_x = test[i:i+100, :-1]
    batch_y = test[i:i+100, -1]
    acc = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1., training :False})
    acc_lst.append(acc)
print(sum(acc_lst) / len(acc_lst))

