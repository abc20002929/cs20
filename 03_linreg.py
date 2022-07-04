#coding:utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#disable warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('/Users/huipang.lvjj/1_work/cs20_tf/stanford-tensorflow-tutorials/examples')
import utils

DATA_FILE = 'stanford-tensorflow-tutorials/examples/data/birth_life_2010.txt'
# Step 1: read in the data
data, n_samples = utils.read_birth_life_data(DATA_FILE)
# Step 2: create Dataset and iterator
n_samples = 5
data = np.array([[37.887,10],[24.935,15],[19.067,20],[15.913,25],[13.67,30]],dtype=np.float32)
#data = np.array([[21.6,18],[13.7,28],[16.7,24]],dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
iterator = dataset.make_initializable_iterator()
X,Y = iterator.get_next()
# Step 3: create weight and bias, initialized to 0
w = tf.get_variable('weights',initializer=tf.constant(0.0))
b = tf.get_variable('bias',initializer=tf.constant(0.0))

# Step 4
Y_predicted = X * w + b

'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    for i in range(4):
        #X,Y = iterator.get_next()
        print(sess.run([X,Y,Y_predicted]))
'''

# Step 5, loss
loss = tf.square(Y - Y_predicted,name='loss')

# Step 6, 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 10000
    for i in range(epochs):
        sess.run(iterator.initializer)
        total_loss = 0
        try:
            while True:
                _, l = sess.run([optimizer,loss])
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass
        print('Epoch {0}:{1}'.format(i,total_loss/n_samples))

    w_out,b_out = sess.run([w,b])
    print('w:%f,b:%f'%(w_out,b_out))

    
# plot the results
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data with squared error')
# plt.plot(data[:,0], data[:,0] * (-5.883589) + 85.124306, 'g', label='Predicted data with Huber loss')
plt.legend()
plt.show()



