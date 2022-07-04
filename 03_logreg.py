#coding:utf-8
import tensorflow as tf
import numpy as np
#disable warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('/Users/huipang.lvjj/1_work/cs20_tf/stanford-tensorflow-tutorials/examples')
import utils

# Step 1: read in the data
mnist_folder = './mnist'
#utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: create Dataset and iterator
batch_size = 10000

train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)

val_data = tf.data.Dataset.from_tensor_slices(val)
val_data = val_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
val_init = iterator.make_initializer(val_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data


def net_linear(img):
    # Step 3: create weight and bias, initialized to 0
    w = tf.get_variable('weights',shape=(784,10),initializer=tf.random_normal_initializer(0,0.01))
    b = tf.get_variable('bias',shape=(1,10),initializer=tf.zeros_initializer())
    return tf.matmul(img,w) + b

def net_linear_2(img):
    w1 = tf.get_variable('weights1',shape=(784,100),initializer=tf.random_normal_initializer(0,0.01))
    b1 = tf.get_variable('bias1',shape=(1,100),initializer=tf.zeros_initializer())
    w2 = tf.get_variable('weights',shape=(100,10),initializer=tf.random_normal_initializer(0,0.01))
    b2 = tf.get_variable('bias',shape=(1,10),initializer=tf.zeros_initializer())

    out1 = tf.matmul(img,w1) + b1
    return tf.matmul(out1,w2) + b2

# Step 4
logits = net_linear_2(img)

# Step 5, loss
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')# mean over all the examples in batch

# Step 6, 
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# Step 7, acc
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(label,1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 30
    for i in range(epochs):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer,loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
        
        # validate the model
        if i%10 != 9:
            continue
        sess.run(val_init)
        t0 = 0
        try:
            while True:
                correct_v = sess.run(accuracy)
                t0 += correct_v
        except tf.errors.OutOfRangeError:
            pass
        n_val = 5000
        print('val acc {0}'.format(t0/n_val))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass
    n_test = 10000
    print('Accuracy {0}'.format(total_correct_preds/n_test))
