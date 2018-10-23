import tensorflow as tf

#disable warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#graphs & sessions
x = 3
y = 5
op1 = tf.multiply(x,y)
op2 = tf.add(x,x)
a = tf.add(op1,op2)
with tf.Session() as sess:
    print(sess.run(a))
    #print(sess.run([op1,op2]))
    

#two graphs , better to use subgraph in one graph
'''
g1 = tf.get_default_graph()
g2 = tf.Graph()
with g1.as_default():
    a = tf.Constant(3)
with g2.as_default():
    b = tf.Constant(5)
'''
