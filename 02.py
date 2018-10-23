#coding:utf-8
import tensorflow as tf

#disable warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#TensorBoard
x = 3
y = 5
a = tf.add(x,y)

#
#create the summary *AFTER* graph definition OR *BEFORE* running your session
#
writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())
with tf.Session() as sess:
    #writer = tf.summary.FileWriter('./graphs',sess.graph)
    print(sess.run(a))   
writer.close()

#tensorboard --logdir="./graphs" --port 6006 #http://localhost:6006/




#constants
b = tf.constant([[0,1],[2,3]], name='b')
#zeros/ones
z = tf.zeros([2,3],tf.int32)  #[[0,0,0],[0,0,0]]
z1= tf.zeros_like(b) #[[0,0],[0,0]]
#fill
f = tf.fill([2,3],8) #[[8,8,8],[8,8,8]]



#constant保存在graph中，而Variable保存在内存-tf会分配。前者小写属于单个op,后者大写属于类(可使用多个op)
#print(tf.get_default_graph().as_graph_def())
v1 = tf.Variable([[0,1],[2,3]],name="v1")
v2 = tf.get_variable("v2",initializer=tf.constant([[0,1],[2,3]]))#建议

#Variable必须初始化, 或者用assign op代替(init op也是一个assing op)
#assign_op = v1.assign(....)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([v1,v2]))
    print(v1.eval()) # same as -> sess.run(v1)

