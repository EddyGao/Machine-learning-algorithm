# -*- coding:utf-8 -*-  
#tensorflow实现线性回归

import tensorflow as tf 
import numpy as np 

x_data = np.random.rand(100).astype(np.float32)
y_data = 0.3*x_data+ 0.1

W = tf.Variable(tf.random_uniform([1] , -1.0,1.0))
b = tf.Variable(tf.zeros([1]))

y = W*x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100):
	sess.run(train)
	if step % 10 == 0 :
		print sess.run(W) , ',  ',sess.run(b)
