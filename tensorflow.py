# TENSORFLOW 1.9.0-rc2

import tensorflow as tf

a = tf.constant(5)
sess = tf.InteractiveSession()
# print(sess.run(a))

b = tf.placeholder(dtype=tf.float32) 
c = tf.placeholder(dtype=tf.float32) 
#placeholder : first run no value, empty cell first
# get actual value when we run it 

d = b + c
print(sess.run(d, feed_dict = {b:10, c:10}))
# print(sess.run(d))

w = tf.Variable(5)
sess.run(tf.global_variables_initializer())
print(sess.run(w))

print("TF VERSION" + tf.VERSION)