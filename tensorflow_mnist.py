import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 

# mnist = input_data #google providing us datasets 
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
mnist.train.images

#shape specifies which until what they would accept
x = tf.placeholder(dtype = tf.float32, shape = [None, 784]) 
b = mnist.train.labels[1]
b.shape
print(mnist.train.images)

w = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))
print(b)

y = tf.nn.softmax(tf.matmul(x,w)+b)
y_ = tf.placeholder(dtype = tf.float32, shape = [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for steps in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels}))