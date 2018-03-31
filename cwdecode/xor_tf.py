import tensorflow as tf

import config

#l1 = tf.layers.Dense(128, activation=tf.nn.relu)
#l2 = tf.layers.Dense(128, activation=tf.nn.relu)
#l3 = tf.contrib.rnn.BasicRNNCell(128, tf.nn.relu)
#l4 = tf.layers.Dense(128, activation=tf.nn.relu)

x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

_ = tf.layers.dense(x_,  2, activation=tf.sigmoid)
y = tf.layers.dense( _, 1, activation=tf.sigmoid)

cost = tf.reduce_mean( -1 * (y_ * tf.log(y) + (1 - y_) * tf.log(1.0 - y) ) )

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})

    if i % 1000 == 0:
        print('Epoch ', i)
        print('Hypothesis ', sess.run(y, feed_dict={x_: XOR_X, y_: XOR_Y}))
        #print('Theta1 ', sess.run(Theta1))
        #print('Bias1 ', sess.run(Bias1))
        #print('Theta2 ', sess.run(Theta2))
        #print('Bias2 ', sess.run(Bias2))
        print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
