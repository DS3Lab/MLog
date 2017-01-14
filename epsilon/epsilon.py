#!/usr/bin/python

import tensorflow as tf
import numpy as np
from IPython import embed
from time import time
import struct

data_path = 'epsilon.data'

batch_size = 100
feature = 2000


def get_data():
	global batch_size
	global feature

	f = open(data_path)
	X = 28
	Y = X + 400000 * 2000 * 4 + 24
	N = 400000
	i = 0
	while True:
	    if i >= 400000:
		i = 0
	    f.seek(X + 2000 * i * 4)
	    x = struct.unpack(">%df" % (batch_size * 2000), f.read(batch_size * 2000 * 4))
	    x = np.array(x)
	    x = np.resize(x, [batch_size, 2000])

	    f.seek(Y + i * 4)
	    y = struct.unpack(">%df" % (batch_size), f.read(batch_size * 4))
	    y = np.array(y)
	    y = np.resize(y, [batch_size])

	    i += batch_size
	    yield(x, y)

def main():
	with tf.device("/gpu:0"):
	    x = tf.placeholder(tf.float32, shape=(batch_size, feature))
	    y = tf.placeholder(tf.float32, shape=(batch_size))
	    W = tf.Variable(tf.random_normal((feature, 1), mean=0.0, stddev=0.03))
	    b = tf.Variable(tf.zeros(()))

	    p = tf.matmul(x, W) + b

	    loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - y * tf.squeeze(p)))

	    train = tf.train.AdamOptimizer(0.01).minimize(loss)

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	data = get_data()
	start = time()
	log = open("log_epsilon_tf_gpu", "w")
	test_time = 0.0
	for epoch in range(1, 201):
	    for batch in range(1, 3201):
		_x, _y = data.next()
		result = sess.run([train, loss], feed_dict={x:_x, y:_y})
		log.write("%.3f\t%d\t%d\t%.8f\n" % (time()-start-test_time, epoch, batch, result[1]))
		log.flush()

	    accu = 0
	    a = time()
	    for batch in range(800):
		_x, _y = data.next()
		result = sess.run(p, feed_dict={x:_x, y:_y})
		accu += np.count_nonzero(np.transpose(result) * _y > 0)
	    log.write("%d\t%.6f\n" % (epoch, accu / 80000.0))
	    log.flush()
	    b = time()
	    test_time += b - a
	log.close()

if __name__ == '__main__':
	main()

