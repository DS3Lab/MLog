#!/usr/bin/python

import numpy as np
import tensorflow as tf
import time
import IPython

data_path = ["data/data_batch_%d" % i for i in range(1,6)]

batch_size = 100
classify_num = 10
lamda = 1e-4

data = []
answ = []

test = []
resu = []

def read_data():
	def unpickle(file):
		import cPickle
		fo = open(file, 'rb')
		dict = cPickle.load(fo)
		fo.close()
		return dict

	global data, answ, test, resu
	
	for path in data_path:
		dic = unpickle(path)
		data.append(dic['data'])
		answ.append(dic['labels'])

	data = np.array(data, dtype=np.float).reshape((-1, 3, 32,32)) / 128.0 - 1.0
	data = np.transpose(data, axes=[0,2,3,1])
	answ = np.array(answ, dtype=np.int).reshape((-1,))

	dic = unpickle('data/test_batch')
	test = np.array(dic['data'], dtype=np.float).reshape((-1, 3, 32, 32)) / 128.0 - 1.0
	test = np.transpose(test, axes=[0,2,3,1])
	resu = np.array(dic['labels'], dtype=np.int).reshape((-1,))

	IPython.embed()

def cnn():
    image = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 3))

    label = tf.placeholder(tf.int32, shape=(batch_size,))
    logit = tf.one_hot(label, classify_num)

    out = image

    with tf.device("/gpu:0"):

	L = 1
	T = 3
	K = 300

	while L <= 5:
	    w = tf.Variable(tf.random_normal((3, 3, T, K), mean=0.0, stddev=0.03))
	    tf.add_to_collection('losses', tf.nn.l2_loss(w) * lamda)
	    b = tf.Variable(tf.zeros((K,)))
	    out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(out, w, [1,1,1,1], "SAME"), b))


	    out = tf.nn.max_pool(out, [1,3,3,1], [1,2,2,1], "SAME")

	    w = tf.Variable(tf.random_normal((1, 1, K, K), mean=0.0, stddev=0.03))
	    tf.add_to_collection('losses', tf.nn.l2_loss(w) * lamda)
	    b = tf.Variable(tf.zeros((K,)))
	    out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(out, w, [1,1,1,1], "SAME"), b))

	    T = K
	    K += 300
	    L += 1

	out = tf.reshape(out, (batch_size, T))

	w = tf.Variable(tf.random_normal((T, T), mean=0.0, stddev=0.03))
	tf.add_to_collection('losses', tf.nn.l2_loss(w) * lamda)
	b = tf.Variable(tf.zeros((1,T)))
	out = tf.nn.relu(tf.matmul(out, w) + b )

	w = tf.Variable(tf.random_normal((T, 10), mean=0.0, stddev=0.1))
	tf.add_to_collection('losses', tf.nn.l2_loss(w) * lamda)
	b = tf.Variable(tf.zeros((1,10)))
	pred = tf.matmul(out, w) + b 

	result = tf.argmax(pred, 1)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, logit))
	tf.add_to_collection('losses', loss)
	total_loss = tf.add_n(tf.get_collection('losses'))

    train = tf.train.AdamOptimizer(0.0003).minimize(total_loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    start = 0
    stime = time.time()
    ttime = 0.0
    log = open("log_cifar_tf_gpu", "w")

    for epoch in range(100):
	start = 0
	global data, answ

	for batch in range(500):
	    end = start + batch_size
	    img = data[start:end]
	    lab = answ[start:end]
	    start += batch_size
	    _, l = sess.run([train, loss], feed_dict={image:img, label:lab})

	    log.write("%.3lf\t%d\t%d\t%.6lf\n" % (time.time() - stime - ttime, epoch+1, batch+1,l))
	    log.flush()

	a = time.time()
	right = 0
	for batch in range(100):
	    img = test[batch*batch_size : (batch+1)*batch_size]
	    lab = resu[batch*batch_size : (batch+1)*batch_size]
	    res = sess.run(result, feed_dict={image:img})
	    right += np.count_nonzero(res == lab)
	log.write("%d\t%.6f\n" % (epoch+1, right / 10000.0))
	log.flush()
	b = time.time()
	ttime += b - a 

def main():
	read_data()
	cnn()

if __name__ == '__main__':
	main()
