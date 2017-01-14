#!/usr/bin/python

import tensorflow as tf
import numpy as np

dataset_path = "imdb_new" 

batch_size = 100

vocab_size = 20000
max_length = 300

feature = 100
lamda = 5e-5

dataset = []

f = open(dataset_path)
for l in f:
	l = l.strip().split()
	label = int(l[0])
	seq = map(int, l[2:])
	dataset.append((label, seq[:max_length]))

def data():
	global dataset, batch_size, max_length

	for i in range(len(dataset) // batch_size):
		start = i * batch_size
		end = start + batch_size
		batch = dataset[start : end]
		X = np.zeros([batch_size, max_length], np.int32)
		Y = np.zeros(batch_size, np.int32)
		L = np.zeros(batch_size, np.int32)
		j = 0
		for label, seq in batch:
			Y[j] = (label + 1) // 2
			L[j] = len(seq)
			for k in range(len(seq)):
				X[j][k] = seq[k]
			j += 1
		yield (X, Y, L)

embedding_matrix = tf.Variable(tf.random_normal((vocab_size, feature), mean=0.0, stddev=0.01))
tf.add_to_collection('losses', tf.nn.l2_loss(embedding_matrix) * lamda)

seq = tf.placeholder(shape=(batch_size, max_length), dtype=tf.int32)
inputs = tf.nn.embedding_lookup(embedding_matrix, seq)

label = tf.placeholder(shape=(batch_size), dtype=tf.int32)
logit = tf.one_hot(label, 2)

length = tf.placeholder(shape=(batch_size), dtype=tf.int32)

cell = tf.nn.rnn_cell.LSTMCell(feature, state_is_tuple=True) 
outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

pooling = tf.reduce_mean(outputs, 1)

lr_W = tf.Variable(tf.random_normal((feature, 2), mean=0.0, stddev=0.1))
tf.add_to_collection('losses', tf.nn.l2_loss(lr_W) * lamda)
lr_b = tf.Variable(tf.zeros((1,2)))

pred = tf.matmul(pooling, lr_W) + lr_b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, logit))

tf.add_to_collection("losses", loss)
ans = tf.argmax(pred, 1)

total_loss = tf.add_n(tf.get_collection("losses"))

train = tf.train.AdamOptimizer(0.01).minimize(total_loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

from time import time
start = time()
test_time = 0.0
log = open("log_imdb_tf_gpu", "w")

for epoch in range(1,201):
	d = data()
	for batch in range(1,251):
		X, Y, L = d.next()
		_, l = sess.run([train, loss], feed_dict={seq:X, label:Y, length:L})
		log.write("%.3f\t%d\t%d\t%.8f\n" % (time()-start-test_time, epoch, batch, l))
		log.flush()
	right = 0
	for batch in range(250):
		X, Y, L = d.next()
		a = sess.run(ans, feed_dict={seq:X, label:Y, length:L})
		right += np.count_nonzero(a == Y)
	log.write("%d\t%.6f\n" % (epoch, right / 25000.0))
	log.flush()

log.close()

