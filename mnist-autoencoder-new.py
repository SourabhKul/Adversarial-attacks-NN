#

#   mnist_ae1.py   date. 7/4/2016

#   

#   Autoencoder tutorial code

#

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import numpy as np

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

import tensorflow as tf



# Import data

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)



# Variables

x = tf.placeholder("float", [None, 784])

y_ = tf.placeholder("float", [None, 10])



w_enc = tf.Variable(tf.random_normal([784, 625], mean=0.0, stddev=0.05))

w_dec = tf.Variable(tf.random_normal([625, 784], mean=0.0, stddev=0.05))

# w_dec = tf.transpose(w_enc) # if you use tied weights

b_enc = tf.Variable(tf.zeros([625]))

b_dec = tf.Variable(tf.zeros([784]))



# Create the model

def model(X, w_e, b_e, w_d, b_d):

    encoded = tf.sigmoid(tf.matmul(X, w_e) + b_e)

    decoded = tf.sigmoid(tf.matmul(encoded, w_d) + b_d)

    

    return encoded, decoded



encoded, decoded = model(x, w_enc, b_enc, w_dec, b_dec)



# Cost Function basic term

cross_entropy = -1. * x * tf.log(decoded) - (1. - x) * tf.log(1. - decoded)

loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)



# Train

init = tf.initialize_all_variables()



with tf.Session() as sess:

    sess.run(init)

    print('Training...')

    for i in range(20001):

        batch_xs, batch_ys = mnist.train.next_batch(256)

        train_step.run({x: batch_xs, y_: batch_ys})

        

        if i % 1000 == 0:

            train_loss = loss.eval({x: batch_xs, y_: batch_ys})

            print('  step, loss = %6d: %6.3f' % (i, train_loss))

        

    # generate decoded image with test data

    test_fd = {x: mnist.test.images, y_: mnist.test.labels}

    decoded_imgs = decoded.eval(test_fd)

    print('loss (test) = ', loss.eval(test_fd))

    

x_test = mnist.test.images



n = 10  # how many digits we will display

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



plt.savefig('mnist_ae1.png')