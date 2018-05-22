from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['backend'] = "qt4agg"
plt.rcParams['backend.qt4'] = "PySide"
from tensorflow.examples.tutorials.mnist import input_data
# read in data
mnist = input_data.read_data_sets("", one_hot=True)

# define number of hidden layers and nodes per hidden layer
n_nodes_hl = 784
n_nodes_h2 = 1500
n_nodes_h3 = 1500

# define number of classes and batch size to avoid ram overload
n_classes = 10
batch_size = 1000

# x is data, flattened out, so it will be a 1 x 784. it's well defined, so that the data of that shape is input and random data may not be pushed into the tf
x = tf.placeholder ('float',[None, 784])
# y is label
y = tf.placeholder ('float',[None, 10])

def neural_network_model(data):
  hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl]))} 
  hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl, n_nodes_h2])), 'biases': tf.Variable(tf.random_normal([n_nodes_h2]))} 
  hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])), 'biases': tf.Variable(tf.random_normal([n_nodes_h3]))} 
  output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_h3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))} 
  
  # computation at each node = (input-data * wieghts) + biases
  l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['biases'])
  l1 = tf.nn.relu(l1)

  l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'])
  l2 = tf.nn.relu(l2)
  
  l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),hidden_3_layer['biases'])
  l3 = tf.nn.relu(l3)

  output = tf.matmul(l3, output_layer['weights'])+ output_layer['biases']

  return(output)
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #number for cycles
    hm_epochs = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,  cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
train_neural_network(x)


# Training Parameters
learning_rate = 0.01
num_steps = 3000
batch_size = 800

display_step = 1000
examples_to_show = 100

alpha_set = [4000]
# Network Parameters
num_hidden_1 = 256 # 1st layer num features (originally 256)
num_hidden_2 = 256 # 2nd layer num features (the latent dim) (originally 128)
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

for alpha in alpha_set:

    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)
    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X
    target_pred_adv = neural_network_model(y_pred)
    target_pred_true = neural_network_model(y_true)
    # Define loss and optimizer, minimize the squared error
    loss = (tf.reduce_mean(tf.pow(y_true - y_pred, 2))) #- tf.abs(tf.reduce_max(tf.nn.softmax(logits=target_pred_adv))-tf.reduce_max(tf.nn.softmax(logits=target_pred_true)))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    n = 10
    output = neural_network_model(X)
    classification_orig=np.empty(n)
    classification_adv=np.empty(n)
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

# Start Training
# Start a new TF session

    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training
        for i in range(1, num_steps+1):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x, _ = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))

        # Testing
        # Encode and decode images from test set and visualize their reconstruction.
    with tf.Session() as sess:
               
        for i in range(n):
            # MNIST test set
            batch_x, _ = mnist.test.next_batch(1000)
            # Encode and decode the digit image
            g = sess.run(decoder_op, feed_dict={X: batch_x})
            # test images by applying them to target
            
            
            classification_orig = sess.run(tf.argmax(output,1), feed_dict={x: batch_x})
            classification_adv = sess.run(tf.argmax(output,1), feed_dict={x: g})
            print('orig:',classification_orig,' adv:',classification_adv)
            print('Accuracy for alpha',alpha,'=', np.sum(classification_adv!=classification_orig))
            
        canvas_orig = np.empty((28 * n, 28 * n))
        canvas_recon = np.empty((28 * n, 28 * n))
        for i in range(n):
            for j in range(n):           
                # Draw the original digits
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    batch_x[j].reshape([28, 28])
                #prediction_orig = neural_network_model(batch_x[j].reshape([1,784]))
                #print('prediction for original image ',i,j,'is', tf.nn.softmax(prediction_orig))
            # Display reconstructed images
        for i in range(n):
            for j in range(n):
                # Draw the reconstructed digits
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    g[j].reshape([28, 28])
                
                #prediction_adv = neural_network_model(x)
                #print(sess.run(prediction_adv, feed_dict= {x: g[j].reshape(mnist.test.images.shape)}))
                #print('prediction for adv image ',i,j,'is',tf.run(prediction_adv))
        print("Original Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.show()

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()

"""
output = neural_network_model(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    classification_orig=np.empty(n)
    classification_adv=np.empty(n)
    difference = 0
    for j in range(n):
        classification_orig[j] = sess.run(tf.argmax(output,1), feed_dict={x: batch_x[j].reshape(1,784)})
        classification_adv[j] = sess.run(tf.argmax(output,1), feed_dict={x: g[j].reshape(1,784)})
        print('orig:',classification_orig[j],' adv:',classification_adv[j])
        if classification_adv[j] != classification_orig[j]:
            difference += 1
print('Difference',difference)
"""
    
     