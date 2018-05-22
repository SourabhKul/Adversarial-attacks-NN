import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from random import randint
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
y = tf.placeholder ('float', [None, 10])

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
    hm_epochs = 3

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

output = neural_network_model(x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num = randint(0,mnist.test.images.shape[0])
    img = mnist.test.images[num]
    classification = sess.run(output, feed_dict={x: [img]})
    print(classification)

    