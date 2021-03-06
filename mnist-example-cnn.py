import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read in data
mnist = input_data.read_data_sets("", one_hot=True)

# define number of hidden layers and nodes per hidden layer


# define number of classes and batch size to avoid ram overload
n_classes = 10
batch_size = 128

# x is data, flattened out, so it will be a 1 x 784. it's well defined, so that the data of that shape is input and random data may not be pushed into the tf
x = tf.placeholder ('float',[None, 784])
# y is label
y = tf.placeholder ('float')

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window   movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights  = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
                'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
                'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
                'out':tf.Variable(tf.random_normal([1024,n_classes])),} 
    baises  = {'b_conv1':tf.Variable(tf.random_normal([32])),
                'b_conv2':tf.Variable(tf.random_normal([64])),
                'b_fc':tf.Variable(tf.random_normal([1024])),
                'out':tf.Variable(tf.random_normal([n_classes])),} 
    x = tf.reshape(x, shape=[-1,28,28,1])
    
    conv1 = conv2d(x, weights['W_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['W_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+baises['b_fc'])

    output = tf.matmul(fc, weights['out'])+ baises['out']

    return(output)
def train_neural_network(x):
    prediction =  convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #number for cycles
    hm_epochs = 20

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
