from mpi4py import MPI
import time
from enum import Enum
import random
from mnist_dataset import train, test
from mnist import get_mnist_data, set_mnist_data, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE, MNIST_TEST_BATCH_SIZE
import tensorflow as tf

class SimpleNet:
    def __init__(self, sess):
        self.sess = sess
        self.step_num = 0

        train_data, test_data = get_mnist_data()
        self.test_iterator = test_data.batch(MNIST_TEST_BATCH_SIZE).make_initializable_iterator()
        self.test_next = self.test_iterator.get_next()
        
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        one_hot_y_ = tf.one_hot(self.y_, 10)
        

        self.w_fc1 = tf.Variable(tf.truncated_normal([28*28*1, 10], stddev=0.1))
        self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[10]))

        x_flat = tf.reshape(self.x, [-1, 28*28*1])
        fc1 = tf.nn.relu(tf.matmul(x_flat, self.w_fc1) + self.b_fc1)
        self.y = tf.nn.dropout(fc1, self.keep_prob)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y_, logits=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(0.1)

        print(self.y.shape, one_hot_y_.shape)   
        correct_prediction = tf.equal(
            tf.argmax(self.y, axis=1), tf.argmax(self.y_, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.vars = [self.w_fc1, self.b_fc1]

    def initialize_variables(self):
        self.sess.run([var.initializer for var in self.vars])

    def get_accuracy(self):
        self.sess.run(self.test_iterator.initializer)
        size_accuracy = 0
        try:
            while True:
                test_images, test_labels = self.sess.run(self.test_next)
                batch_size = test_images.shape[0]
                batch_accuracy = self.sess.run(self.accuracy,
                                                feed_dict={self.x: test_images, self.y_: test_labels,
                                                            self.keep_prob: 1})
                size_accuracy += batch_size * batch_accuracy
        except tf.errors.OutOfRangeError:
            pass
        return size_accuracy / MNIST_TEST_SIZE


class Instruction(Enum):
    EXIT = 0
    INIT = 1
    TRAIN = 2
    GET = 3

set_mnist_data(train('MNIST_data/'), test('MNIST_data/'))
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    reqs = []
    for worker in range(1, comm.Get_size()):
        reqs.append(comm.isend(Instruction.INIT, dest=worker))
    for req in reqs:
        req.wait()

    reqs = []
    for worker in range(1, comm.Get_size()):
        reqs.append(comm.isend(Instruction.GET, worker))
    for req in reqs:
        req.wait()
    for worker in range(1, comm.Get_size()):
        data = comm.recv(source=worker)
        print('[{}]acc = {}'.format(worker, data))

    for worker in range(1, comm.Get_size()):
        reqs.append(comm.isend(Instruction.EXIT, worker))
    for req in reqs:
        req.wait()
    
else:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)

    test_graph = SimpleNet(sess)

    while True:
        data = comm.recv(source=0)
        if data == Instruction.INIT:
            test_graph.initialize_variables()
        elif data == Instruction.GET:
            comm.send(test_graph.get_accuracy(), dest=0)
        elif data == Instruction.EXIT:
            break
    