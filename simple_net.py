import tensorflow as tf
import numpy as np

class SimpleNet:
    def __init__(self, sess, cluster_id):
        self.sess = sess
        self.cluster_id = cluster_id
        self.train_step = 0

        x_train = [[0.0, 1.0]]
        y_train = [[1.0]]
        self.input_layer = tf.constant(x_train)
        self.w1 = tf.Variable(tf.random_uniform([2, 1]))
        self.b1 = tf.Variable(tf.random_uniform([1]))
        self.output_layer = tf.sigmoid(tf.matmul(self.input_layer, self.w1) + self.b1)

        self.loss = tf.reduce_sum(tf.square(y_train - self.output_layer))
        self.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        self.vars = [self.w1, self.b1]

    def init_variables(self):
        self.sess.run([var.initializer for var in self.vars])

    def train(self, num_steps):
        for i in range(num_steps):
            self.sess.run(self.train_op)
            self.train_step += 1

    def get_loss(self):
        return self.sess.run(self.loss)

    def get_values(self):
        return [self.cluster_id] + self.sess.run([self.loss, self.w1, self.b1])

    def set_values(self, values):
        return