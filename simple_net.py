import tensorflow as tf
import numpy as np

class SimpleNet:
    def __init__(self, sess, cluster_id, hparams):
        self.sess = sess
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.train_step = 0
        self.need_explore = False

        x_train = [[0.0, 1.0]]
        y_train = [[1.0]]
        self.input_layer = tf.constant(x_train)
        self.w1 = tf.Variable(tf.random_uniform([2, 1]))
        self.b1 = tf.Variable(tf.random_uniform([1]))
        self.output_layer = tf.sigmoid(tf.matmul(self.input_layer, self.w1) + self.b1)

        self.loss = tf.reduce_sum(tf.square(y_train - self.output_layer))

        if hparams['opt_case']['optimizer'] == 'gd':
            learn_rate = hparams['opt_case']['lr']
            self.optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(1.0)
        #self.optimizer = tf.train.GradientDescentOptimizer(1.0)

        self.train_op = self.optimizer.minimize(self.loss)

        self.trainable_vars = [self.w1, self.b1]

    def init_variables(self):
        self.sess.run([var.initializer for var in self.trainable_vars])

    def train(self, num_steps):
        for i in range(num_steps):
            self.sess.run(self.train_op)
            self.train_step += 1

    def perturb_hparams_and_update_graph(self):
        return

    def get_loss(self):
        return self.sess.run(self.loss)

    def get_values(self):
        return [self.cluster_id, self.get_loss(), self.sess.run(self.trainable_vars), self.hparams]

    def set_values(self, values):
        for i in range(len(self.trainable_vars)):
            self.trainable_vars[i].load(values[2][i], self.sess)
        
        self.hparams = values[3]
        if self.hparams['opt_case']['optimizer'] == 'gd':
            learn_rate = self.hparams['opt_case']['lr']
            self.optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(1.0)
        #self.optimizer = tf.train.GradientDescentOptimizer(1.0)

        self.train_op = self.optimizer.minimize(self.loss)

