"""
A convolutional neural network for MNIST that is compatible with
population-based training.
"""

from typing import Any, List, Tuple, Callable
import math
import random
import os
import tensorflow as tf
from pbt import Hyperparameter, HyperparamsUpdate, HyperparamsGraph
from mnist import ConvNet as MNISTConvNet, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE, MNIST_TEST_BATCH_SIZE,\
    get_mnist_data


class FloatHyperparameter(Hyperparameter):
    """
    A type of Hyperparameter with a single floating-point value.
    """
    def _limited(self, value):
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        return value

    def __init__(self, name, graph, unused,
                 value_setter, factor,
                 min_value, max_value):
        """
        Creates a new FloatHyperparameter of graph <graph> with descriptive
        name <name> and initial unused status <unused>.

        <value_setter> is a Callable that samples and returns an initial value.
        <factor> is the factor by which the value will be randomly multiplied
        or divided when perturbed. <min_value> is the minimum possible value,
        or None if there should be none. <max_value> is the maximum possible
        value, or None if there should be none.
        """
        super().__init__(name, graph, unused)
        self.value_setter = value_setter
        self.factor = factor
        self.min_value = min_value
        self.max_value = max_value
        self.value = tf.Variable(self._limited(value_setter()), trainable=False)

    def __str__(self):
        return str(self.get_value())

    def initialize_variables(self):
        self.graph.sess.run(self.value.initializer)

    def get_value(self):
        return self.graph.sess.run(self.value)

    def set_value(self, value):
        self.value.load(value, self.graph.sess)

    def perturb(self):
        value = self.get_value()
        if random.random() < 0.5:
            value *= self.factor
        else:
            value /= self.factor
        self.set_value(self._limited(value))

    def resample(self):
        self.set_value(self._limited(self.value_setter()))


class OptimizerInfo:
    """
    Stores a TensorFlow Optimizer and information about it.
    """
    def __init__(self, optimizer,
                 to_minimize, hyperparams):
        """
        Creates a new OptimizerInfo for <optimizer>.

        <to_minimize> is a TensorFlow Tensor that <optimizer> should be used to
        minimize, and <hyperparams> is a list of all of the Hyperparameters
        that affect <optimizer>'s behavior.
        """
        self.optimizer = optimizer
        self.minimizer = optimizer.minimize(to_minimize)
        self.vars = optimizer.variables()
        self.hyperparams = hyperparams


class OptimizerHyperparameter(Hyperparameter):
    """
    A Hyperparameter whose value is one of several TensorFlow Optimizers.
    """
    def _set_sub_hyperparams_unused(self, unused):
        for hyperparam in self.opt_info[self.opt_index].hyperparams:
            hyperparam.unused = unused

    def __init__(self, graph, to_minimize):
        """
        Creates a new OptimizerHyperparameter of <graph> with Optimizers that
        can be used to minimize the TensorFlow Tensor <to_minimize>.
        """
        super().__init__('Optimizer', graph, False)
        self.opt_info = []
        learning_rate = FloatHyperparameter('Learning rate', self.graph, True,
                                            lambda: 10 ** random.uniform(-6, 0), 1.2, 10 ** -6, 1)
        # GradientDescentOptimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate.value)
        self.opt_info.append(OptimizerInfo(optimizer, to_minimize, [learning_rate]))
        # AdagradOptimizer
        optimizer = tf.train.AdagradOptimizer(learning_rate.value, 0.01)
        self.opt_info.append(OptimizerInfo(optimizer, to_minimize, [learning_rate]))
        # MomentumOptimizer
        momentum = FloatHyperparameter('Momentum', self.graph, True,
                                       lambda: random.uniform(0, 1), 1.2, 0, 1)
        optimizer = tf.train.MomentumOptimizer(learning_rate.value, momentum.value)
        self.opt_info.append(OptimizerInfo(optimizer, to_minimize, [learning_rate, momentum]))
        # AdamOptimizer
        optimizer = tf.train.AdamOptimizer(learning_rate.value)
        self.opt_info.append(OptimizerInfo(optimizer, to_minimize, [learning_rate]))
        self.opt_index = random.randrange(len(self.opt_info))
        self._set_sub_hyperparams_unused(False)

    def __str__(self):
        return self.opt_info[self.opt_index].optimizer.__class__.__name__

    def initialize_variables(self):
        self.graph.sess.run([var.initializer for info in self.opt_info for var in info.vars])

    def get_value(self):
        return (self.opt_index, self.graph.sess.run(self.opt_info[self.opt_index].vars))

    def set_value(self, value):
        opt_index, var_values = value
        self._set_sub_hyperparams_unused(True)
        self.opt_index = opt_index
        vars = self.opt_info[opt_index].vars
        for i in range(len(vars)):
            vars[i].load(var_values[i], self.graph.sess)
        self._set_sub_hyperparams_unused(False)

    def _switch_to_opt(self, opt_index):
        self._set_sub_hyperparams_unused(True)
        self.opt_index = opt_index
        info = self.opt_info[self.opt_index]
        self.graph.sess.run([var.initializer for var in info.vars])
        for hyperparam in info.hyperparams:
            hyperparam.resample()
            hyperparam.unused = False

    def perturb(self):
        num_opts = len(self.opt_info)
        if num_opts >= 2:
            self._switch_to_opt((self.opt_index + random.randrange(1, num_opts)) % num_opts)

    def resample(self):
        self._switch_to_opt(random.randrange(len(self.opt_info)))

    def get_current_minimizer(self):
        """
        Returns a TensorFlow Operation that uses this OptimizerHyperparameter's
        current Optimizer to minimize the Tensor specified in its initializer.
        """
        return self.opt_info[self.opt_index].minimizer


class ConvNet(HyperparamsGraph):
    """
    A PBT-compatible version of an MNIST convnet that trains itself to minimize
    cross entropy with a variable optimizer, optimizer parameters, and dropout
    keep probability.
    """
    def __init__(self, num, sess):
        """
        Creates a new ConvNet, numbered <num> in its population, with
        associated Session <sess>.

        This method uses mnist.get_mnist_data() to obtain this ConvNet's
        training and testing data. Thus, mnist.set_mnist_data() must be called
        before any ConvNets are initialized.
        """
        super().__init__(num, sess)
        train_data, test_data = get_mnist_data()
        self.train_next = train_data\
            .shuffle(MNIST_TRAIN_SIZE).batch(50).repeat().make_one_shot_iterator().get_next()
        self.test_iterator = test_data.batch(MNIST_TEST_BATCH_SIZE).make_initializable_iterator()
        self.test_next = self.test_iterator.get_next()
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.int32, [None])
        one_hot_y_ = tf.one_hot(self.y_, 10)
        self.keep_prob = FloatHyperparameter('Keep probability', self, False,
                                             lambda: random.uniform(0.1, 1), 1.2, 0.1, 1)
        self.net = MNISTConvNet(sess, self.x, one_hot_y_, self.keep_prob.value)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y_, logits=self.net.y))
        self.optimizer = OptimizerHyperparameter(self, cross_entropy)
        self.accuracy = None

    def initialize_variables(self):
        super().initialize_variables()
        self.net.initialize_variables()

    def get_value(self):
        return (self.step_num, self.sess.run(self.net.vars),
                [hyperparam.get_value() for hyperparam in self.hyperparams],
                self.last_update, self.accuracy)

    def set_value(self, value):
        step_num, var_values, hyperparam_values, last_update, accuracy = value
        self.step_num = step_num
        for i in range(len(self.net.vars)):
            self.net.vars[i].load(var_values[i], self.sess)
        for i in range(len(self.hyperparams)):
            self.hyperparams[i].set_value(hyperparam_values[i])
        self.last_update = last_update
        self.accuracy = accuracy

    def get_accuracy(self):
        """
        Returns this ConvNet's accuracy score on its testing Dataset.
        """
        if self.accuracy is None:
            self.sess.run(self.test_iterator.initializer)
            size_accuracy = 0
            try:
                while True:
                    test_images, test_labels = self.sess.run(self.test_next)
                    batch_size = test_images.shape[0]
                    batch_accuracy = self.sess.run(self.net.accuracy,
                                                   feed_dict={self.x: test_images, self.y_: test_labels,
                                                              self.keep_prob.value: 1})
                    size_accuracy += batch_size * batch_accuracy
            except tf.errors.OutOfRangeError:
                pass
            self.accuracy = size_accuracy / MNIST_TEST_SIZE
        return self.accuracy

    def get_metric(self):
        return self.get_accuracy()

    def _train_step(self):
        train_images, train_labels = self.sess.run(self.train_next)
        self.sess.run(self.optimizer.get_current_minimizer(),
                      feed_dict={self.x: train_images, self.y_: train_labels})
        self.accuracy = None
        self.step_num += 1

    def train(self):
        while True:
            self._train_step()
            if self.step_num % 500 == 0:
                break

    def explore(self):
        """
        Randomly perturbs some of this ConvNet's hyperparameters.
        """
        # Ensure that at least one used hyperparameter is perturbed
        rand = random.randrange(1, 2 ** sum(1 for hyperparam in self.hyperparams if not hyperparam.unused))
        perturbed_used_hyperparam = False
        for i in range(len(self.hyperparams)):
            hyperparam = self.hyperparams[i]
            if perturbed_used_hyperparam or hyperparam.unused:
                if random.random() < 0.5:
                    hyperparam.perturb()
            elif rand & (2 ** i) != 0:
                hyperparam.perturb()
                perturbed_used_hyperparam = True
        self.record_update()