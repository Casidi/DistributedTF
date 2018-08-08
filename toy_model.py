import tensorflow as tf

class ToyModel:
    def __init__(self, sess, cluster_id, hparams):
        self.sess = sess
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.train_step = 0
        self.need_explore = False

        self.theta_0 = tf.Variable(0.9)
        self.theta_1 = tf.Variable(0.9)

        if cluster_id == 0:
            self.hparams['h_0'] = 0.0
            self.hparams['h_1'] = 1.0
        else:
            self.hparams['h_0'] = 1.0
            self.hparams['h_1'] = 0.0

        self.surrogate_obj = 1.2 - (self.hparams['h_0']*tf.square(self.theta_0) + self.hparams['h_1']*tf.square(self.theta_1))
        self.obj = 1.2 - (tf.square(self.theta_0) + tf.square(self.theta_1))
        self.loss = tf.square((self.obj - self.surrogate_obj))

        self.optimizer = tf.train.GradientDescentOptimizer(0.02)
        self.train_op = self.optimizer.minimize(self.loss)

        self.trainable_vars = [self.theta_0, self.theta_1]
        self.train_log = []

    def init_variables(self):
        self.sess.run([var.initializer for var in self.trainable_vars])

    def train(self, num_steps):
        for i in range(num_steps):
            self.sess.run(self.train_op)
            self.train_log.append(self.sess.run(self.trainable_vars))
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
        print 'loss after set = {} val_exp={} val_real={}'.format(self.sess.run([self.loss, self.obj, self.surrogate_obj]), values[2], self.sess.run(self.trainable_vars))
        
        '''self.hparams = values[3]
        self.surrogate_obj = 1.2 - (self.hparams['h_0']*tf.square(self.theta_0) + self.hparams['h_1']*tf.square(self.theta_1))
        self.obj = 1.2 - (tf.square(self.theta_0) + tf.square(self.theta_1))
        self.loss = tf.square((self.obj - self.surrogate_obj))

        self.optimizer = tf.train.GradientDescentOptimizer(0.02)
        self.train_op = self.optimizer.minimize(self.loss)'''

        