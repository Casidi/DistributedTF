import tensorflow as tf
import random

class ToyModel:
    def __init__(self, sess, cluster_id, hparams):
        self.sess = sess
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.train_step = 0
        self.need_explore = False

        self.perturb_factors = [0.8, 1.2]
        self.lr = 0.02  

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
        #self.loss = tf.square(self.theta_0) + tf.square(self.theta_1)
        self.fake_loss = tf.square(self.theta_0) + tf.square(self.theta_1)

        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

        self.trainable_vars = [self.theta_0, self.theta_1]
        self.train_log = []

    def init_variables(self):
        self.sess.run([var.initializer for var in self.trainable_vars])

    def train(self, num_steps):
        for i in range(num_steps):
            self.train_log.append(self.sess.run(self.trainable_vars))
            self.sess.run(self.train_op)
            self.train_step += 1

    def _perturb_float(self, val, limit_min, limit_max):
            # Noted, some hp value can't exceed reasonable range
            float_string = str(limit_min)
            if 'e' in float_string:
                _, n_digits = float_string.split('e')
                if '-' in n_digits:
                    n_digits = int(n_digits)*-1
                else:
                    n_digits = int(n_digits)
            else:
                n_digits = str(limit_min)[::-1].find('.')
            min = val * self.perturb_factors[0]
            max = val * self.perturb_factors[1]
            if min < limit_min:
                min = limit_min
                n_digits += 1
            if max > limit_max:
                max = limit_max
            val = random.uniform(min, max)
            val = round(val, n_digits)
            
            return val

    def perturb_hparams_and_update_graph(self):
        self.hparams['h_0'] = self._perturb_float(self.hparams['h_0'], 0.0, 1.0)
        self.hparams['h_1'] = self._perturb_float(self.hparams['h_1'], 0.0, 1.0)

        self.surrogate_obj = 1.2 - (self.hparams['h_0']*tf.square(self.theta_0) + self.hparams['h_1']*tf.square(self.theta_1))
        self.obj = 1.2 - (tf.square(self.theta_0) + tf.square(self.theta_1))
        self.loss = tf.square((self.obj - self.surrogate_obj))
        #self.loss = tf.square(self.theta_0) + tf.square(self.theta_1)
        self.fake_loss = tf.square(self.theta_0) + tf.square(self.theta_1)

        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

        #NOTE: We don't reload the variables here, but it seems working.
        self.trainable_vars = [self.theta_0, self.theta_1]

    #the loss is not the same as the loss to compute the gradients
    #this may be a bug of the paper
    def get_loss(self):
        return self.sess.run(self.fake_loss)

    def get_values(self):
        return [self.cluster_id, self.get_loss(), self.sess.run(self.trainable_vars), self.hparams]

    def set_values(self, values):
        for i in range(len(self.trainable_vars)):
            self.trainable_vars[i].load(values[2][i], self.sess)
        
        '''self.hparams = values[3]
        self.surrogate_obj = 1.2 - (self.hparams['h_0']*tf.square(self.theta_0) + self.hparams['h_1']*tf.square(self.theta_1))
        self.obj = 1.2 - (tf.square(self.theta_0) + tf.square(self.theta_1))
        self.loss = tf.square((self.obj - self.surrogate_obj))

        self.optimizer = tf.train.GradientDescentOptimizer(0.02)
        self.train_op = self.optimizer.minimize(self.loss)'''

        