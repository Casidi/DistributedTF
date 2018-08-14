import tensorflow as tf
import random

class ToyModel:
    def __init__(self, cluster_id, hparams):
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.train_step = 0
        self.need_explore = False

        self.perturb_factors = [0.8, 1.2]
        self.lr = 0.02

        if cluster_id == 0:
            self.hparams['h_0'] = 0.0
            self.hparams['h_1'] = 1.0
        else:
            self.hparams['h_0'] = 1.0
            #self.hparams['h_1'] = 0.0
            self.hparams['h_1'] = float('nan')  # Use for testing NaN error handling

        self.build_graph_from_hparams(is_first_call=True)
        self.train_log = []

    def train(self, num_steps):
        for i in range(num_steps):
            self.train_log.append(self.sess.run(self.trainable_vars))
            self.sess.run(self.train_op)
            self.train_step += 1    

    def perturb_hparams_and_update_graph(self):
        self.hparams['h_0'] = self._perturb_float(self.hparams['h_0'], 0.0, 1.0)
        self.hparams['h_1'] = self._perturb_float(self.hparams['h_1'], 0.0, 1.0)
        self.build_graph_from_hparams(is_first_call=False)

    def get_accuracy(self):
        return self.sess.run(self.obj)

    def get_values(self):
        return [self.cluster_id, self.get_accuracy(), self.sess.run(self.trainable_vars), self.hparams]

    def set_values(self, values):
        for i in range(len(self.trainable_vars)):
            self.trainable_vars[i].load(values[2][i], self.sess)
        
        # Skip the copying of hyper-parameters to obtain the result of the paper
        '''self.hparams = values[3]
        self.surrogate_obj = 1.2 - (self.hparams['h_0']*tf.square(self.theta_0) + self.hparams['h_1']*tf.square(self.theta_1))
        self.obj = 1.2 - (tf.square(self.theta_0) + tf.square(self.theta_1))
        self.loss = tf.square((self.obj - self.surrogate_obj))

        self.optimizer = tf.train.GradientDescentOptimizer(0.02)
        self.train_op = self.optimizer.minimize(self.loss)'''
    
    def build_graph_from_hparams(self, is_first_call):
        if not is_first_call:
            old_values = self.sess.run(self.trainable_vars)

        self.tf_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(graph=self.tf_graph, config=config)
        
        with self.tf_graph.as_default():
            self.theta_0 = tf.Variable(0.9)
            self.theta_1 = tf.Variable(0.9)
            self.surrogate_obj = 1.2 - (self.hparams['h_0']*tf.square(self.theta_0) + self.hparams['h_1']*tf.square(self.theta_1))
            self.obj = 1.2 - (tf.square(self.theta_0) + tf.square(self.theta_1))
            self.loss = tf.square((self.obj - self.surrogate_obj))

            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss)

            self.trainable_vars = [self.theta_0, self.theta_1]
        self.sess.run([var.initializer for var in self.trainable_vars])

        if not is_first_call:
            for i in range(len(self.trainable_vars)):
                self.trainable_vars[i].load(old_values[i], self.sess)

    def _perturb_float(self, val, limit_min, limit_max):
            #NOTE: some hp value can't exceed reasonable range
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