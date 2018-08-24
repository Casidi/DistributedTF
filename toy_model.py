import tensorflow as tf
import numpy as np
import random
import os
import csv

def main(hp, model_id, save_base_dir, data_dir, train_epochs):
    save_dir = save_base_dir + str(model_id)

    tf.reset_default_graph()
    theta_0 = tf.Variable(0.9)
    theta_1 = tf.Variable(0.9)
    surrogate_obj = 1.2 - (hp['h_0']*tf.square(theta_0) + hp['h_1']*tf.square(theta_1))
    obj = 1.2 - (tf.square(theta_0) + tf.square(theta_1))
    
    global_step = tf.train.get_or_create_global_step()
    loss = tf.square((obj - surrogate_obj))
    optimizer = tf.train.GradientDescentOptimizer(0.02)
    train_op = optimizer.minimize(loss, global_step=global_step)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if os.path.isdir(save_dir):
            saver.restore(sess, os.path.join(save_dir, "model.ckpt"))

        results_to_log = []
        for i in range(train_epochs):
            results_to_log.append(sess.run([theta_0, theta_1, global_step, obj]) 
                                    + [hp['opt_case']['optimizer'], hp['opt_case']['lr']])
            sess.run(train_op)

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        saver.save(sess, os.path.join(save_dir, "model.ckpt"))

        filename = os.path.join(save_dir,'theta.csv')
        file_exists = os.path.isfile(filename)
        fields=['theta_0','theta_1']
        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            if not file_exists:
                writer.writeheader()

            for i in results_to_log:
                writer.writerow({'theta_0': i[0], 'theta_1': i[1]})

        filename = os.path.join(save_dir,'learning_curve.csv')
        file_exists = os.path.isfile(filename)
        fields=['global_step','accuracy', 'optimizer', 'lr']
        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            if not file_exists:
                writer.writeheader()

            for i in results_to_log:
                writer.writerow({'global_step': i[2], 'accuracy': i[3], 'optimizer':i[4], 'lr': i[5]})
        
        return sess.run([global_step, obj])

class ToyModel:
    def __init__(self, cluster_id, hparams, save_base_dir):
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.save_base_dir = save_base_dir
        self.epoches_trained = 0
        self.need_explore = False

        self._perturb_factors = [0.8, 1.2]

        if isinstance(self.hparams['batch_size'], np.ndarray):
            self.hparams['batch_size'] = self.hparams['batch_size'].item()

        if cluster_id == 0:
            self.hparams['h_0'] = 0.0
            self.hparams['h_1'] = 1.0
        else:
            self.hparams['h_0'] = 1.0
            self.hparams['h_1'] = 0.0

        self.accuracy = 0.0
    
    def train(self, epoches_to_train, total_epochs):
        data_dir = ''
        step, self.accuracy = \
            main(self.hparams, self.cluster_id, self.save_base_dir, data_dir, epoches_to_train)
        self.epoches_trained += epoches_to_train

    def perturb_hparams(self):
        self.hparams['h_0'] = self._perturb_float(self.hparams['h_0'], 0.0, 1.0)
        self.hparams['h_1'] = self._perturb_float(self.hparams['h_1'], 0.0, 1.0)

    def get_accuracy(self):
        return self.accuracy

    def get_values(self):
        return [self.cluster_id, self.get_accuracy(), self.hparams]

    #overwrite the copying of hparam
    def set_values(self, values):
        '''if self.cluster_id == 0:
            self.hparams['h_0'] = 0.0
            self.hparams['h_1'] = 1.0
        else:
            self.hparams['h_0'] = 1.0
            self.hparams['h_1'] = 0.0'''
        self.hparams = values[2]

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
        min = val * self._perturb_factors[0]
        max = val * self._perturb_factors[1]
        if min < limit_min:
            min = limit_min
            n_digits += 1
        if max > limit_max:
            max = limit_max
        val = random.uniform(min, max)
        val = round(val, n_digits)
        
        return val