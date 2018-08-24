import tensorflow as tf
import os
import csv

from model_base import ModelBase

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

class ToyModel(ModelBase):
    def __init__(self, cluster_id, hparams, save_base_dir):
        super(ToyModel, self).__init__(cluster_id, hparams, save_base_dir)

        if cluster_id == 0:
            self.hparams['h_0'] = 0.0
            self.hparams['h_1'] = 1.0
        else:
            self.hparams['h_0'] = 1.0
            self.hparams['h_1'] = 0.0
    
    def train(self, epoches_to_train, total_epochs):
        data_dir = ''
        step, self.accuracy = \
            main(self.hparams, self.cluster_id, self.save_base_dir, data_dir, epoches_to_train)
        self.epoches_trained += epoches_to_train

    #overwrite the copying of hparam
    def set_values(self, values):
        if self.cluster_id == 0:
            self.hparams['h_0'] = 0.0
            self.hparams['h_1'] = 1.0
        else:
            self.hparams['h_0'] = 1.0
            self.hparams['h_1'] = 0.0