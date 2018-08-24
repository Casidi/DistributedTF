from constants import get_hp_range_definition
import sys
import random
import numpy as np
sys.path.insert(0, './resnet')

from resnet import cifar10_main
from model_base import ModelBase

class Cifar10Model(ModelBase):
    def __init__(self, cluster_id, hparams, save_base_dir):
        super(Cifar10Model, self).__init__(cluster_id, hparams, save_base_dir)

        #for debugging
        '''self.hparams = {
            'opt_case': {'lr': 0.1, 'optimizer': 'Momentum', 'momentum': 0.9},
            'decay_steps': 20,
            'decay_rate': 0.1,
            'weight_decay': 2e-4,
            'regularizer': 'l2_regularizer',
            'initializer': 'he_init',
            'batch_size': 128}'''
        
        #if self.cluster_id == 0:
        #    self.hparams['opt_case']['lr'] = 100000
    
    def train(self, num_epoch, total_epochs):
        data_dir = '/home/K8S/dataset/cifar10'
        self.accuracy, model_id = \
            cifar10_main.main(self.hparams, self.cluster_id, 
                    self.save_base_dir, data_dir, 
                    num_epoch, total_epochs, self.epoches_trained)
        self.epoches_trained += num_epoch