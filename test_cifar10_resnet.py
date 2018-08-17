import sys
import os
sys.path.insert(0, './resnet')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#NOTE: tensorflow >= 1.10 required
from resnet import cifar10_main
import unittest
import subprocess

class Cifar10ModelTestCase(unittest.TestCase):
    def setUp(self):
        self.hp = {
            'opt_case': {'lr': 0.1, 'optimizer': 'Momentum', 'momentum': 0.9},
            'decay_steps': 20,
            'decay_rate': 0.1,
            'weight_decay': 2e-4,
            'regularizer': 'l2_regularizer',
            'initializer': 'he_init',
            'batch_size': 128}
        self.save_base_dir = './resnet/model_'
        self.data_dir = '/home/K8S/dataset/cifar10'

        subprocess.call(['rm', '-rf', './resnet/model_*'])

    def tearDown(self):
        subprocess.call(['rm', '-rf', './resnet/model_*'])

    def test_seperate_calls_to_main(self):
        total_epoches = 20
        for i in range(total_epoches):
            eval_accuracy, model_id = cifar10_main.main(self.hp, 0, self.save_base_dir, self.data_dir, 1)
            print 'Epoch {}, acc = {}'.format(i+1, eval_accuracy)

        eval_accuracy, model_id = cifar10_main.main(self.hp, 1, self.save_base_dir, self.data_dir, total_epoches)


unittest.main(verbosity=2)