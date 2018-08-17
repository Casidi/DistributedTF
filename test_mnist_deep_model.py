import sys
import os
sys.path.insert(0, './mnist_deep')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from mnist_deep import mnist_main
import unittest
import subprocess

class ToyModelTestCase(unittest.TestCase):
    def setUp(self):
        self.hp = {
            'opt_case': {'lr': 0.1, 'optimizer': 'Momentum', 'momentum': 0.9},
            'decay_steps': 20,
            'decay_rate': 0.1,
            'weight_decay': 2e-4,
            'regularizer': 'l2_regularizer',
            'initializer': 'he_init',
            'batch_size': 128}
        self.save_base_dir = './savedata/model_'
        self.data_dir = '/home/K8S/dataset/cifar10'

        subprocess.call(['rm', '-rf', './savedata/model_*'])

    def tearDown(self):
        subprocess.call(['rm', '-rf', './savedata/model_*'])

    def test_availability(self):
        mnist_main.main(self.hp, 0, self.save_base_dir, self.data_dir, 1)

unittest.main(verbosity=2)

