import sys
import os
sys.path.insert(0, './mnist_deep')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from mnist_deep import mnist_main
from constants import generate_random_hparam
import unittest
import subprocess

class MNISTDeepModelTestCase(unittest.TestCase):
    def setUp(self):
        self.hp = generate_random_hparam()
        self.hp_good = {'opt_case':{'optimizer': 'gd', 'lr':0.0001},
                            'batch_size': 256}
        self.save_base_dir = './savedata/model_'
        self.data_dir = '/home/K8S/dataset/mnist'

        subprocess.call(['rm', '-rf', './savedata/model_*'])

    def tearDown(self):
        subprocess.call(['rm', '-rf', './savedata/model_*'])

    '''def test_seperated_calls_to_main(self):
        for i in range(10):
            step, acc = mnist_main.main(self.hp, 0, self.save_base_dir, self.data_dir, 1)
            print 'Step {}, acc = {}'.format(step, acc)'''

    def test_multiple_hp_learning_curve(self):
        all_hparams = []
        for i in range(5):
            hp = generate_random_hparam()
            hp['opt_case']['lr'] /= 10000.0
            all_hparams.append(hp)
        for i in range(len(all_hparams)):
            for j in range(20):
                step, acc = mnist_main.main(all_hparams[i], i, self.save_base_dir, self.data_dir, 1)
                print 'Model {}, step={}, acc={}'.format(i, step, acc)

unittest.main(verbosity=2)

