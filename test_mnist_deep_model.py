import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from mnist_model import MNISTModel
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

        subprocess.call(['rm', '-rf', 'savedata'])
        subprocess.call(['mkdir', 'savedata'])

    def test_basic(self):
        model = MNISTModel(0, generate_random_hparam(), 'savedata/model_')
        
    def test_perturb_hparams(self):
        model = MNISTModel(0, generate_random_hparam(), 'savedata/model_')
        model.perturb_hparams()

unittest.main(verbosity=2)

