from mnist_deep_model import MNISTDeepModel
from mnist_dataset import load_dataset
from constants import load_hp_space
import hyperopt.pyll.stochastic
import unittest
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class ToyModelTestCase(unittest.TestCase):
    def test_train_step(self):
        hp_space = load_hp_space()
        hparam = hyperopt.pyll.stochastic.sample(hp_space)

        model = MNISTDeepModel(0, hparam)
        model.train(10)
        self.assertEqual(model.train_step, 10)

    def test_copy_model(self):
        hp_space = load_hp_space()
        model1 = MNISTDeepModel(0, hyperopt.pyll.stochastic.sample(hp_space))
        model2 = MNISTDeepModel(1, hyperopt.pyll.stochastic.sample(hp_space))

        model1.train(1)
        model2.train(1)
        self.assertFalse(np.array_equal(np.asarray(model1.get_values()[2]), np.asarray(model2.get_values()[2])))

        model1.set_values(model2.get_values())

        # All the values of the two models should be the same except the cluster_id
        self.assertEqual(model1.get_accuracy(), model2.get_accuracy())
        self.assertEqual(np.asarray(model1.get_values()[2]).shape, np.asarray(model2.get_values()[2]).shape)
        self.assertTrue(np.array_equal(np.asarray(model1.get_values()[2]), np.asarray(model2.get_values()[2])))
        self.assertEqual(model1.get_values()[3], model2.get_values()[3])
        self.assertNotEqual(model1.cluster_id, model2.cluster_id)
        
load_dataset()
unittest.main(verbosity=2)

