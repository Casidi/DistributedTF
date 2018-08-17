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

        model1.train(10)
        model2.train(10)
        self.assertFalse(np.array_equal(np.asarray(model1.get_values()[2]), np.asarray(model2.get_values()[2])))

        
        model1.set_values(model2.get_values())
        
        
        _, _, model1_trainable_vars, model1_hparams = model1.get_values()
        _, _, model2_trainable_vars, model2_hparams = model2.get_values()
        # Note model1_trainable_vars is a list of lists with varying number of elements
        # In this case, np.asarray(model1_trainable_vars) returns an array of python lists, not array of arrays
        # So make an array of arrays by loop
        array_model1_trainable_vars = np.array([np.array(xi) for xi in model1_trainable_vars])
        array_model2_trainable_vars = np.array([np.array(xi) for xi in model2_trainable_vars])
        
        
        ### All the values of the two models should be the same except the cluster_id
        self.assertEqual(model1.get_accuracy(), model2.get_accuracy())
        self.assertEqual(array_model1_trainable_vars.shape, array_model2_trainable_vars.shape)
        
        # Because any two floats that should represent the same number often don't
        # And numpy.array_equal works when our array only contains integers
        # Here we should use numpy.allclose to compare 2 floats
        # Note numpy.allclose doesn't support multi-dim array
        for i in xrange(len(array_model1_trainable_vars)):
            self.assertTrue(np.allclose(array_model1_trainable_vars[i], array_model2_trainable_vars[i]),
                            msg='The weights should be the same!!')
        self.assertEqual(model1_hparams, model2_hparams,
                            msg='The hyper parameters should be the same!!')
        self.assertNotEqual(model1.cluster_id, model2.cluster_id)
        
                
load_dataset()
unittest.main(verbosity=2)

