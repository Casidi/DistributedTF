from __future__ import print_function

import toy_model
from toy_model import ToyModel 
import unittest
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class ToyModelTestCase(unittest.TestCase):
    def setUp(self):
        if os.path.isdir('savedata'):
            shutil.rmtree('savedata', ignore_errors=True)
        os.mkdir('savedata')

    def test_basic(self):
        toy_model.main({"h_0": 1.0, "h_1":0.0}, 0, 'savedata/model_', '', 1)

    def test_model_class_init(self):
        model = ToyModel(0, {"h_0": 1.0, "h_1":0.0})

    def test_model_class_train(self):
        model = ToyModel(0, {"h_0": 1.0, "h_1":0.0})
        acc1 = model.get_accuracy()
        epoch1 = model.epoches_trained
        model.train(1)
        acc2 = model.get_accuracy()
        epoch2 = model.epoches_trained
        self.assertNotEqual(acc1, acc2)
        self.assertEqual(epoch1+1, epoch2)

    def test_save_load(self):
        step1, acc = toy_model.main({"h_0": 1.0, "h_1":0.0}, 0, 'savedata/model_', '', 10)
        step2, acc = toy_model.main({"h_0": 1.0, "h_1":0.0}, 0, 'savedata/model_', '', 10)
        step3, acc = toy_model.main({"h_0": 1.0, "h_1":0.0}, 1, 'savedata/model_', '', 10)
        self.assertEqual(step1, 10)
        self.assertEqual(step2, 20)
        self.assertEqual(step3, 10)

        if os.path.isdir('savedata'):
            shutil.rmtree('savedata', ignore_errors=True)
        os.mkdir('savedata')
        step4, acc = toy_model.main({"h_0": 1.0, "h_1":0.0}, 0, 'savedata/model_', '', 10)
        self.assertEqual(step4, 10)

unittest.main(verbosity=2)