import toy_model 
import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class ToyModelTestCase(unittest.TestCase):
    def test_train_step(self):
        toy_model.main({"h_0": 1.0, "h_1":0.0}, 0, 'savedata/model_', '', 1)    

unittest.main(verbosity=2)