from toy_model import ToyModel
import unittest

class ToyModelTestCase(unittest.TestCase):
    def test_train_step(self):
        hparam = {'h_0':1.0, 'h_1': 0.0}
        model = ToyModel(0, hparam)
        model.train(10)
        self.assertEqual(model.train_step, 10)

    def test_copy_model(self):
        model1 = ToyModel(0, {'h_0':1.0, 'h_1': 0.0})
        model2 = ToyModel(1, {'h_0':0.0, 'h_1': 1.0})
        model1.train(1)
        model2.train(1)
        self.assertNotEqual(model1.get_values()[2], model2.get_values()[2])
        model1.set_values(model2.get_values())
        self.assertEqual(model1.get_values()[2], model2.get_values()[2])
        self.assertNotEqual(model1.get_values()[3], model2.get_values()[3])
        self.assertNotEqual(model1.cluster_id, model2.cluster_id)
        

unittest.main(verbosity=2)