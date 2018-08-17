import sys
sys.path.insert(0, './resnet')

#NOTE: tensorflow >= 1.10 required
from resnet import cifar10_main

class Cifar10Model:
    def __init__(self, cluster_id, hparams):
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.epoches_trained = 0
        self.need_explore = False

        #for debugging
        self.hparams = {
            'opt_case': {'lr': 0.1, 'optimizer': 'Momentum', 'momentum': 0.9},
            'decay_steps': 20,
            'decay_rate': 0.1,
            'weight_decay': 2e-4,
            'regularizer': 'l2_regularizer',
            'initializer': 'he_init',
            'batch_size': 128}

        self.accuracy = 0.0
    
    def train(self):
        save_base_dir = './savedata/model_'
        data_dir = '/home/K8S/dataset/cifar10'
        self.accuracy, model_id = \
            cifar10_main.main(self.hparams, self.cluster_id, save_base_dir, data_dir, 1)
        self.epoches_trained += 1
        return

    def perturb_hparams_and_update_graph(self):
        #TODO: implement this function to get the exploring working
        return

    def get_accuracy(self):
        return self.accuracy

    def get_values(self):
        return [self.cluster_id, self.get_accuracy(), self.hparams]

    def set_values(self, values):
        self.hparams = values[2]
