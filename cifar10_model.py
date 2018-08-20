import sys
sys.path.insert(0, './resnet')

from resnet import cifar10_main

class Cifar10Model:
    def __init__(self, cluster_id, hparams, save_base_dir):
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.save_base_dir = save_base_dir
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
    
    def train(self, num_epoch):
        data_dir = '/home/K8S/dataset/cifar10'
        self.accuracy, model_id = \
            cifar10_main.main(self.hparams, self.cluster_id, self.save_base_dir, data_dir, num_epoch)
        self.epoches_trained += num_epoch
        return

    def perturb_hparams(self):
        #TODO: implement this function to get the exploring working
        return

    def get_accuracy(self):
        return self.accuracy

    def get_values(self):
        return [self.cluster_id, self.get_accuracy(), self.hparams]

    def set_values(self, values):
        self.hparams = values[2]
