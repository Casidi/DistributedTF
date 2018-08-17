import sys
sys.path.insert(0, './mnist_deep')

#NOTE: tensorflow >= 1.10 required
from mnist_deep import mnist_main

class MNISTModel:
    def __init__(self, cluster_id, hparams):
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.epoches_trained = 0
        self.need_explore = False

        self.accuracy = 0.0
    
    def train(self):
        save_base_dir = './savedata/model_'
        data_dir = '/home/K8S/dataset/mnist'
        step, self.accuracy = \
            mnist_main.main(self.hparams, self.cluster_id, save_base_dir, data_dir, 1)
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
