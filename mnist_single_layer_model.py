import tensorflow as tf
from mnist_dataset import get_train_batch, get_test_data

class MNISTSingleLayerModel:
    def __init__(self, cluster_id, hparams):
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.train_step = 0
        self.need_explore = False

        self.build_graph_from_hparams()
        self.train_log = []