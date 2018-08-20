from __future__ import print_function
import math

from constants import WorkerInstruction

from cifar10_model import Cifar10Model
from toy_model import ToyModel
from mnist_model import MNISTModel

class TrainingWorker:
    def __init__(self, comm, master_rank):
        self.worker_graphs = []
        self.is_expolore_only = False
        self.rank = comm.Get_rank()
        self.comm = comm
        self.master_rank = master_rank

    def main_loop(self):
        while True:
            data = self.comm.recv(source=self.master_rank)
            inst = data[0]
            if inst == WorkerInstruction.ADD_GRAPHS:
                hparam_list = data[1]
                cluster_id_begin = data[2]
                self.is_expolore_only = data[3]
                self.add_graphs(hparam_list, cluster_id_begin)
            elif inst == WorkerInstruction.TRAIN:
                num_steps = data[1]
                self.train(num_steps)
            elif inst == WorkerInstruction.GET:
                self.comm.send(self.get_all_values(), dest=self.master_rank)
            elif inst == WorkerInstruction.SET:
                vars_to_set = data[1]
                self.set_values(vars_to_set)
            elif inst == WorkerInstruction.EXPLORE:
                self.explore_necessary_graphs()
            elif inst == WorkerInstruction.EXIT:
                break
            else:
                print('Invalid instruction!!!!')

    def add_graphs(self, hparam_list, id_begin):
        cluster_id_end = id_begin + len(hparam_list)
        print('[{}]Got {} hparams'.format(self.rank, len(hparam_list)))

        for i in range(id_begin, cluster_id_end):
            hparam = hparam_list[i - id_begin]
            #new_graph = ToyModel(i, hparam, './savedata/model_')
            new_graph = MNISTModel(i, hparam, './savedata/model_')
            #new_graph = Cifar10Model(i, hparam, './savedata/model_')
            self.worker_graphs.append(new_graph)

    def train(self, num_epoches):
        for g in self.worker_graphs:
            g.train(num_epoches)
            print('Graph {} epoch = {},  acc = {}'.format(g.cluster_id, g.epoches_trained, g.get_accuracy()))
            if math.isnan(g.get_accuracy()) == True:
                self.worker_graphs.remove(g)
                print('[WARNING] The calculated accuracy of the graph is NaN, the program has removed the graph.')

    def get_all_values(self):
        vars_to_send = []
        for g in self.worker_graphs:
            vars_to_send.append(g.get_values())
        return vars_to_send

    def set_values(self, values_to_set):
        for v in values_to_set:
            for g in self.worker_graphs:
                if g.cluster_id == v[0]:
                    g.set_values(v)
                    g.need_explore = True

    def explore_necessary_graphs(self):
        for g in self.worker_graphs:
            if g.need_explore or self.is_expolore_only:
                print('[{}]Exploring graph {}'.format(self.rank, g.cluster_id))
                g.perturb_hparams()
                g.need_explore = False