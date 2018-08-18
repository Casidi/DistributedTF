from __future__ import print_function

from constants import WorkerInstruction

from cifar10_model import Cifar10Model
from toy_model import ToyModel
from mnist_model import MNISTModel

class TrainingWorker:
    def __init__(self, comm, master_rank):
        self.worker_graphs = []
        self.is_expolore_only = False

        while True:
            data = comm.recv(source=master_rank)
            inst = data[0]
            if inst == WorkerInstruction.ADD_GRAPHS:
                hparam_list = data[1]
                cluster_id_begin = data[2]
                self.is_expolore_only = data[3]
                cluster_id_end = cluster_id_begin + len(hparam_list)
                print('[{}]Got {} hparams'.format(rank, len(hparam_list)))

                for i in range(cluster_id_begin, cluster_id_end):
                    hparam = hparam_list[i-cluster_id_begin]
                    #new_graph = ToyModel(i, hparam)
                    new_graph = MNISTModel(i, hparam)
                    #new_graph = Cifar10Model(i, hparam)
                    self.worker_graphs.append(new_graph)
            elif inst == WorkerInstruction.TRAIN:
                num_steps = data[1]
                for g in self.worker_graphs[:]:  # Take a copy of the list and then iterate over it, or the iteration will fail with unexpected results.
                    g.train(num_steps)
                    print('Graph {} epoch = {},  acc = {}'.format(g.cluster_id, g.epoches_trained, g.get_accuracy()))
                    if math.isnan(g.get_accuracy()) == True:
                        self.worker_graphs.remove(g)
                        print('[WARNING] The calculated accuracy of the graph is NaN, the program has removed the graph.')
            elif inst == WorkerInstruction.GET:
                vars_to_send = []
                for g in self.worker_graphs:
                    vars_to_send.append(g.get_values())
                comm.send(vars_to_send, dest=master_rank)
            elif inst == WorkerInstruction.SET:
                vars_to_set = data[1]
                for v in vars_to_set:
                    for g in self.worker_graphs:
                        if g.cluster_id == v[0]:
                            g.set_values(v)
                            g.need_explore = True
            elif inst == WorkerInstruction.EXPLORE:
                for g in self.worker_graphs:
                    if g.need_explore or self.is_expolore_only:
                        print('[{}]Exploring graph {}'.format(rank, g.cluster_id))
                        g.perturb_hparams_and_update_graph()
                        g.need_explore = False
            elif inst == WorkerInstruction.EXIT:
                break
            else:
                print('Invalid instruction!!!!')

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.Comm.Get_parent()
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('[{}]This is training worker!!!'.format(rank))