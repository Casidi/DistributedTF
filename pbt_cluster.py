import math

import hyperopt.pyll.stochastic
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np

from constants import WorkerInstruction
from constants import get_hp_range_definition, load_hp_space

class PBTCluster:
    def __init__(self, pop_size, comm, master_rank, do_exploit=True, do_explore=True):
        self.pop_size = pop_size
        self.comm = comm
        self.master_rank = master_rank
        self.do_exploit = do_exploit
        self.do_explore = do_explore

        self.build_all_graphs()

    def build_all_graphs(self):
        all_hparams_need_training = []
        hp_space = load_hp_space()
        for i in range(self.pop_size):         
            hparams = hyperopt.pyll.stochastic.sample(hp_space)
            all_hparams_need_training.append(hparams)

        print('Population size = {}'.format(self.pop_size))
        graphs_per_worker = math.ceil(float(self.pop_size) / float((self.comm.Get_size() - 1)))
        graphs_to_make = len(all_hparams_need_training)

        reqs = []
        num_workers_sent = 0
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                begin = num_workers_sent * graphs_per_worker
                end = min(graphs_per_worker, graphs_to_make) + begin
                hparams_for_the_worker = all_hparams_need_training[begin: end]
                reqs.append(self.comm.isend((WorkerInstruction.ADD_GRAPHS, hparams_for_the_worker, begin), dest=i))
                graphs_to_make -= graphs_per_worker
                num_workers_sent += 1
        for req in reqs:
            req.wait()

    def kill_all_workers(self):
        reqs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.EXIT, ), dest=i))
        for req in reqs:
            req.wait()

    def train(self, until_step_num):
        round = 0
        while until_step_num > 0:
            print '\nRound {}'.format(round)
            steps_to_train = min(4, until_step_num)

            reqs = []
            for i in range(0, self.comm.Get_size()):
                if i != self.master_rank:
                    reqs.append(self.comm.isend((WorkerInstruction.TRAIN, steps_to_train), dest=i))
            for req in reqs:
                req.wait()

            until_step_num -= 4
            round += 1

            if self.do_exploit:
                self.exploit()
            if self.do_explore:
                self.explore()

    def exploit(self):
        reqs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.GET, ), dest=i))
        for req in reqs:
            req.wait()

        all_values = []
        cluster_id_to_worker_rank = {}
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                data = self.comm.recv(source=i)
                all_values += data
                for d in data:
                    cluster_id_to_worker_rank[d[0]] = i

        # copy top 25% to bottom 25%
        all_values = sorted(all_values, key=lambda value: value[1])
        '''print 'The ranking before exploit'
        for i in all_values:
            print 'graph {}, loss={}'.format(i[0], i[1])'''
        num_graphs_to_copy = math.ceil(float(self.pop_size) / 4.0)
        graphs_need_updating = []
        for i in range(num_graphs_to_copy):
            bottom_index = i
            top_index = len(all_values) - num_graphs_to_copy + i
            all_values[bottom_index][1] = all_values[top_index][1] #copy loss, not necessary
            all_values[bottom_index][2] = all_values[top_index][2] #copy trainable variables
            all_values[bottom_index][3] = all_values[top_index][3] #copy hparams
            graphs_need_updating.append(bottom_index)

        #only update the bottom graphs
        worker_rank_to_graphs_need_updating = {}
        for i in range(self.comm.Get_size()):
            worker_rank_to_graphs_need_updating[i] = []
        for i in graphs_need_updating:
            worker_rank = cluster_id_to_worker_rank[all_values[i][0]]
            worker_rank_to_graphs_need_updating[worker_rank].append(all_values[i])

        reqs = []
        for worker_rank, values in worker_rank_to_graphs_need_updating.iteritems():
            reqs.append(self.comm.isend((WorkerInstruction.SET, values), dest=worker_rank))
        for req in reqs:
            req.wait()

    def explore(self):
        reqs = []
        for i in range(self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.EXPLORE, ), dest=i))
        for req in reqs:
            req.wait()

    def report_plot_for_toy_model(self):
        training_log = self.get_all_training_log()

        linspace_x = np.linspace(start=0, stop=1, num=100)
        linspace_y = np.linspace(start=0, stop=1, num=100)
        x, y = np.meshgrid(linspace_x, linspace_y)
        z = 1.2 - (x ** 2 + y ** 2)
        
        pyplot.xlabel(r'$\theta_0$')
        pyplot.ylabel(r'$\theta_1$')
        pyplot.xlim(0, 1)
        pyplot.ylim(0, 1)
        
        pyplot.plot(zip(*training_log[0])[0], zip(*training_log[0])[1], '.', color='black')
        pyplot.plot(zip(*training_log[1])[0], zip(*training_log[1])[1], '.', color='red')
        pyplot.contour(x, y, z, colors='lightgray')
        #pyplot.show()

        if self.do_exploit and self.do_explore:
            out_file_name = 'PBT.png'
        elif self.do_exploit and not self.do_explore:
            out_file_name = 'exploit_only.png'
        elif not self.do_exploit and self.do_explore:
            out_file_name = 'explore_only.png'
        else:
            out_file_name = 'grid_search.png'
        pyplot.savefig(out_file_name)
        print 'Writing results to {}'.format(out_file_name)

    def get_all_training_log(self):
        reqs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.GET_TRAIN_LOG, ), dest=i))
        for req in reqs:
            req.wait()

        all_logs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                data = self.comm.recv(source=i)
                all_logs += data
        return all_logs
