from hyperopt import hp
import hyperopt.pyll.stochastic
import math

from matplotlib import pyplot
import numpy as np

from constants import WorkerInstruction

class PBTCluster:
    def __init__(self, pop_size, comm, master_rank):
        self.pop_size = pop_size
        self.comm = comm
        self.master_rank = master_rank

        self.build_all_graphs()
        self.initialize_all_graphs()

    def build_all_graphs(self):
        #load HPs and send to workers
        all_hparams_need_training = []
        hp_space = self.load_hp_space()
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
                print begin, end
                hparams_for_the_worker = all_hparams_need_training[begin: end]
                reqs.append(self.comm.isend((WorkerInstruction.ADD_GRAPHS, hparams_for_the_worker, begin), dest=i))
                graphs_to_make -= graphs_per_worker
                num_workers_sent += 1
        for req in reqs:
            req.wait()

    def initialize_all_graphs(self):
        reqs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.INIT, ), dest=i))
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
        reqs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.TRAIN, until_step_num), dest=i))
        for req in reqs:
            req.wait()

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
        print 'The ranking before exploit'
        for i in all_values:
            print 'graph {}, loss={}'.format(i[0], i[1])
        num_graphs_to_copy = math.ceil(float(self.pop_size) / 4.0)
        graphs_need_updating = []
        for i in range(num_graphs_to_copy):
            top_index = i
            bottom_index = len(all_values) - num_graphs_to_copy + i
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

    def report_plot(self):
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
        pyplot.savefig('explore_only' + '.png')

        return

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

    def get_hp_range_definition(self):
        range_def_dict = {
            'h_0': [0.0, 1.0], 'h_1': [0.0, 1.0],

            'optimizer_list': ['Adadelta', 'Adagrad', 'Momentum', \
                    'Adam', 'RMSProp', 'gd'],
            'lr': {
                    'Adadelta': [0.01, 0.1, 1.0],
                    'Adagrad': [0.0001, 0.001, 0.01],
                    'Momentum': [0.000001, 0.00001, 0.0001],
                    'Adam': [0.00001, 0.0001, 0.001],
                    'RMSProp': [0.000001, 0.00001, 0.0001, 0.001],
                    'gd': [0.00001, 0.0001, 0.001]
                    },
            'momentum': [0.00, 0.99],
            'grad_decay': [0.50, 0.99],
            'decay_steps': [30, 40, 50, 60, 70, 80, 90, 100],
            'decay_rate': [0.1, 1.0],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, \
                            0.5, 0.6, 0.7, 0.8, 0.9],
            'regularizer': ['l1_regularizer', \
                            'l2_regularizer', \
                            'l1_l2_regularizer', \
                            'None'],
            'initializer': ['tf.glorot_normal_initializer', \
                            'orthogonal', \
                            'tf.keras.initializers.he_normal',
                            'None'],
            'batch_size': [255],
            'num_filters_1': [24, 32],
            'kernel_size_1': [3, 5, 7],
            'kernel_size_2': [3, 5, 7],
            'activation': ['relu', 'softplus', 'tanh', 'sigmoid', 'selu']
            }
        return range_def_dict
    
    def load_hp_space(self):
        range_def = self.get_hp_range_definition()
        space = {
            'h_0': hp.uniform('h_0', \
                        range_def['h_0'][0], range_def['h_0'][1]),
            'h_1': hp.uniform('h_1', \
                        range_def['h_1'][0], range_def['h_1'][1]),

            'opt_case':hp.choice('opt_case',
            [
                {
                    'optimizer': 'Adadelta',
                    'lr': hp.choice('lr', range_def['lr']['Adadelta'])
                },
                {
                    'optimizer': 'Adagrad',
                    'lr': hp.choice('lr', range_def['lr']['Adagrad'])
                },
                {
                    'optimizer': 'Momentum',
                    'lr': hp.choice('lr', range_def['lr']['Momentum']),
                    'momentum': hp.uniform('momentum', \
                        range_def['momentum'][0], range_def['momentum'][1])
                },
                {
                    'optimizer': 'Adam',
                    'lr': hp.choice('lr', range_def['lr']['Adam'])
                },
                {
                    'optimizer': 'RMSProp',
                    'lr': hp.choice('lr', range_def['lr']['RMSProp']),
                    'grad_decay': hp.uniform('grad_decay', \
                        range_def['grad_decay'][0], range_def['grad_decay'][1]),
                    'momentum': hp.uniform('momentum', \
                        range_def['momentum'][0], range_def['momentum'][1])
                },
                {
                    'optimizer': 'gd',
                    'lr': hp.choice('lr', range_def['lr']['gd'])
                }
            ]),
            'decay_steps': hp.choice('decay_steps', \
                        range_def['decay_steps']),
            'decay_rate': hp.uniform('decay_rate', \
                        range_def['decay_rate'][0], range_def['decay_rate'][1]),
            'dropout': hp.choice('dropout', \
                        range_def['dropout']),
            'regularizer': hp.choice('regularizer', \
                        range_def['regularizer']),
            'initializer': hp.choice('initializer', \
                        range_def['initializer']),
            'batch_size': hp.randint('batch_size', range_def['batch_size'][0]),
            
            # To be continued
            'num_filters_1': hp.choice('num_filters_1', \
                        [24, 32]),
            'kernel_size_1': hp.choice('kernel_size_1', \
                        [3, 5, 7]),
            'kernel_size_2': hp.choice('kernel_size_2', \
                        [3, 5, 7]),
            'activation': hp.choice('kernel_size_2', \
                        ['relu', 'softplus', 'tanh', 'sigmoid', 'selu'])
            }
        return space