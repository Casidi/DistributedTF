from mpi4py import MPI
from enum import Enum
from hyperopt import hp
import hyperopt.pyll.stochastic
import math
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Instruction(Enum):
    ADD_GRAPHS = 0
    INIT = 1
    EXIT = 2

class SimpleNet:
    def __init__(self, sess, cluster_id):
        self.sess = sess
        self.cluster_id = cluster_id

class PBTCluster:
    def __init__(self, pop_size, comm):
        self.pop_size = pop_size
        self.comm = comm

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
        graphs_per_worker = math.ceil(float(self.pop_size) / float((comm.Get_size() - 1)))
        graphs_to_make = len(all_hparams_need_training)

        reqs = []
        for worker_rank in range(1, comm.Get_size()):
            begin = (worker_rank-1)*graphs_per_worker
            end = min(graphs_per_worker, graphs_to_make) + begin
            hparams_for_the_worker = all_hparams_need_training[begin: end]
            reqs.append(comm.isend((Instruction.ADD_GRAPHS, hparams_for_the_worker, begin), dest=worker_rank))
            graphs_to_make -= graphs_per_worker
        for req in reqs:
            req.wait()

    def initialize_all_graphs(self):
        reqs = []
        for i in range(1, comm.Get_size()):
            reqs.append(comm.isend((Instruction.INIT, ), dest=i))
        for req in reqs:
            req.wait()

    def kill_all_workers(self):
        reqs = []
        for i in range(1, comm.Get_size()):
            reqs.append(comm.isend((Instruction.EXIT, ), dest=i))
        for req in reqs:
            req.wait()

    def train(self, until_step_num):
        print('Train {} steps'.format(until_step_num))

    def get_hp_range_definition(self):
        range_def_dict = {
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
        space = {'opt_case':hp.choice('opt_case',
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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    cluster = PBTCluster(50, comm)
    cluster.train(10)
    cluster.kill_all_workers()
else:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)
    worker_graphs = []
    while True:
        data = comm.recv(source=0)
        inst = data[0]
        if inst == Instruction.ADD_GRAPHS:
            hparam_list = data[1]
            cluster_id_begin = data[2]
            cluster_id_end = cluster_id_begin + len(hparam_list)
            print('[{}]Got {} hparams'.format(rank, len(hparam_list)))

            for i in range(cluster_id_begin, cluster_id_end):
                worker_graphs.append(SimpleNet(sess, i))
        elif inst == Instruction.INIT:
            print('[{}]Initializing graphs'.format(rank))
        elif inst == Instruction.EXIT:
            break
        else:
            print('Invalid instruction!!!!')