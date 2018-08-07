# mpirun --oversubscribe -n 5 python main_manager.py
#docker run hongfr/mpi-with-ssh
from mpi4py import MPI
from enum import Enum
from hyperopt import hp
import hyperopt.pyll.stochastic
import math
import tensorflow as tf
import numpy as np
import os
import time
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class WorkerInstruction(Enum):
    ADD_GRAPHS = 0
    INIT = 1
    EXIT = 2
    TRAIN = 3
    GET = 4

class SimpleNet:
    def __init__(self, sess, cluster_id):
        self.sess = sess
        self.cluster_id = cluster_id
        self.train_step = 0

        x_train = [[0.0, 1.0]]
        y_train = [[1.0]]
        self.input_layer = tf.constant(x_train)
        self.w1 = tf.Variable(tf.random_uniform([2, 1]))
        self.b1 = tf.Variable(tf.random_uniform([1]))
        self.output_layer = tf.sigmoid(tf.matmul(self.input_layer, self.w1) + self.b1)

        self.loss = tf.reduce_sum(tf.square(y_train - self.output_layer))
        self.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        self.vars = [self.w1, self.b1]

    def init_variables(self):
        self.sess.run([var.initializer for var in self.vars])

    def train(self, num_steps):
        for i in range(num_steps):
            self.sess.run(self.train_op)
            self.train_step += 1

    def get_loss(self):
        return self.sess.run(self.loss)

    def get_values(self):
        return [self.cluster_id] + self.sess.run([self.loss, self.w1, self.b1])

    def set_values(self, values):

        return

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
            reqs.append(comm.isend((WorkerInstruction.ADD_GRAPHS, hparams_for_the_worker, begin), dest=worker_rank))
            graphs_to_make -= graphs_per_worker
        for req in reqs:
            req.wait()

    def initialize_all_graphs(self):
        reqs = []
        for i in range(1, comm.Get_size()):
            reqs.append(comm.isend((WorkerInstruction.INIT, ), dest=i))
        for req in reqs:
            req.wait()

    def kill_all_workers(self):
        reqs = []
        for i in range(1, comm.Get_size()):
            reqs.append(comm.isend((WorkerInstruction.EXIT, ), dest=i))
        for req in reqs:
            req.wait()

    def train(self, until_step_num):
        reqs = []
        for i in range(1, comm.Get_size()):
            reqs.append(comm.isend((WorkerInstruction.TRAIN, until_step_num), dest=i))
        for req in reqs:
            req.wait()

    def exploit(self):
        reqs = []
        for i in range(1, comm.Get_size()):
            reqs.append(comm.isend((WorkerInstruction.GET, ), dest=i))
        for req in reqs:
            req.wait()

        all_values = []
        cluster_id_to_worker_rank = {}
        for i in range(1, comm.Get_size()):
            data = comm.recv(source=i)
            all_values += data
            for d in data:
                cluster_id_to_worker_rank[d[0]] = i

        # copy top 25% to bottom 25%
        print 'After exploit'
        all_values = sorted(all_values, key=lambda value: value[1])
        num_graphs_to_copy = math.ceil(float(self.pop_size) / 4.0)
        for i in range(num_graphs_to_copy):
            top_index = i
            bottom_index = len(all_values) - num_graphs_to_copy + i
            all_values[bottom_index][1] = all_values[top_index][1] #copy loss, not necessary
            all_values[bottom_index][2] = all_values[top_index][2] #copy w1
            all_values[bottom_index][2] = all_values[top_index][2] #copy b1
        print all_values

        return

    def explore(self):
        return


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
    cluster = PBTCluster(10, comm)
    time.sleep(1)
    print 'round 1'
    cluster.train(10)
    cluster.exploit()
    time.sleep(1)
    print 'round 2'
    cluster.train(10)
    cluster.exploit()
    time.sleep(1)
    print 'round 3'
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
        if inst == WorkerInstruction.ADD_GRAPHS:
            hparam_list = data[1]
            cluster_id_begin = data[2]
            cluster_id_end = cluster_id_begin + len(hparam_list)
            print('[{}]Got {} hparams'.format(rank, len(hparam_list)))

            for i in range(cluster_id_begin, cluster_id_end):
                worker_graphs.append(SimpleNet(sess, i))
        elif inst == WorkerInstruction.INIT:
            print('[{}]Initializing graphs'.format(rank))
            for g in worker_graphs:
                g.init_variables()
        elif inst == WorkerInstruction.TRAIN:
            num_steps = data[1]
            #time.sleep(random.uniform(0.0, 3.0)) #simulate the calculation time
            for g in worker_graphs:
                g.train(num_steps)
                print 'Graph {} step = {},  loss = {}'.format(g.cluster_id, g.train_step, g.get_loss())
        elif inst == WorkerInstruction.GET:
            vars_to_send = []
            for g in worker_graphs:
                vars_to_send.append(g.get_values())
            comm.send(vars_to_send, dest=0)
        elif inst == WorkerInstruction.EXIT:
            break
        else:
            print('Invalid instruction!!!!')