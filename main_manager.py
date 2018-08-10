# mpirun --oversubscribe -n 5 python main_manager.py
#docker run hongfr/mpi-with-ssh

from mpi4py import MPI
import os
import time
import tensorflow as tf

from pbt_cluster import PBTCluster
from simple_net import SimpleNet
from toy_model import ToyModel
from constants import WorkerInstruction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
master_rank = 0
if rank == master_rank:
    #The PBT case
    cluster = PBTCluster(2, comm, master_rank)
    #The exploit only case
    #cluster = PBTCluster(2, comm, master_rank, do_explore=False)
    #The explore only case(still dirty, needs to change the code around line 78 to make this work)
    #cluster = PBTCluster(2, comm, master_rank, do_exploit=False)
    #The grid search case
    #cluster = PBTCluster(2, comm, master_rank, do_exploit=False, do_explore=False)

    cluster.train(100)

    cluster.report_plot_for_toy_model()
    cluster.kill_all_workers()
else:
    worker_graphs = []

    while True:
        data = comm.recv(source=master_rank)
        inst = data[0]
        if inst == WorkerInstruction.ADD_GRAPHS:
            hparam_list = data[1]
            cluster_id_begin = data[2]
            cluster_id_end = cluster_id_begin + len(hparam_list)
            print('[{}]Got {} hparams'.format(rank, len(hparam_list)))

            for i in range(cluster_id_begin, cluster_id_end):
                hparam = hparam_list[i-cluster_id_begin]
                #new_graph = SimpleNet(sess, i, hparam)
                new_graph = ToyModel(i, hparam)
                worker_graphs.append(new_graph)
        elif inst == WorkerInstruction.INIT:
            print('[{}]Initializing graphs'.format(rank))
            for g in worker_graphs:
                g.init_variables()
        elif inst == WorkerInstruction.TRAIN:
            num_steps = data[1]
            for g in worker_graphs:
                g.train(num_steps)
                print 'Graph {} step = {},  loss = {}'.format(g.cluster_id, g.train_step, g.get_loss())
        elif inst == WorkerInstruction.GET:
            vars_to_send = []
            for g in worker_graphs:
                vars_to_send.append(g.get_values())
            comm.send(vars_to_send, dest=master_rank)
        elif inst == WorkerInstruction.SET:
            vars_to_set = data[1]
            for v in vars_to_set:
                for g in worker_graphs:
                    if g.cluster_id == v[0]:
                        print '[{}]Updating graph {} lr = {}'.format(rank, g.cluster_id, g.optimizer._learning_rate)
                        g.set_values(v)
                        g.need_explore = True
        elif inst == WorkerInstruction.EXPLORE:
            for g in worker_graphs:
                #g.need_explore = True # dirty fix to generate the explore only graph
                if g.need_explore:
                    print '[{}]Exploring graph {}'.format(rank, g.cluster_id)
                    g.perturb_hparams_and_update_graph()
                    g.need_explore = False
        elif inst == WorkerInstruction.GET_TRAIN_LOG:
            logs_to_send = []
            for g in worker_graphs:
                logs_to_send.append(g.train_log)
            comm.send(logs_to_send, dest=master_rank)
        elif inst == WorkerInstruction.EXIT:
            break
        else:
            print('Invalid instruction!!!!')