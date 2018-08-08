# mpirun --oversubscribe -n 5 python main_manager.py
#docker run hongfr/mpi-with-ssh

from mpi4py import MPI
import os
import time
import tensorflow as tf

from pbt_cluster import PBTCluster
from simple_net import SimpleNet
from constants import WorkerInstruction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
master_rank = 0
if rank == master_rank:
    cluster = PBTCluster(10, comm, master_rank)
    time.sleep(1)
    print 'round 1'
    cluster.train(10)
    cluster.exploit()
    cluster.explore()
    time.sleep(1)
    print 'round 2'
    cluster.train(10)
    cluster.exploit()
    cluster.explore()
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
        data = comm.recv(source=master_rank)
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
            for g in worker_graphs:
                g.train(num_steps)
                print 'Graph {} step = {},  loss = {}'.format(g.cluster_id, g.train_step, g.get_loss())
        elif inst == WorkerInstruction.GET:
            vars_to_send = []
            for g in worker_graphs:
                vars_to_send.append(g.get_values())
            comm.send(vars_to_send, dest=master_rank)
        elif inst == WorkerInstruction.EXIT:
            break
        else:
            print('Invalid instruction!!!!')