'''
The common entry point of all nodes.
'''

from __future__ import print_function

from mpi4py import MPI
import os
import subprocess

from pbt_cluster import PBTCluster
from training_worker import TrainingWorker

from cifar10_model import Cifar10Model
from toy_model import ToyModel
from mnist_model import MNISTModel
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
##################
# configurations
##################
# The parameters to generate toy model graphs
# master_rank = 0
# train_round = 30
# population_size = 2
# epochs_per_round = 4
# target_model = ToyModel
# do_exploit = True
# do_explore = True

master_rank = 0
train_round = 20 
population_size = 20
epochs_per_round = 1
do_exploit = True
do_explore = True

if len(sys.argv) >= 2:
    population_size = int(sys.argv[1])

#target_model = ToyModel
target_model = MNISTModel
#target_model = Cifar10Model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == master_rank:
    subprocess.call(['rm', '-rf', 'savedata'])
    subprocess.call(['mkdir', 'savedata'])

    # Generates HPs and send to workers.
    cluster = PBTCluster(population_size, comm, master_rank, 
                    epochs_per_round=epochs_per_round, 
                    do_exploit=do_exploit, do_explore=do_explore)

    cluster.dump_all_models_to_json('savedata/initial_hp.json')
    elapsed_time = cluster.train(train_round)

    with open('test_results.txt', 'a') as result_file:
        result_file.write('n = {}, pop_size = {}, time = {}s\n'.format(comm.Get_size(), population_size, elapsed_time))

    if target_model == ToyModel:
        cluster.report_plot_for_toy_model()
    cluster.report_accuracy_plot()
    cluster.report_lr_plot()
    cluster.report_best3_plot()
    cluster.report_best_model()
    cluster.print_profiling_info()
    cluster.kill_all_workers()
else:
    worker = TrainingWorker(comm, master_rank, target_model_class=target_model)
    worker.main_loop()
