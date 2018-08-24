from __future__ import print_function

from mpi4py import MPI
import os
import subprocess

from pbt_cluster import PBTCluster
from training_worker import TrainingWorker

from cifar10_model import Cifar10Model
from toy_model import ToyModel
from mnist_model import MNISTModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#configurations
#The parameters to generate toy model graphs
master_rank = 0
train_round = 30
population_size = 2
epochs_per_round = 4
target_model = ToyModel

# master_rank = 0
# train_round = 5
# population_size = 4
# epochs_per_round = 1
# #target_model = ToyModel
# #target_model = MNISTModel
# target_model = Cifar10Model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == master_rank:
    subprocess.call(['rm', '-rf', 'savedata'])
    subprocess.call(['mkdir', 'savedata'])

    #The PBT case
    #cluster = PBTCluster(population_size, comm, master_rank, epochs_per_round=epochs_per_round)
    #The exploit only case
    #cluster = PBTCluster(population_size, comm, master_rank, epochs_per_round=epochs_per_round, do_explore=False)
    #The explore only case
    #cluster = PBTCluster(population_size, comm, master_rank, epochs_per_round=epochs_per_round, do_exploit=False)
    #The grid search case
    cluster = PBTCluster(population_size, comm, master_rank, epochs_per_round=epochs_per_round, do_exploit=False, do_explore=False)

    cluster.dump_all_models_to_json('savedata/initial_hp.json')
    cluster.train(train_round)    
    cluster.dump_all_models_to_json('savedata/final_hp.json')

    if target_model == ToyModel:
        cluster.report_plot_for_toy_model()
    #cluster.report_accuracy_plot()
    #cluster.report_lr_plot()
    #cluster.report_best_model()
    cluster.kill_all_workers()
else:
    worker = TrainingWorker(comm, master_rank, target_model_class=target_model)
    worker.main_loop()
