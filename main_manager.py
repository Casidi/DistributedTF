from __future__ import print_function

from mpi4py import MPI
import os
import subprocess
import time
import datetime
import tensorflow as tf

from pbt_cluster import PBTCluster
from training_worker import TrainingWorker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
master_rank = 0
if rank == master_rank:
    subprocess.call(['rm', '-rf', 'savedata'])
    subprocess.call(['mkdir', 'savedata'])

    #The PBT case
    cluster = PBTCluster(2, comm, master_rank)
    #The exploit only case
    #cluster = PBTCluster(2, comm, master_rank, do_explore=False)
    #The explore only case
    #cluster = PBTCluster(2, comm, master_rank, do_exploit=False)
    #The grid search case
    #cluster = PBTCluster(2, comm, master_rank, do_exploit=False, do_explore=False)

    start_time = time.time()

    cluster.train(50)
    cluster.flush_all_instructions()

    end_time = time.time()
    print('Training takes {}'.format(datetime.timedelta(seconds=(end_time-start_time))))

    cluster.report_plot_for_toy_model()
    #cluster.report_accuracy_plot()

    cluster.kill_all_workers()
else:
    worker = TrainingWorker(comm, master_rank)
    worker.main_loop()
