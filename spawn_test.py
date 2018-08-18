from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD.Spawn(sys.executable, args=['training_worker.py'], maxprocs=2)
comm.Disconnect()