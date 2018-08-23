# DistributedTF
The distributed tensorflow project with MPI.

## How to run
The -n parameter should be at least 2 (1 master + 1 worker)
### Test locally
```shell=
$ mpirun --oversubscribe -n 5 python main_manager.py
```
### Test on cluster
```shell=
$ mpirun -host host1_ip,host2_ip --oversubscribe -n 5 python main_manager.py
```