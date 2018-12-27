# DistributedTF
The distributed tensorflow project with MPI.

## How to run
The -n parameter should be at least 2 (1 master + 1 worker)
### Install dependencies
wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.3.tar.gz
tar -xvf openmpi-3.1.3.tar.gz
cd openmpi-3.1.3
./configure && make && sudo make install
pip install mpi4py matplotlib tensorflow hyperopt

### Download the dataset and put into ./datasets
mkdir datasets && cd datasets
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

### Test locally
```shell=
$ mpirun --oversubscribe -n 2 python main_manager.py
```
### Test on cluster
```shell=
$ mpirun -host host1_ip,host2_ip --oversubscribe -n 5 python main_manager.py
```
