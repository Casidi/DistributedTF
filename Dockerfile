FROM tensorflow/tensorflow

RUN apt-get update && \
    apt-get --yes install mpich && \
    pip install mpi4py && \
    apt-get --yes install openssh-server