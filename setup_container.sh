: '
apt-get  update
apt-get --yes install mpich
pip install mpi4py
apt-get --yes install openssh-server
'

ssh-keygen -f /root/.ssh/id_rsa -t rsa -N ''
cp /root/.ssh/id_rsa.pub /nfs_shared/key_$(hostname)
cp /nfs_shared/ssh_config /root/.ssh/config

: '
useradd -m  mpiuser
echo mpiuser:0000 | chpasswd
mkdir /home/mpiuser/.ssh
chgrp mpiuser /home/mpiuser/.ssh
chown mpiuser /home/mpiuser/.ssh

apt-get --yes install sudo
chmod 777 /nfs_shared
sudo -u mpiuser sh -c "ssh-keygen -f /home/mpiuser/.ssh/id_rsa -t rsa -N ''"
sudo -u mpiuser sh -c "cp /home/mpiuser/.ssh/id_rsa.pub /nfs_shared/key_$(hostname)"
sudo -u mpiuser sh -c "cp /nfs_shared/ssh_config /home/mpiuser/.ssh/config"
'

/etc/init.d/ssh restart