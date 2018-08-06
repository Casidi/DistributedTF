# docker run -v /home/icls456251/Desktop/DistributedTF:/DistributedTF -it tensorflow/tensorflow:latest bash
# docker run -v /home/icls456251/Desktop/DistributedTF:/DistributedTF -it my-mpi-image
#docker run -v /home/icls456251/Desktop/DistributedTF:/nfs_shared -it tensorflow/tensorflow bash
# docker build -t my-mpi-image .

import docker

def remote_control_container(container, cmd):
    original_text_to_send = '{}\n'.format(cmd)
    s = container.attach_socket(params={'stdin': 1, 'stream': 1})
    s.send(original_text_to_send.encode('utf-8'))
    s.close()

client = docker.from_env()

master_node = client.containers.list()[0]
master_node.exec_run("rm -f /DistributedTF/all_public_keys", user='mpiuser')
master_node.exec_run("touch /DistributedTF/all_public_keys", user='mpiuser')
for node in client.containers.list():
    node.exec_run("rm -f /home/mpiuser/.ssh/id_rsa.pub", user='mpiuser')
    node.exec_run("rm -f /home/mpiuser/.ssh/id_rsa", user='mpiuser')
    exit_code, output = node.exec_run("ssh-keygen -f /home/mpiuser/.ssh/id_rsa -t rsa -N ''", user='mpiuser')
    print output
    exit_code, output = node.exec_run("sh -c 'cat /home/mpiuser/.ssh/id_rsa.pub >> /DistributedTF/all_public_keys'", user='mpiuser')
    print output

for node in client.containers.list():
    node.exec_run("touch /home/mpiuser/.ssh/authorized_keys", user='mpiuser')
    node.exec_run("cp /DistributedTF/all_public_keys /home/mpiuser/.ssh/authorized_keys", user='mpiuser')
    node.exec_run('cp /DistributedTF/ssh_config /home/mpiuser/.ssh/config', user='mpiuser')
    exit_code, output = node.exec_run('/etc/init.d/ssh restart')
    print output

# the su may requires password
#remote_control_container(master_node, 'su mpiuser')
#remote_control_container(master_node, 'mpirun -hostfile /DistributedTF/hostfile -n 6 python /DistributedTF/mpi_main.py')