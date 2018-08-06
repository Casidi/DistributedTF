# minikube mount /home/icls456251/Desktop/DistributedTF:/DistributedTF

from kubernetes import client, config, watch
from kubernetes.stream import stream

config.load_kube_config()

v1 = client.CoreV1Api()

print("Listing pods with their IPs:")
ret = v1.list_pod_for_all_namespaces(watch=False)
hostfile = open('hostfile', 'w')
for i in ret.items:
    if i.metadata.namespace == 'default':
        print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
        print(i.status.container_statuses[0].name)
        '''result = v1.connect_get_namespaced_pod_exec(i.metadata.name, i.metadata.namespace, 
            command='ls', container='my-pod', tty=True)
        print result'''
        hostfile.write(i.status.pod_ip + '\n')

hostfile.close()