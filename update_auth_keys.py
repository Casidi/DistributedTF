import os

with open('/root/.ssh/authorized_keys', 'w') as file:
    for i in os.listdir('/nfs_shared'):
        if i.startswith('key_'):
            with open(os.path.join('/nfs_shared', i), 'r') as key_file:
                file.write(key_file.read())
            print 'processing {}'.format(i)
