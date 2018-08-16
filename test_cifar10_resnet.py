import sys
import os
sys.path.insert(0, './resnet')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#NOTE: tensorflow >= 1.10 required
from resnet import cifar10_main

hp = {
    'opt_case': {'lr': 0.1, 'optimizer': 'Momentum', 'momentum': 0.9},
    'decay_steps': 20,
    'decay_rate': 0.1,
    'weight_decay': 2e-4,
    'regularizer': 'l2_regularizer',
    'initializer': 'he_init',
    'batch_size': 128}

model_id = 7
save_base_dir = './resnet/model_'
data_dir = '/home/K8S/dataset/cifar10'
train_epochs = 1

for i in range(50):
    eval_accuracy, model_id = \
        cifar10_main.main(hp, model_id, save_base_dir, data_dir, train_epochs)
    print 'Epoch {}, acc = {}'.format(i+1, eval_accuracy)
