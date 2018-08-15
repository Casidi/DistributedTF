import sys
sys.path.insert(0, './resnet')

from resnet import cifar10_main

hp = {
    'opt_case': {'lr': 0.1, 'optimizer': 'Momentum', 'momentum': 0.9},
    'decay_steps': 20,
    'decay_rate': 0.1,
    'weight_decay': 2e-4,
    'regularizer': 'l2_regularizer',
    'initializer': 'he_init',
    'batch_size': 128}

model_id = 0
save_base_dir = './model_'
data_dir = '/home/icls456251/Desktop/cifar10'
train_epochs = 1

eval_accuracy, model_id = \
    cifar10_main.main(hp, model_id, save_base_dir, data_dir, train_epochs)