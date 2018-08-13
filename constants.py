from enum import Enum
from hyperopt import hp

class WorkerInstruction(Enum):
    ADD_GRAPHS = 0
    EXIT = 2
    TRAIN = 3
    GET = 4
    SET = 5
    EXPLORE = 6
    GET_TRAIN_LOG = 7

def get_hp_range_definition():
    range_def_dict = {
        'h_0': [0.0, 1.0], 'h_1': [0.0, 1.0],

        'optimizer_list': ['Adadelta', 'Adagrad', 'Momentum', \
                'Adam', 'RMSProp', 'gd'],
        'lr': {
                'Adadelta': [0.01, 0.1, 1.0],
                'Adagrad': [0.0001, 0.001, 0.01],
                'Momentum': [0.000001, 0.00001, 0.0001],
                'Adam': [0.00001, 0.0001, 0.001],
                'RMSProp': [0.000001, 0.00001, 0.0001, 0.001],
                'gd': [0.00001, 0.0001, 0.001]
                },
        'momentum': [0.00, 0.99],
        'grad_decay': [0.50, 0.99],
        'decay_steps': [30, 40, 50, 60, 70, 80, 90, 100],
        'decay_rate': [0.1, 1.0],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, \
                        0.5, 0.6, 0.7, 0.8, 0.9],
        'regularizer': ['l1_regularizer', \
                        'l2_regularizer', \
                        'l1_l2_regularizer', \
                        'None'],
        'initializer': ['tf.glorot_normal_initializer', \
                        'orthogonal', \
                        'tf.keras.initializers.he_normal',
                        'None'],
        'batch_size': [255],
        'num_filters_1': [24, 32],
        'kernel_size_1': [3, 5, 7],
        'kernel_size_2': [3, 5, 7],
        'activation': ['relu', 'softplus', 'tanh', 'sigmoid', 'selu']
        }
    return range_def_dict
    
def load_hp_space():
    range_def = get_hp_range_definition()
    space = {
        'h_0': hp.uniform('h_0', \
                    range_def['h_0'][0], range_def['h_0'][1]),
        'h_1': hp.uniform('h_1', \
                    range_def['h_1'][0], range_def['h_1'][1]),

        'opt_case':hp.choice('opt_case',
        [
            {
                'optimizer': 'Adadelta',
                'lr': hp.choice('lr', range_def['lr']['Adadelta'])
            },
            {
                'optimizer': 'Adagrad',
                'lr': hp.choice('lr', range_def['lr']['Adagrad'])
            },
            {
                'optimizer': 'Momentum',
                'lr': hp.choice('lr', range_def['lr']['Momentum']),
                'momentum': hp.uniform('momentum', \
                    range_def['momentum'][0], range_def['momentum'][1])
            },
            {
                'optimizer': 'Adam',
                'lr': hp.choice('lr', range_def['lr']['Adam'])
            },
            {
                'optimizer': 'RMSProp',
                'lr': hp.choice('lr', range_def['lr']['RMSProp']),
                'grad_decay': hp.uniform('grad_decay', \
                    range_def['grad_decay'][0], range_def['grad_decay'][1]),
                'momentum': hp.uniform('momentum', \
                    range_def['momentum'][0], range_def['momentum'][1])
            },
            {
                'optimizer': 'gd',
                'lr': hp.choice('lr', range_def['lr']['gd'])
            }
        ]),
        'decay_steps': hp.choice('decay_steps', \
                    range_def['decay_steps']),
        'decay_rate': hp.uniform('decay_rate', \
                    range_def['decay_rate'][0], range_def['decay_rate'][1]),
        'dropout': hp.choice('dropout', \
                    range_def['dropout']),
        'regularizer': hp.choice('regularizer', \
                    range_def['regularizer']),
        'initializer': hp.choice('initializer', \
                    range_def['initializer']),
        'batch_size': hp.randint('batch_size', range_def['batch_size'][0]),
        
        # To be continued
        'num_filters_1': hp.choice('num_filters_1', \
                    [24, 32]),
        'kernel_size_1': hp.choice('kernel_size_1', \
                    [3, 5, 7]),
        'kernel_size_2': hp.choice('kernel_size_2', \
                    [3, 5, 7]),
        'activation': hp.choice('kernel_size_2', \
                    ['relu', 'softplus', 'tanh', 'sigmoid', 'selu'])
        }
    return space