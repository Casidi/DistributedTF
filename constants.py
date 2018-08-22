from enum import Enum
from hyperopt import hp
import hyperopt.pyll.stochastic

class WorkerInstruction(Enum):
    ADD_GRAPHS = 0
    EXIT = 1
    TRAIN = 2
    GET = 3
    SET = 4
    EXPLORE = 5

def get_hp_range_definition():
    range_def_dict = {
        'h_0': [0.0, 1.0], 'h_1': [0.0, 1.0],

        'optimizer_list': ['Adadelta', 'Adagrad', 'Momentum', \
                'Adam', 'RMSProp', 'gd'],
        'lr': {
                'Adadelta': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                'Adagrad': [1e-3, 1e-2, 1e-1, 0.5, 1.0],
                'Momentum': [1e-3, 1e-2, 1e-1, 0.5, 1.0],
                'Adam': [1e-4, 1e-3, 1e-2, 1e-1],
                'RMSProp': [1e-5, 1e-4, 1e-3],
                'gd': [1e-2, 1e-1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                },
        'momentum': [0.00, 0.9],
        'grad_decay': [0.00, 0.9],
        'decay_steps': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'decay_rate': [0.1, 1.0],
        'weight_decay': [1e-8, 1e-2],
        'regularizer': ['l1_regularizer', \
                        'l2_regularizer', \
                        'l1_l2_regularizer', \
                        'None'],
        'initializer': ['glorot_normal', \
                        'orthogonal', \
                        'he_init',
                        'None'],
        'batch_size': [191]
    }
    return range_def_dict
    
def load_hp_space():
    range_def = get_hp_range_definition()
    space = {       
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
        'weight_decay': hp.uniform('weight_decay', \
                    range_def['weight_decay'][0], range_def['weight_decay'][1]),
        'regularizer': hp.choice('regularizer', \
                    range_def['regularizer']),
        'initializer': hp.choice('initializer', \
                    range_def['initializer']),
        'batch_size': hp.randint('batch_size', range_def['batch_size'][0])
        }
    space['batch_size'] += 65
    return space

def generate_random_hparam():
    hp_space = load_hp_space()
    return hyperopt.pyll.stochastic.sample(hp_space)