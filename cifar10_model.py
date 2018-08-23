from constants import get_hp_range_definition
import sys
import random
import numpy as np
sys.path.insert(0, './resnet')

from resnet import cifar10_main

class Cifar10Model:
    def __init__(self, cluster_id, hparams, save_base_dir):
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.save_base_dir = save_base_dir
        self.epoches_trained = 0
        self.need_explore = False
        self._perturb_factors = [0.8, 1.2]

        #for debugging
        '''self.hparams = {
            'opt_case': {'lr': 0.1, 'optimizer': 'Momentum', 'momentum': 0.9},
            'decay_steps': 20,
            'decay_rate': 0.1,
            'weight_decay': 2e-4,
            'regularizer': 'l2_regularizer',
            'initializer': 'he_init',
            'batch_size': 128}'''
        
        #if self.cluster_id == 0:
        #    self.hparams['opt_case']['lr'] = 100000

        if isinstance(self.hparams['batch_size'], np.ndarray):
            self.hparams['batch_size'] = self.hparams['batch_size'].item()
        self.accuracy = 0.0
    
    def train(self, num_epoch, total_epochs):
        print('Training model {}'.format(self.cluster_id))
        data_dir = '/home/K8S/dataset/cifar10'
        self.accuracy, model_id = \
            cifar10_main.main(self.hparams, self.cluster_id, 
                    self.save_base_dir, data_dir, 
                    num_epoch, total_epochs, self.epoches_trained)
        self.epoches_trained += num_epoch
        return

    def perturb_hparams(self):
        def _perturb_float(val, limit_min, limit_max):
            # Noted, some hp value can't exceed reasonable range
            float_string = str(limit_min)
            if 'e' in float_string:
                _, n_digits = float_string.split('e')
                if '-' in n_digits:
                    n_digits = int(n_digits) * -1
                else:
                    n_digits = int(n_digits)
            else:
                n_digits = str(limit_min)[::-1].find('.')
            min = val * self._perturb_factors[0]
            max = val * self._perturb_factors[1]
            if min < limit_min:
                min = limit_min
                n_digits += 1
            if max > limit_max:
                max = limit_max
            val = random.uniform(min, max)
            val = round(val, n_digits)

            return val

        def _perturb_int(val, limit_min, limit_max):
            # Noted, some hp value can't exceed reasonable range
            if limit_min == limit_max:
                limit_min = 0
            min = int(np.floor(val * self._perturb_factors[0]))
            max = int(np.ceil(val * self._perturb_factors[1]))
            if min < limit_min:
                min = limit_min
            if max > limit_max:
                max = limit_max
            return random.randint(min, max)

        range_def = get_hp_range_definition()
        for key, value in self.hparams.iteritems():
            if isinstance(value, float):
                self.hparams[key] = _perturb_float(value, range_def[key][0], range_def[key][-1])
            elif isinstance(value, int):
                self.hparams[key] = _perturb_int(value, range_def[key][0], range_def[key][-1])
            else:
                if key != 'opt_case':
                    # Model-architecture related HP is kept the same
                    if key == 'num_filters_1' or key == 'kernel_size_1' \
                            or key == 'kernel_size_2' or key == 'activation' \
                            or key == 'initializer':
                        pass
                    else:
                        self.hparams[key] = random.choice(range_def[key])
                else:
                    # Notes, hparams[key]['optimizer'] is kept the same
                    optimizer = self.hparams[key]['optimizer']
                    self.hparams[key]['lr'] = \
                        _perturb_float(self.hparams[key]['lr'], \
                                       range_def['lr'][optimizer][0], \
                                       range_def['lr'][optimizer][-1])

                    if self.hparams[key]['optimizer'] == 'Momentum' \
                            or self.hparams[key]['optimizer'] == 'RMSProp':
                        self.hparams[key]['momentum'] = \
                            _perturb_float(self.hparams[key]['momentum'], \
                                           range_def['momentum'][0], range_def['momentum'][-1])
                    if self.hparams[key]['optimizer'] == 'RMSProp':
                        self.hparams[key]['grad_decay'] = \
                            _perturb_float(self.hparams[key]['grad_decay'], \
                                           range_def['grad_decay'][0], range_def['grad_decay'][-1])
        return

    def get_accuracy(self):
        return self.accuracy

    def get_values(self):
        return [self.cluster_id, self.get_accuracy(), self.hparams]

    def set_values(self, values):
        self.hparams = values[2]
