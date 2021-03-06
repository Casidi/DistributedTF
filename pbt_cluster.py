'''
The implementation of PBT algorithm.
The PBTCluster should only be run on the master node.
'''

from __future__ import print_function

import math
import shutil
import subprocess
import os
import csv
import json
import time
import datetime

from constants import generate_random_hparam
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
from matplotlib import pyplot
import numpy as np
import six

from constants import WorkerInstruction
from constants import get_hp_range_definition, load_hp_space

class PBTCluster:
    def __init__(self, pop_size, comm, master_rank, epochs_per_round, do_exploit=True, do_explore=True):
        self.pop_size = pop_size
        self.comm = comm
        self.master_rank = master_rank
        self.epochs_per_round = epochs_per_round
        self.do_exploit = do_exploit
        self.do_explore = do_explore        

        self.exploit_time = 0

        self.dispatch_hparams_to_workers()

    def dispatch_hparams_to_workers(self):
        all_hparams_need_training = []

        for i in range(self.pop_size):
            all_hparams_need_training.append(generate_random_hparam())
        
        '''if os.path.isfile('initial_hp.json'):
            print('Initialize from initial_hp.json')
            all_hparams_need_training = []
            with open('initial_hp.json', 'r') as fp:
                data = json.load(fp)
            for i in data:
                all_hparams_need_training.append(i['hparams'])
            self.pop_size = len(all_hparams_need_training)'''

        print('Population size = {}'.format(len(all_hparams_need_training)))
        graphs_per_worker = math.ceil(float(self.pop_size) / float((self.comm.Get_size() - 1)))
        graphs_to_make = len(all_hparams_need_training)

        if self.do_explore and not self.do_exploit:
            is_explore_only = True
        else:
            is_explore_only = False

        reqs = []
        num_workers_sent = 0
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                begin = num_workers_sent * graphs_per_worker
                end = min(graphs_per_worker, graphs_to_make) + begin
                hparams_for_the_worker = all_hparams_need_training[begin: end]
                reqs.append(self.comm.isend((WorkerInstruction.ADD_GRAPHS,
                                             hparams_for_the_worker, 
                                             begin, is_explore_only), dest=i))
                graphs_to_make -= graphs_per_worker
                num_workers_sent += 1
        for req in reqs:
            req.wait()

    def kill_all_workers(self):
        reqs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.EXIT,), dest=i))
        for req in reqs:
            req.wait()

    def train(self, round_num):
        start_time = time.time()
        for round in range(round_num):
            round_start_time = time.time()
            print('\nRound {}'.format(round))

            reqs = []
            for i in range(0, self.comm.Get_size()):
                if i != self.master_rank:
                    reqs.append(self.comm.isend((WorkerInstruction.TRAIN, self.epochs_per_round, self.epochs_per_round*round_num), dest=i))
            for req in reqs:
                req.wait()

            if self.do_exploit:
                self.exploit()

            if self.do_explore:
                self.explore()

            print('Round elapsed time: {}\n'.format(datetime.timedelta(seconds=(time.time()-round_start_time))))

        self.flush_all_instructions()
        end_time = time.time()
        print('Total elapsed time: {}'.format(datetime.timedelta(seconds=(end_time-start_time))))
        return end_time - start_time

    def exploit(self):
        reqs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.GET,), dest=i))
        for req in reqs:
            req.wait()

        all_values = []
        cluster_id_to_worker_rank = {}
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                data = self.comm.recv(source=i)
                all_values += data
                for d in data:
                    cluster_id_to_worker_rank[d[0]] = i

        exploit_begin_time = time.time()
        # copy top 25% to bottom 25%
        all_values = sorted(all_values, key=lambda value: value[1])
        self.pop_size = len(all_values)
        '''print 'The ranking before exploit'
        for i in all_values:
            print 'graph {}, loss={}'.format(i[0], i[1])'''
        num_graphs_to_copy = math.ceil(self.pop_size / 4.0)
        graphs_need_updating = []
        for i in range(num_graphs_to_copy):
            bottom_index = i
            top_index = len(all_values) - num_graphs_to_copy + i
            all_values[bottom_index][1] = all_values[top_index][1]  # copy accuracy, not necessary
            all_values[bottom_index][2] = all_values[top_index][2]  # copy hparams

            source_dir = './savedata/model_' + str(all_values[top_index][0])
            destination_dir = './savedata/model_' + str(all_values[bottom_index][0])
            self.copyfiles(source_dir, destination_dir)

            graphs_need_updating.append(bottom_index)
            print('Copied: {} -> {}'.format(all_values[top_index][0], all_values[bottom_index][0]))

        # only update the bottom graphs
        worker_rank_to_graphs_need_updating = {}
        for i in range(self.comm.Get_size()):
            worker_rank_to_graphs_need_updating[i] = []
        for i in graphs_need_updating:
            worker_rank = cluster_id_to_worker_rank[all_values[i][0]]
            worker_rank_to_graphs_need_updating[worker_rank].append(all_values[i])

        reqs = []
        for worker_rank, values in six.iteritems(worker_rank_to_graphs_need_updating):
            reqs.append(self.comm.isend((WorkerInstruction.SET, values), dest=worker_rank))
        for req in reqs:
            req.wait()

        self.exploit_time += time.time() - exploit_begin_time

    def copyfiles(self, src_dir, dest_dir):
        if src_dir == dest_dir:
            print('Warning, src_dir and dest_dir are the same')
            return
        for i in os.listdir(dest_dir):
            path = os.path.join(dest_dir, i)
            if not os.path.isdir(path) and i != 'learning_curve.csv' and i != 'theta.csv' and not i.startswith('events.out') and not i.startswith('.nfs'):
                #print('Removing: {}'.format(path))
                subprocess.call(['rm', '-f', path])
        for i in os.listdir(src_dir):
            path = os.path.join(src_dir, i)
            if not os.path.isdir(path)  and i != 'theta.csv' and i != 'learning_curve.csv' and not i.startswith('events.out') and not i.startswith('.nfs'):
                #print('Copying: {}'.format(path))
                subprocess.call(['cp', path, dest_dir])

    def explore(self):
        reqs = []
        for i in range(self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.EXPLORE,), dest=i))
        for req in reqs:
            req.wait()

    def flush_all_instructions(self):
        # GET will block until all workers finish their instruction queues
        self.get_all_values()

    def get_all_values(self):
        reqs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                reqs.append(self.comm.isend((WorkerInstruction.GET,), dest=i))
        for req in reqs:
            req.wait()

        all_values = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                data = self.comm.recv(source=i)
                all_values += data
        return all_values

    def print_profiling_info(self):
        reqs = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                print('Requesting profiling info for worker {}'.format(i))
                reqs.append(self.comm.isend((WorkerInstruction.GET_PROFILING_INFO,), dest=i))
        for req in reqs:
            req.wait()

        all_infos = []
        for i in range(0, self.comm.Get_size()):
            if i != self.master_rank:
                print('Recv from worker {}'.format(i))
                data = self.comm.recv(source=i)
                all_infos.append(data)

        total_train_time = 0
        total_explore_time = 0
        for i in all_infos:
            total_train_time += i[0]
            total_explore_time += i[1]
        total_train_time /= len(all_infos)
        total_explore_time /= len(all_infos)
        
        print('')
        print('=======Profiling Information========')
        print('Total train time: {}'.format(datetime.timedelta(seconds=(total_train_time))))
        print('Total exploit time: {}'.format(datetime.timedelta(seconds=(self.exploit_time))))
        print('Total explore time: {}\n'.format(datetime.timedelta(seconds=(total_explore_time))))

    def dump_all_models_to_json(self, filename):
        all_values = self.get_all_values()
        all_values = sorted(all_values, key=lambda value: value[1])
        report_list = []
        for i in range(len(all_values)):
            all_values[i][1] = float(all_values[i][1])
            report_list.append({'model_id': all_values[i][0],
                                'accuracy': all_values[i][1],
                                'hparams': all_values[i][2]})

        with open(filename, 'w') as fp:
            json.dump(report_list, fp, indent=4, sort_keys=True)
        print('Saving all models to {}'.format(filename))


    def report_best_model(self):
        all_values = self.get_all_values()
        all_values = sorted(all_values, key=lambda value: value[1])
        report_dict = {}
        report_dict['best_model_id'] = all_values[len(all_values) - 1][0]
        report_dict['best_acc'] = float(all_values[len(all_values) - 1][1])
        report_dict['best_hparams'] = all_values[len(all_values) - 1][2]
        
        filename = 'savedata/best_model.json'
        with open(filename, 'w') as fp:
            json.dump(report_dict, fp, indent=4, sort_keys=True)
        print('Saving best model to {}'.format(filename))
        
    def report_plot_for_toy_model(self):
        csv_file_names = []
        for i in os.listdir('./savedata'):
            if i.startswith('model_'):
                csv_file_names.append(os.path.join('./savedata', i, 'theta.csv'))

        all_theta = []
        for i in csv_file_names:
            acc = []
            with open(i) as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    acc.append([float(row[rows.fieldnames[0]]), float(row[rows.fieldnames[1]])])
            all_theta.append(acc)

        linspace_x = np.linspace(start=0, stop=1, num=100)
        linspace_y = np.linspace(start=0, stop=1, num=100)
        x, y = np.meshgrid(linspace_x, linspace_y)
        z = 1.2 - (x ** 2 + y ** 2)

        pyplot.figure()
        pyplot.xlabel(r'$\theta_0$')
        pyplot.ylabel(r'$\theta_1$')
        pyplot.xlim(0, 1)
        pyplot.ylim(0, 1)

        for i in all_theta:
            pyplot.plot(list(zip(*i))[0], list(zip(*i))[1], '.')
        pyplot.contour(x, y, z, colors='lightgray')
        # pyplot.show()

        if self.do_exploit and self.do_explore:
            pyplot.title('PBT')
            out_file_name = 'toy_PBT.png'
        elif self.do_exploit and not self.do_explore:
            pyplot.title('Exploit only')
            out_file_name = 'toy_exploit_only.png'
        elif not self.do_exploit and self.do_explore:
            pyplot.title('Explore only')
            out_file_name = 'toy_explore_only.png'
        else:
            pyplot.title('Grid search')
            out_file_name = 'toy_grid_search.png'
        out_file_name = os.path.join('savedata', out_file_name)
        pyplot.savefig(out_file_name)
        print('Writing results to {}'.format(out_file_name))

    def report_accuracy_plot(self):
        csv_file_names = []
        for i in os.listdir('./savedata'):
            if i.startswith('model_'):
                csv_file_names.append(os.path.join('./savedata', i, 'learning_curve.csv'))

        all_acc = []
        for i in csv_file_names:
            if not os.path.isfile(i):
                continue
            acc = []
            with open(i) as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    acc.append([int(row[rows.fieldnames[0]]), float(row[rows.fieldnames[1]])])
            all_acc.append(acc)

        pyplot.figure()
        for i in all_acc:
            pyplot.plot(list(zip(*i))[0], list(zip(*i))[1])

        pyplot.xlabel(r'Train epochs')
        pyplot.ylabel(r'Accuracy')
        pyplot.grid(True)

        if self.do_exploit and self.do_explore:
            pyplot.title('PBT')
            out_file_name = 'acc_PBT.png'
        elif self.do_exploit and not self.do_explore:
            pyplot.title('Exploit only')
            out_file_name = 'acc_exploit_only.png'
        elif not self.do_exploit and self.do_explore:
            pyplot.title('Explore only')
            out_file_name = 'acc_explore_only.png'
        else:
            pyplot.title('Grid search')
            out_file_name = 'acc_grid_search.png'
        out_file_name = os.path.join('savedata', out_file_name)
        pyplot.savefig(out_file_name)
        print('Writing results to {}'.format(out_file_name))

    def report_lr_plot(self):
        csv_file_names = []
        for i in os.listdir('./savedata'):
            if i.startswith('model_'):
                csv_file_names.append(os.path.join('./savedata', i, 'learning_curve.csv'))

        all_lr = []
        for i in csv_file_names:
            if not os.path.isfile(i):
                continue
            acc = []
            with open(i) as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    acc.append([int(row[rows.fieldnames[0]]), float(row[rows.fieldnames[3]])])
            all_lr.append(acc)

        pyplot.figure()
        for i in all_lr:
            pyplot.plot(list(zip(*i))[0], list(zip(*i))[1])

        pyplot.xlabel(r'Train epochs')
        pyplot.ylabel(r'Learning rate')
        pyplot.ylim(0, 1)
        pyplot.grid(True)

        if self.do_exploit and self.do_explore:
            pyplot.title('PBT')
            out_file_name = 'lr_PBT.png'
        elif self.do_exploit and not self.do_explore:
            pyplot.title('Exploit only')
            out_file_name = 'lr_exploit_only.png'
        elif not self.do_exploit and self.do_explore:
            pyplot.title('Explore only')
            out_file_name = 'lr_explore_only.png'
        else:
            pyplot.title('Grid search')
            out_file_name = 'lr_grid_search.png'
        out_file_name = os.path.join('savedata', out_file_name)
        pyplot.savefig(out_file_name)
        print('Writing results to {}'.format(out_file_name))

    def report_best3_plot(self):
        csv_file_names = []
        save_base_dir = './savedata'
        for i in os.listdir(save_base_dir):
            if i.startswith('model_'):
                csv_file_names.append(os.path.join(save_base_dir, i, 'learning_curve.csv'))

        all_acc = []
        for i in csv_file_names:
            if not os.path.isfile(i):
                continue
            acc = []
            with open(i) as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    acc.append([int(row[rows.fieldnames[0]]), float(row[rows.fieldnames[1]])])
            all_acc.append(acc)

        max_record_length = -1
        for i in all_acc:
            if len(i) > max_record_length:
                max_record_length = len(i)
            
        top_avg = []
        for i in range(max_record_length):
            column = []
            for j in all_acc:
                if len(j) > i:
                    column.append(j[i][1])
            column = sorted(column)

            epoch_index = 0
            for j in all_acc:
                if len(j) > i:
                    epoch_index = j[i][0]
                    break

            if len(column) == 0:
                top_avg.append((epoch_index, 0.0))
            elif len(column) < 3:
                sum = 0.0
                for j in column:
                    sum += j
                top_avg.append((epoch_index, sum/len(column)))
            else:    
                top_avg.append((epoch_index, (column[-1] + column[-2] + column[-3]) / 3.0))

        pyplot.figure()
        for i in all_acc:
            pyplot.plot(list(zip(*i))[0], list(zip(*i))[1], color=(0.0, 0.0, 0.5, 0.3))
            
        pyplot.plot(list(zip(*top_avg))[0], list(zip(*top_avg))[1], 'r')

        pyplot.xlabel(r'Train epochs')
        pyplot.ylabel(r'Accuracy')
        pyplot.ylim(0, 1)
        pyplot.grid(True)

        if self.do_exploit and self.do_explore:
            pyplot.title('PBT')
            out_file_name = 'best3_PBT.png'
        elif self.do_exploit and not self.do_explore:
            pyplot.title('Exploit only')
            out_file_name = 'best3_exploit_only.png'
        elif not self.do_exploit and self.do_explore:
            pyplot.title('Explore only')
            out_file_name = 'best3_explore_only.png'
        else:
            pyplot.title('Grid search')
            out_file_name = 'best3_grid_search.png'
        out_file_name = os.path.join(save_base_dir, out_file_name)
        pyplot.savefig(out_file_name)
        print('Writing results to {}'.format(out_file_name))
