import os
import re
import time
import argparse
import random
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
FLAGS = parser.parse_args()


directory = FLAGS.dir or "/workdir/optimal-summaries-public/_models/mimic-iii/vasopressor/"
if not os.path.exists(directory):
    os.makedirs(directory)


def launch_job(exp, time_limit=None, mem_limit=None):

    job_command = "python3 -u gpu_vasopressor_bottleneck.py"

    for k, v in exp.items():
        job_command += f" --{k}={v}"

    os.system(job_command)


# Run experiments with randomly sampled hyperparameters.
arr_opt_lr = [1e-4]
arr_opt_weight_decay = [1e-5]
arr_l1_lambda = [1e-3]
arr_cos_sim_lambda = [1e-2]

num_epochs = 1000
save_every = 10
N_experiments = 1


for r in range(1,2):
    for c in range(4,5):
        filename = "bottleneck_r{}_c{}_gridsearch".format(r, c)
        # Write hyperparameters to csv file
        fields = ['num_concepts', 'opt_lr', 'opt_weight_decay', 'l1_lambda', 'cos_sim_lambda','test auc']
        with open('{file_path}.csv'.format(file_path=os.path.join(directory, filename)), 'w+') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(fields)
            
        random.seed(r)
        for n in range(N_experiments):
            # Create exp dictionary d by randomly sampling from each of the arrays.
            for lr in arr_opt_lr:
                for wd in arr_opt_weight_decay:
                    for l1 in arr_l1_lambda:
                        for cs in arr_cos_sim_lambda:
                            d = {}

                            d['num_concepts'] = c
                            d['split_random_state'] = r
                            d['num_epochs'] = num_epochs
                            d['save_every'] = save_every

                            d['opt_lr'] = lr
                            d['opt_weight_decay'] = wd
                            
                            d['l1_lambda'] = l1
                            d['cos_sim_lambda'] = cs

                            d['output_dir'] = directory
                            d['model_output_name'] = 'bottleneck_r' + str(r) + '_c' + str(d['num_concepts']) + '_optlr_' + str(d['opt_lr']) + '_optwd_' + str(d['opt_weight_decay']) + '_l1lambda_' + str(d['l1_lambda']) + '_cossimlambda_' + str(d['cos_sim_lambda']) + '.pt'
                            print(d)
                            launch_job(d)
