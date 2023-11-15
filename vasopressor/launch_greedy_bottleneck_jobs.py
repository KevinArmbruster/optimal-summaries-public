import os
import re
import time
import argparse
import random
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
FLAGS = parser.parse_args()


directory = FLAGS.dir or "/workdir/optimal-summaries-public/vasopressor/models/mimic-iii/vasopressor/"
if not os.path.exists(directory):
    os.makedirs(directory)


def launch_job(exp, time_limit=None, mem_limit=None):

    job_command = "python3 -u gpu_greedy_top_concepts.py"

    for k, v in exp.items():
        job_command += f" --{k}={v}"

    os.system(job_command)


for r in range(1,3):
    d = {}
    d['split_random_state'] = r
    launch_job(d)

