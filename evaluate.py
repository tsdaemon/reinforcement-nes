from multiprocessing import Process
from subprocess import run
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse


def evaluation_process(i, args):
    file_name = '{}{}_{}_{}_{}_{}.json'.format(args['folder'], args['env'], args['alpha'], args['sigma'], args['batches'], i)
    process_parameters = ['python3', 'train.py',
                          '-e', args['env'],
                          '--alpha', str(args['alpha']),
                          '--sigma', str(args['sigma']),
                          '-b', str(args['batches']),
                          '-f', file_name]
    print("Started process {}: {}...".format(i+1, ' '.join(process_parameters)))
    result = run(process_parameters)
    print("Finished process {}: {}.".format(i+1, result))
    return file_name


def generate_data(n_runs, args):
    processes = []

    for i in range(n_runs):
       p = Process(target=evaluation_process, args=(i, args))
       p.start()
       processes.append(p)

    for p in processes:
       p.join()

    print(processes)


def load_data(folder):
    runs = []
    for file in os.listdir(folder):
        with open(folder+file, 'r') as f:
            run_result = json.load(f)
        runs.append(run_result['history'])
    return np.array(runs)


if __name__ == "__main__":
    args = {'alpha': 0.1,
            'sigma': 0.1,
            'batches': 75,
            'folder': './evaluation/',
            'env': 'Acrobot-v1',
            'n_runs': 10}

    generate_data(args['n_runs'], args)
    result = load_data(args['folder'])
    sns.tsplot(data=result)
    plt.show()