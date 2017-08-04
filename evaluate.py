from multiprocessing import Process, Queue
from subprocess import run
import subprocess
import json


def evaluation_process(i, files, args, verbose):
    file_name = '{}{}_{}_{}_{}_{}_{}.json'.format(args['folder'],
                                                  args['env'],
                                                  args['alpha'],
                                                  args['sigma'],
                                                  args['episodes'],
                                                  args['batches'], i)
    process_parameters = ['python3', 'train.py',
                          '-e', args['env'],
                          '--alpha', str(args['alpha']),
                          '--sigma', str(args['sigma']),
                          '-p', str(args['episodes']),
                          '-b', str(args['batches']),
                          '-f', file_name]

    if verbose:
        print("Started process {}: {}...".format(i + 1, ' '.join(process_parameters)))
    run(process_parameters, stdout=None if verbose else subprocess.DEVNULL)
    if verbose:
        print("Finished process {}.".format(i + 1))
    files.put(file_name)


def generate_data(n_runs, args, verbose=False):
    # Init processes infrastructure
    processes = []
    files = Queue()

    # Start all processes and wait for finish
    for i in range(n_runs):
        p = Process(target=evaluation_process, args=(i, files, args, verbose))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Extract return values
    f = []
    while not files.empty():
        f.append(files.get())

    return f


def load_data(files):
    runs = []
    for file in files:
        with open(file, 'r') as f:
            run_result = json.load(f)
        runs.append(run_result['history'])
    return runs


def evaluate_args(args):
    files = generate_data(args['n_runs'], args, True)
    return load_data(files)


if __name__ == "__main__":
    args1 = {'alpha': 0.2229,
             'sigma': 0.2542,
             'batches': 75,
             'folder': './evaluation/',
             'env': 'Acrobot-v1',
             'episodes': 30,
             'n_runs': 10}

    args2 = {'alpha': 0.2229,
             'sigma': 0.2542,
             'batches': 75,
             'folder': './evaluation/',
             'env': 'Acrobot-v1',
             'episodes': 20,
             'n_runs': 10}

    args3 = {'alpha': 0.2229,
             'sigma': 0.2542,
             'batches': 75,
             'folder': './evaluation/',
             'env': 'Acrobot-v1',
             'episodes': 15,
             'n_runs': 10}

    args4 = {'alpha': 0.2229,
             'sigma': 0.2542,
             'batches': 75,
             'folder': './evaluation/',
             'env': 'Acrobot-v1',
             'episodes': 10,
             'n_runs': 10}

    args = [args1, args2, args3, args4]
    results = list(map(evaluate_args, args))
    d = [args, results]
    with open('./results/ev_2.json', 'w') as f:
        json.dump(d, f)
