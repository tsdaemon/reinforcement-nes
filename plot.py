import matplotlib.pylab as plb
import seaborn as sns
import json
import numpy as np

if __name__ == "__main__":
    args = {
        'file': './results/ev_2.json',
        'output_file': './results/ev_2.png',
        'unit': 'episodes'
    }

    with open(args['file'], 'r') as f:
        a_args, results = json.load(f)

    a_args = a_args[:3]
    results = results[:3]
    legend = ['{}: {}'.format(args['unit'], arg[args['unit']]) for arg in a_args]

    data = np.moveaxis(np.array(results), 0, -1)

    sns.tsplot(data=data, value='reward', condition=legend)
    plb.savefig(args['output_file'])
