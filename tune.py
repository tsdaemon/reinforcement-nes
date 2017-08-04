from evaluate import generate_data, load_data
import numpy as np
from bayes_opt import BayesianOptimization


def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

if __name__ == "__main__":
    args = {'batches': 75,
            'folder': './evaluation/',
            'env': 'Acrobot-v1',
            'n_runs': 5}

    def evaluate(sigma, alpha, episodes):
        # Set values of tune parameters
        args['sigma'] = sigma
        args['alpha'] = alpha
        args['episodes'] = int(episodes)
        # Perform estimation
        files = generate_data(5, args)
        result = load_data(files)
        # Calculate mean values for training history
        m = np.mean(np.array(result), axis=0)
        assert m.shape[0] == args['batches']
        # Find best 100 episodes window
        window = 100/int(episodes)
        ma = movingaverage(m, window)
        return max(ma)

    # Hyperparameters to optimize
    parameters = {
        'sigma': [0.02, 0.3],
        'alpha': [0.001, 0.1],
        'episodes': [10, 50]
    }

    opt = BayesianOptimization(evaluate, parameters, verbose=1)

    num_iter = 25
    init_points = 5

    result = opt.maximize(init_points=init_points, n_iter=num_iter)
    print(result)