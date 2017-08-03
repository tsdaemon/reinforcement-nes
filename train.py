import argparse
from nes import NESOptimizer
import gym
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--n-episodes', default=10, type=int,
                    help="Number of episodes in batch")

parser.add_argument('-e', '--env-id', type=str, default="CartPole-v1",
                    help="Environment id")

parser.add_argument('-a', '--alpha', type=float, default=0.1, help="Learning rate")

parser.add_argument('-s', '--sigma', type=float, default=0.1, help="Noise scale")

parser.add_argument('-b', '--n-batches', type=int, default=100,
                    help="Number of batches")

parser.add_argument('-v', '--verbose', type=bool, default=True,
                    help="Show or not intermediate results")

parser.add_argument('-r', '--render', type=bool, default=False,
                    help="Render environment")

parser.add_argument('-k', '--api-key', type=str, default='',
                    help="API key to upload solution")

parser.add_argument('-f', '--file', type=str, default='',
                    help="File to save results")


def train(args):
    env = gym.make(args.env_id)
    if len(args.api_key) > 0:
        env = gym.wrappers.Monitor(env, './tmp', force=True)

    nes = NESOptimizer(env, args.alpha, args.sigma, args.elite_set)
    w, history = nes.optimize(env, args.n_batches, args.n_episodes, args.verbose, args.render)
    env.close()

    if len(args.api_key) > 0:
        gym.upload('./tmp', api_key=args.api_key)

    return history, w


if __name__ == "__main__":
    args = parser.parse_args()
    reward_history, w = train(args)

    if len(args.file) == 0:
        plt.plot(reward_history)
        plt.show()
    else:
        results = {'history':reward_history, 'weights': w}
        with open(args.file, 'w') as f:
            json.dump(results, f)





