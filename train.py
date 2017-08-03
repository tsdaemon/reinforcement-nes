import argparse
from nes import NESOptimizer
import gym
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--n-threads', default=4, type=int,
                    help="Number of threads to use")

parser.add_argument('-p', '--n-episodes', default=10, type=int,
                    help="Number of episodes in batch")

parser.add_argument('-e', '--env-id', type=str, default="CartPole-v1",
                    help="Environment id")

parser.add_argument('-v', '--verbose', type=bool, default=True,
                    help="Show or not intermediate results")

parser.add_argument('-d', '--std-zero', type=bool, default=False,
                    help="Stop or not if reward standard deviation becomes zero")

parser.add_argument('-a', '--alpha', type=float, default=0.1, help="Learning rate")

parser.add_argument('-s', '--sigma', type=float, default=0.1, help="Noise scale")

parser.add_argument('-b', '--n-batches', type=int, default=100,
                    help="Number of batches")


def train_parallel(args):
    envs = [gym.make(args.env_id) for i in range(args.n_threads)]
    nes = NESOptimizer(envs[0], args.alpha, args.sigma)
    w, history = nes.optimize_parallel(envs, args.n_batches, args.n_episodes, args.verbose, args.std_zero)
    return history


def train(args):
    env = gym.make(args.env_id)
    nes = NESOptimizer(env, args.alpha, args.sigma)
    w, history = nes.optimize(env, args.n_batches, args.n_episodes, args.verbose, args.std_zero)
    return history


if __name__ == "__main__":
    args = parser.parse_args()

    if args.n_threads == 1:
        reward_history = train(args)
    else:
        reward_history = train_parallel(args)

    plt.plot(reward_history)
    plt.show()




