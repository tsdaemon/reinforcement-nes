from rl import get_space_info, get_policy
import numpy as np
from multiprocessing import Pool


class NESOptimizer(object):
    def __init__(self, env, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma
        self.obs_space = get_space_info(env.observation_space)
        self.action_space = get_space_info(env.action_space)
        self.policy = get_policy(self.obs_space, self.action_space)

    def optimize_parallel(self, envs, n_batches, n_episodes_in_batch, verbose=False, stop_on_std_zero=True):
        """ Performs parallel optimization for given environments """

        # Prepare weights
        w = np.random.rand(self.obs_space['n'] * self.action_space['n'])
        # Number of simultaneously running threads are equal to number of environments passed
        n_threads = len(envs)
        pool = Pool(processes=n_threads)
        # Obtained reward stored to evaluate total performance
        reward_history = []
        rewards = np.zeros(n_episodes_in_batch)

        for j in range(n_batches):
            # Generates random noise for each parameter
            N = np.random.normal(scale=self.sigma, size=(n_episodes_in_batch, w.shape[0]))

            def run(n_thread, n_episode):
                env = envs[n_thread]
                w_try = w + N[n_episode]
                reward, steps, _ = self._run_episode(env, w_try)
                rewards[n_episode] = reward

            # Evaluates all sets of parameters in parallel
            episodes_done = 0
            while episodes_done < n_episodes_in_batch:
                episodes_to_run = min(n_threads, n_episodes_in_batch-episodes_done)
                pool.map(lambda n_thread: run(n_thread, episodes_done+n_thread), range(episodes_to_run))
                episodes_done += episodes_to_run

            reward_history.append(np.mean(rewards))

            w, stop = self._update_w(rewards, stop_on_std_zero, n_episodes_in_batch, N, w, n_episodes_in_batch)

            if stop:
                break

            if verbose:
                print("Batch {}/{}, reward mean {}, reward standard deviation {}".format(j + 1, n_batches, m, std))

        return w, reward_history

    def optimize(self, env, n_batches, n_episodes_in_batch, verbose=False, stop_on_std_zero=True):
        """ Performs parallel optimization for given environments """

        # Prepare weights
        w = np.random.rand(self.obs_space['n'] * self.action_space['n'])
        # Obtained reward stored to evaluate total performance
        reward_history = []
        rewards = np.zeros(n_episodes_in_batch)

        for j in range(n_batches):
            # Generates random noise for each parameter
            N = np.random.normal(scale=self.sigma, size=(n_episodes_in_batch, w.shape[0]))

            for i in range(n_episodes_in_batch):
                w_try = w + N[i]
                reward = self._run_episode(env, w_try)
                rewards[i] = reward
            reward_history.append(np.mean(rewards))

            w, stop = self._update_w(rewards, stop_on_std_zero, n_episodes_in_batch, N, w, n_episodes_in_batch)

            if stop:
                break

            if verbose:
                print("Batch {}/{}, reward mean {}, reward standard deviation {}".format(j + 1, n_batches, m, std))

        return w, reward_history

    def _update_w(self, rewards, stop_on_std_zero, N, w, n_episodes_in_batch):
        # If no difference in reward, probably optimization reached the minimum
        std = np.std(rewards)
        m = np.mean(rewards)
        if std == 0 and stop_on_std_zero: return w, True
        # Reward transformed to weights
        A = (rewards - m) / std
        # Finally weights are changed on a vector of weighted sum of
        w += self.alpha / (n_episodes_in_batch * self.sigma) * np.dot(N.T, A)
        return w, False

    def _run_episode(self, env, w):
        """ Evaluates single episode for given environment and given parameters w """
        done = False
        observation = env.reset()
        ep_reward = 0
        while not done:
            action = self.policy(w, observation)
            observation, reward, done, _ = env.step(action)
            ep_reward += reward

        return ep_reward
