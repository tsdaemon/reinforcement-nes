from rl import get_space_info, get_policy
import numpy as np


class NESOptimizer(object):
    def __init__(self, env, alpha, sigma, elite_set):
        self.alpha = alpha
        self.sigma = sigma
        self.elite_set = elite_set
        self.obs_space = get_space_info(env.observation_space)
        self.action_space = get_space_info(env.action_space)
        self.policy = get_policy(self.obs_space, self.action_space)

    def optimize(self, env, n_batches, n_episodes_in_batch,
                 verbose=False, render=False):
        """ Performs optimization for given environment """
        if verbose:
            self._print_start()

        # Prepare weights
        w = np.zeros(self.obs_space['n'] * self.action_space['n']) # np.random.rand(self.obs_space['n'] * self.action_space['n'])
        # Obtained reward stored to evaluate total performance
        reward_history = []
        rewards = np.zeros(n_episodes_in_batch)

        for j in range(n_batches):
            # Generates random noise for each parameter
            N = np.random.normal(scale=self.sigma, size=(n_episodes_in_batch-1, w.shape[0]))
            # Use previous set of parameters as a last set of parameters for this batch to prevent negative changes
            N = np.vstack((N, np.zeros(w.shape[0])))

            # Evaluate the changes
            for i in range(n_episodes_in_batch):
                w_try = w + N[i]
                reward = self._run_episode(env, w_try, render)
                rewards[i] = reward
            reward_history.append(np.mean(rewards))

            w, stop = self._update_w(rewards, N, w, n_episodes_in_batch)

            if verbose:
                print("Batch {}/{}, reward mean {}, reward standard deviation {}".format(j + 1, n_batches, np.mean(rewards), np.std(rewards)))

            if stop:
                break

        return w, reward_history

    def _print_start(self):
        print("Started oprimization for environment with {} {} observation parameters and {} {} actions"
              .format(self.obs_space['n'], "discrete" if self.obs_space['discrete'] else "continuous",
                      self.action_space['n'], "discrete" if self.action_space['discrete'] else "continuous"))

    def _update_w(self, rewards, N, w, n_episodes_in_batch):
        std = np.std(rewards)
        m = np.mean(rewards)
        # If no difference in rewards, nothing to change
        if std == 0: return w
        # Reward transformed to weights
        A = (rewards - m) / std
        # Weights are changed on a vector of weighted sum of
        w += self.alpha / (n_episodes_in_batch * self.sigma) * np.dot(N.T, A)
        return w

    def _run_episode(self, env, w, render=False):
        """ Evaluates single episode for given environment and given parameters w """
        done = False
        observation = env.reset()
        ep_reward = 0
        w = w.reshape(self.obs_space['n'], self.action_space['n'])
        while not done:
            action = self.policy(w, observation)
            if render:
                env.render()
            observation, reward, done, _ = env.step(action)
            ep_reward += reward

        return ep_reward
