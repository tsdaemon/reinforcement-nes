import numpy as np


def get_space_info(space):
    """ Extract parameters of gym environment """
    discrete = hasattr(space, 'n')
    if discrete:
        n = space.n
    else:
        n = len(space.sample())
    return {'n': n, 'discrete': discrete}


def get_policy(obs_space_info, action_space_info):
    def linear_policy(w, observation):
        """ continuous observations and actions """
        return np.outer(observation, w)

    def sigmoid_policy(w, observation):
        """ continuous observations, discrete actions """
        return np.argmax(np.outer(observation, w))

    def one_hot_sigmoid_policy(w, observation):
        one_hot = np.eye(obs_space_info['n'])[observation]
        return np.argmax(np.outer(one_hot, w))

    if not obs_space_info['discrete'] and not action_space_info['discrete']:
        return linear_policy

    elif not obs_space_info['discrete'] and action_space_info['discrete']:
        return sigmoid_policy

    else:
        return one_hot_sigmoid_policy

