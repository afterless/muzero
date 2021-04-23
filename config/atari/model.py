# TODO:
# Use attention in the dynamic network?
# Implement Residual Blocks and test their efficiency
# We might want to relegate all the networks to nn.Module classes?

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import BaseMuZeroNet

class MuZeroNet(BaseMuZeroNet):
    def __init__(self, input_shape, action_space_n, reward_support_size, value_support_size, 
                 inverse_value_transform, inverse_reward_transform):
        super(MuZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform)

        self._representation = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=6),
            nn.LeakyReLU(),
            nn.MaxPool2d([5, 10]),
            nn.Conv2d(1, 1, kernel_size=(6, 10))
        )

        self._dynamics_s = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(1, 1, kernel_size=(2, 1)),
            nn.LeakyReLU()
        )

        self._dynamics_r = nn.Sequential(
            nn.Flatten(),
            nn.Linear(222, 64),
            nn.LeakyReLU(),
            nn.Linear(64, reward_support_size)
        )

        self._prediction_p = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(34, 24), # same goes for here
            nn.Linear(24, action_space_n)
        )

        self._prediction_v = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(34, 24), # same goes for here
            nn.Linear(24, value_support_size),
        )

        self.action_space_n = action_space_n

    # Representation Network
    def representation(self, obs_history):
        obs_history = obs_history.transpose_(1, 3).transpose_(2, 3)
        return self._representation(obs_history)

    # Dynamics Network
    def dynamics(self, state, action):
        assert len(state.shape) == 4
        assert action.shape[1] == 1

        # Concatenate the action vector with the state tensor

        action_one_hot = torch.zeros(size=(action.shape[0], self.action_space_n),
                                     dtype=torch.float32, device=action.device)

        action_one_hot.scatter_(1, action, 1.0)
        action_one_hot = action_one_hot[:, None, None, :]
        
        x = torch.cat((state, action_one_hot), dim=2)

        next_state = self._dynamics_s(x)
        reward = self._dynamics_r(x)

        return next_state, reward

    # Prediction network
    def prediction(self, state):
        policy_logit = self._prediction_p(state)
        value = self._prediction_v(state)
        return policy_logit, value
