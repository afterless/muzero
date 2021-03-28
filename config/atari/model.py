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

        self.hidden_chan = 1
        self.action_space_n = action_space_n

        #import pdb; pdb.set_trace()

        self._dynamics_s = nn.Sequential(
            nn.Conv2d(self.hidden_chan, self.hidden_chan, kernel_size=(2, 2))
        )

        self._dynamics_r = nn.Sequential(
            nn.Conv2d(self.hidden_chan, self.hidden_chan, kernel_size=(2, 2))
        )

        self._prediction_p = nn.Sequential(

        )

        self._prediction_v = nn.Sequential(
            nn.Conv2d(self.hidden_chan, self.hidden_chan, kernel_size=[4, 4])
        )


    # Representation Network
    def representation(self, obs_history):
        obs_history = obs_history.transpose_(1, 3).transpose_(2, 3)
        return self._representation(obs_history)

    # Dynamics Network
    def dynamics(self, state, action):
        assert len(state.shape) == 4

        # Concatenate the action vector with the state tensor

        action_one_hot = torch.zeros(size=(action.shape, state.shape[2]),
                                     dtype=torch.float32, device=action.device)
        action_one_hot.scatter_()

        x = torch.cat((state, action_one_hot), dim=1)

        next_state = self._dynamics_s(x)
        reward = self._dynamics_r(x)

        return next_state, reward

    # Prediction network
    def prediction(self, state):
        policy_logit = self._prediction_p(state)
        value = self._prediction_v(state)

        return policy_logit, value


# Helper Modules / functions

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))
        return F.relu(residual + x)

def get_output_shape(model, tensor_dim):
    with torch.no_grad():
        return model(torch.rand(*(tensor_dim))).data.shape
