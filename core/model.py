# TODO:
# Combine all the networks into one large class, with a function for both initial_reference()
# along with recurrent_inference(). Include potentially the reverse transform revision posed in 
# the MuZero paper
# Implement Reward into the dynamics function
# Make sure that NetworkOutput is returned 
# Make sure to ensure that the action_logit will need to be passed through a softmax

from typing import NamedTuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkOuput(NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[int, float]
    hidden_state: List[float]

class BaseMuZeroNet(nn.Module):
    def __init__(self, inverse_value_transform, inverse_reward_transform):
        super(BaseMuZeroNet, self).__init__()
        self.inverse_value_transform = inverse_value_transform 
        self.inverse_reward_transform = inverse_reward_transform

    # Representation Network
    def representation(self, obs_history):
        raise NotImplementedError

    # Dynamics Network
    def dynamics(self, state, action):
        raise NotImplementedError

    # Prediction network
    def prediction(self, state):
        raise NotImplementedError

    def initial_reference(self, obs) -> NetworkOuput:
        # representation + prediction function
        state = self.representation(obs)
        actor_logit, value = self.prediction(state)

        if not self.training:
            value = self.inverse_value_transform(value)

        return NetworkOuput(value, 0, actor_logit, state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOuput:
        # dynamics + prediction function
        state, reward = self.dynamics(hidden_state, action)
        actor_logit, value = self.prediction(state)

        if not self.training:
            value = self.inverse_value_transform(value)
            reward = self.inverse_reward_transform(reward)

        return NetworkOuput(value, reward, actor_logit, state)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
