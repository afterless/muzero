from collections import deque

import torch
import numpy as np

from core.game import Game
from core.game import Action

class AtariControlWrapper(Game):
    def __init__(self, env, k, discount: float):
        """
        env: Gym environmnet
        """
        super().__init__(env, env.action_space.n, discount)
        self.frames = deque([], maxlen=k)

    def legal_actions(self):
        return [Action(i) for i in range(self.env.action_space.n)]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.history.append(action)
        self.obs_history.append(obs)
        self.rewards.append(reward)

        return self.obs(len(self.rewards)), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.rewards = []
        self.obs_history = []
        self.history = []

        self.obs_history.append(obs)

        return self.obs(0)

    def obs(self, i):
        return self.obs_history[i]

    def close(self):
        self.env.close()
