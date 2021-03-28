import numpy as np
import ray
import torch

from .model import BaseMuZeroNet as Network

# The ReplayBuffer uses a different setup compared to the MuZero paper, where we use a prioiritzed experience replay 
# rather than choosing at random
@ray.remote
class ReplayBuffer(object):
    def __init__(self, config, prob_alpha):
        self.batch_size = config.batch_size
        self.window_size = config.window_size

        self.buffer = []
        self.priorities = []
        self.game_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self.prob_alpha = prob_alpha

    def save_game(self, game, priorities=None):
        if priorities is None:
            max_prior = self.priorities.max() if self.buffer else 1
            self.priorities = np.concatenate((self.priorities, [max_prior for _ in range(len(game))]))
        else:
            assert len(game) == len(priorities) # priorities should be the same length of game steps
            self.priorities = np.concatenate((self.priorities, priorities))

        self.buffer.append(game)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(game))]
        self._eps_collected += 1

    def sample_batch(self, num_unroll_steps: int, td_steps: int, beta: float = 1, model=None, config=None):
        obs_batch, action_batch, reward_batch, value_batch, policy_batch = [], [], [], [], []

        probs = np.array(self.priorities) ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.priorities), self.batch_size, p=probs)

        total = len(self.priorities)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        indices = torch.tensor(indices)
        weights = torch.tensor(weights).float()

        for idx in indices:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]
            _actions = game.history[game_pos:game_pos+num_unroll_steps]
            # randomly sample action to complete num_unroll_steps
            _actions += [np.random.randint(0, game.action_space_size) for _ in range(num_unroll_steps - len(_actions))]

            obs_batch.append(game.obs(game_pos))
            action_batch.append(_actions)
            value, reward, policy = game.make_target(game_pos, num_unroll_steps, td_steps, model, config)
            reward_batch.append(reward)
            value_batch.append(value)
            policy_batch.append(policy)

        obs_batch = torch.tensor(obs_batch).float()
        action_batch = torch.tensor(action_batch).float()
        reward_batch = torch.tensor(reward_batch).float()
        value_batch = torch.tensor(value_batch).float()
        policy_batch = torch.tensor(policy_batch).float()

        return obs_batch, action_batch, reward_batch, value_batch, policy_batch, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prior in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prior

    def remove_to_fit(self):
        if self.size() > self.window_size:
            num_excess_games = self.size() - self.window_size
            excess_game_steps = sum([len(game) for game in self.buffer[:num_excess_games]])
            del self.buffer[:num_excess_games]
            self.priorities = self.priorities[excess_game_steps:]
            del self.game_look_up[:excess_game_steps]
            self.base_idx += num_excess_games

    def size(self):
        return len(self.buffer)

    def episodes_collected(self):
        return self._eps_collected

    def get_priorities(self):
        return self.priorities
