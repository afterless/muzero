# TODO:
import torch
import numpy as np

from typing import List

from .model import BaseMuZeroNet

class Player(object):
    def __init__(self, id=1):
        self.id = id

    def __eq__(self, other):
        if not isinstance(other, Player):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id

class Action(object):
    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

class ActionHistory(object):
    """
    Simple History container used inside the MCTS search

    Only used to keep track of the actions executed and perform any
    operations on them
    """

    def __init__(self, history: List[Action], action_space_size):
        self.history = list(history)
        self.action_space_size = action_space_size 

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: int):
        self.history.append(action)

    def last_action(self, action: int) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [a for a in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()

# Below is a Game object that serves as both a logger and wrapper around OpenAI games
# To aid in both recording observations, rewards, etc. along with generating training data

class Game(object):
    def __init__(self, env, discount: float, config=None):
        self.env = env
        self.history = []
        self.rewards = []
        self.obs_history = []
        
        self.child_visits = []
        self.root_values = []

        self.action_space_size = env.action_space.n
        self.discount = discount

        self.config = config

    def legal_actions(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def obs(self, i):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, model):
        # The value target is the discounted root value of the search tree
        # N steps into the future, plus the discounted sum of all rewards until then
        target_values, target_policies, target_rewards = [], [], []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                if model is None:
                    value = self.root_values[bootstrap_index] * self.discount**td_steps
                else:
                    # Appendix H: using a target network optimized for recent parameters provides
                    # fresher, stable, n-step, bootstrapped target for the value function
                    obs = self.obs(bootstrap_index)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    network_output = model.initial_reference(obs)
                    value = network_output.value
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i

            # For simplicity, the network always predicts the most recently received reward
            # The following code ensures that these rewards are aligned in the training
            # process
            if current_index > 0 and current_index <= len(self.root_values):
                last_reward = self.rewards[current_index - 1] 
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                target_values.append(value)
                target_rewards.append(last_reward)
                # Appendix H: using a target network optimized for recent parameters provides
                # better quality policy than the original MCTS search
                # The fresh policy is used as a target for 80% of the updates during MuZero training
                if model is not None and np.random.random() <= self.config.revisit_policy_serach_rate:
                    from mcts import MCTS, Node
                    root = Node(0)
                    obs = self.obs(current_index)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    network_output = model.initial_reference(obs)
                    root.expand(self.to_play(), self.legal_actions(), network_output)
                    MCTS(self.config).run(root, self.action_history(current_index), model)
                    self.store_search_statistics(root)

                target_policies.append(self.child_visits[current_index])
            else:
                # States past the end of the game are treated as absorbing states
                target_values.append(0)
                target_rewards.append(last_reward)
                # Note: Target policy is set to 0 so that no policy loss is calculated for them
                target_policies.append([0 for _ in range(len(self.child_visits[0]))])

        return target_values, target_rewards, target_policies

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (a for a in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0 
            for a in action_space
        ])
        self.root_values.append(root.value())

    def action_history(self, idx=None) -> ActionHistory:
        if idx is None:
            return ActionHistory(self.history, self.action_space_size)
        else:
            return ActionHistory(self.history[:idx], self.action_space_size)

    def to_play(self):
        return Player()

    def __len__(self):
        return len(self.rewards)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)
