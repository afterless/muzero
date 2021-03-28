from typing import Optional
import math
import collections

import torch
import numpy as np

from .game import ActionHistory

# Below we define the core MCTS that MuZero uses to find the best course of actions
# We run N simulations always starting at the root (clean reset) and traverse down the root using network
# output and the UCB formula until we reach a leaf node that is yet to be expanded

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object):
    def __init__(self, known_bounds: Optional[KnownBounds]=None):
        self.maximum = known_bounds.max if known_bounds else -float('inf')
        self.minimum = known_bounds.min if known_bounds else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float):
        if self.minimum > self.maximum:
            # We normalize only when we have a set maximum and minimum
            return value
        return (value - self.minimum) / (self.maximum - self.minimum)

class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0 

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, to_play, actions, network_output):
        self.to_play = to_play
        self.hidden_state = network_output.hidden_state
        self.reward = network_output.reward
        # softmax over policy logits
        policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            self.children[action] = Node(p / policy_sum)

    def add_exploration_noise(self, dirichlet_alpha, exploration_faction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_faction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac
 

class MCTS(object):
    def __init__(self, config):
        self.config = config

    def run(self, root, action_history: ActionHistory, model):
        min_max_stats = MinMaxStats(self.config.known_bounds)

        for _ in range(self.config.num_simulations):
            history = action_history.clone()
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                history.add_action(action)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and previous hidden state.
            parent = search_path[-2]
            network_output = model.recurrent_inference(parent.hidden_state, history.last_action())

            node.expand(history.to_play(), history.action_space(), network_output)

            self.backpropagate(search_path, network_output.value, history.to_play(), min_max_stats)

    def select_child(self, node, min_max_stats):
        _, action, child = max((self.ucb_score(node, child, min_max_stats), action, child)
                               for action, child in node.children.items())
        return action, child

    def ucb_score(self, parent, child, min_max_stats):
        pb_c = math.log(
            (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init   

        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        for node in search_path:
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + self.config.discount * value
