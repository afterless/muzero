import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class Network:
    def __init__(cls, config):
        if config.network == "fcn":
            return FCN(
                
            )
        elif config.netwowrk == "resnet":
            return ResNet(

            )
        else:
            raise NotImplementedError("The network can either be fcn or resnet")


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class Model(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    @abstractmethod
    def initial_reference(self, obs):
        pass

    @abstractmethod
    def recurrent_inference(self, state, action):
        pass

    def get_weights(self):
       return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

class FCN(Model):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_policy_layers,
        fc_value_layers,
        fc_dynamics_layers,
        fc_representation_layers,
        support_size
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation = nn.DataParallel(
            mlp(
                observation_shape[0] * observation_shape[1] * observation_shape[2]
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size
            )
        )

        self.dynamics_state = nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )

        self.dynamics_reward = nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_reward_layers,
                self.full_support_size
            )
        )

        self.prediction_policy = nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_policy_layers,
                self.action_space_size
            )
        )

        self.prediction_value = nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_value_layers,
                self.full_support_size
            )
        )


    def prediction(self, state):
        policy_logits = self.prediction_policy()


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output=nn.Identity,
    activation=nn.ReLU
):
    """
    Create multi-layer perceptron based on observation sizes
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

