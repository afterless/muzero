import argparse
import logging.config
import os

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir

ray.init()
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="MuZero Implementation in PyTorch")
    parser.add_argument('--env', required=True, help="Name of the environment")
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'), help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case', required=True, choices=['atari'], help="Choose which environment sandbox that which you wish to train a model on domains (default: %(default)s)")
    parser.add_argument('--opr', required=True, choices=['train', 'test'])
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False, 
                        help='If enabled, logs additional values '
                        '(gradients, target_value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False, 
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: %(default)s)')
    parser.add_argument('--value_loss_coeff', type=float, default=None,
                        help='scale for value loss (default: %(default)s)')
    parser.add_argument('--revisit_policy_search_rate', type=float, default=None,
                        help='rate at which the target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--priority_prob_alpha', action='store_true', default=False,
                        help="Decides on probability in Replay Buffer")
    parser.add_argument('--use_max_priority', action='store_true', default=False, 
                        help="Forces max priority assignment for new data in replay buffer")
    parser.add_argument('--use_priority', action='store_true', default=False, 
                        help="Uses priority for data sampling in replay buffer. ")
    parser.add_argument('--use_target_model', action='store_true', default=False, 
                        help="Use target model for bootstrap value estimation (default: %(default)s)")
    parser.add_argument('--test_episodes,', type=int, default=10,
                        help='Evaluation episode count (default: %(default)s)')

    # Process arguments 
    args = parser.parse_args()
    args.device = 'cuda:0' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
    assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
        ' Revisit policy search rate has to be [0, 1]'

    # seeding random generators / initialize ray
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # import corresponding configuration, env_wrapper, and model depending on env choice
    if args.case == 'atari':
        from config.atari import muzero_config
    else:
        raise Exception('Invalid --case option')

    # set config as per arguments
    exp_path = muzero_config.set_config(args)
    exp_path, log_base_path = make_results_dir(exp_path, args)

    init_logger(log_base_path)

    try:
        if args.opr == 'train':
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            train(muzero_config, summary_writer)

        elif args.opr == 'test':
            assert os.path.exists(muzero_config.model_path), 'model not found at {}'.format(muzero_config.model_path)
            model = muzero_config.get_uniform_network().to('cpu')
            model.load_state_dict(torch.load(muzero_config.model_path, map_location=torch.device('cpu')))
            test_score = test(muzero_config, model, args.test_episodes, device='cpu', render=args.render, save_video=True)
            logging.getLogger('test').info('Test Score: {}'.format(test_score))

        else:
            raise Exception('Please select a valid option (--opr) to be performed')

        ray.shutdown()
    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)
