import argparse

# import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--env-name',
        type=str,
        default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument(
        '--num-stacks',
        type=int,
        default=8)
    parser.add_argument(
        '--num-steps',
        type=int,
        default=400)
    parser.add_argument(
        '--test-steps',
        type=int,
        default=2000)
    parser.add_argument(
        '--num-frames',
        type=int,
        default=100000)

    # other parameter
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-img',
        type=bool,
        default=True)
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--play-game',
        type=bool,
        # default=False)
        default = True)
    args = parser.parse_args()

    return args
