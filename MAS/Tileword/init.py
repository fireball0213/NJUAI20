# -*- coding: UTF-8 -*- #
"""
@filename:init.py
@author:201300086
@time:2023-04-11
"""
import argparse
import numpy as np
from numpy import random
from tqdm import tqdm
from environment import Agent, TileWorld


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=[0], type=list)  # [5, 2, 0]
    parser.add_argument("--p", default=1, type=float,
                        help="Clock cycle of the agent planning.")
    parser.add_argument("--k", default=4, type=int,
                        help="Agent reconsider its plan every k steps.")
    parser.add_argument("--m", default=1, type=int,
                        help="Clock cycle of the agent action.")
    parser.add_argument("--obstacles", default=[(0, 1), (3, 1), (8, 8), (8, 9), (7, 7), (
        4, 12), (13, 2), (10, 7), (14, 4), (7, 13)], type=list, help="positions of obstacles.")
    parser.add_argument("--l_min", default=240, type=int,
                        help="Minimum of the life-expectancy.")
    parser.add_argument("--l_max", default=960, type=int,
                        help="Maximum of the life-expectancy.")
    parser.add_argument("--g_min", default=60, type=int,
                        help="Minimum of the gestation.")
    parser.add_argument("--g_max", default=240, type=int,
                        help="Maximum of the gestation.")
    parser.add_argument("--s_min", default=1, type=int,
                        help="Minimum score of a hole.")
    parser.add_argument("--s_max", default=10, type=int,
                        help="Maximum score of a hole.")
    parser.add_argument("--d_coef", default=-1, type=int,
                        help="Coefficient of the distance when computing utility.")
    parser.add_argument("--a_coef", default=-1, type=int,
                        help="Coefficient of the age when computing utility.")
    parser.add_argument("--s_coef", default=5, type=int,
                        help="Coefficient of the score when computing utility.")
    parser.add_argument("--grid_l", default=15, type=int,
                        help="The length of the grid.")
    parser.add_argument("--gamma", default=1, type=int,
                        help="The Rate of changes of the world.")
    parser.add_argument("--iterations", default=5000, type=int,
                        help="The number of iterations of the simulation.")
    parser.add_argument("--reaction_strategy", default="blind", type=str)
    args = parser.parse_args()

    args = reset_args(args)
    return args


def reset_args(args):
    args.l_min = int(args.l_min / args.gamma)
    args.l_max = int(args.l_max / args.gamma)
    args.g_min = int(args.g_min / args.gamma)
    args.g_max = int(args.g_max / args.gamma)
    return args


def run(args):
    epsilons = []
    for seed in args.seed:
        random.seed(seed)
        env = TileWorld(args)
        agent = Agent(args, env)
        bar = tqdm(range(args.iterations))
        for iteration in bar:
            action = agent.update(env)
            score = env.update(action)
            agent.score += score
            bar.set_description(
                "Processing iters: {}/{}".format(iteration, args.iterations))

        print("env score: ", env.total_score, "   agent score: ", agent.score)
        epsilons.append(agent.score / env.total_score)
    return np.mean(epsilons)
