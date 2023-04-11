# -*- coding: UTF-8 -*- #
"""
@filename:my_test.py
@author:201300086
@time:2023-04-10
"""

from matplotlib import colors, pyplot as plt
import matplotlib
import numpy as np
from utils import get_gamma_score_list, my_plot_parameters, record_time

linew = 2
markerss = 5
log10gammas = np.linspace(0, 2, 20, endpoint=True)
gammas = 10 ** log10gammas


@record_time
def test_gamma(args):  # 随环境变化率影响
    epsilons = get_gamma_score_list(args, gammas)
    for i in range(2):
        if i == 0:
            plt.plot(gammas, epsilons, linestyle='-', marker='o',
                     markersize=markerss, linewidth=linew, color='blue', label='experimental')
            my_plot_parameters(r'$\gamma$', 'gamma', i)
        else:
            plt.plot(log10gammas, epsilons, linestyle='-', marker='o',
                     markersize=markerss, linewidth=linew, color='blue', label='experimental')
            my_plot_parameters(r'$\log_{10}\gamma$', 'gamma', i)


@record_time
def test_p(args):
    for i, k in enumerate([1, 4, 5 * args.grid_l]):
        args.k = k
        epsilons = []
        for p in [1, 2, 4, 8]:
            args.p = p
            epsilons.append(get_gamma_score_list(args, gammas))

        plt.plot(log10gammas, epsilons[0], linestyle='-', marker='+',
                 markersize=markerss, linewidth=linew, color='blue', label='p=1')
        plt.plot(log10gammas, epsilons[1], linestyle='--', marker='*',
                 markersize=markerss, linewidth=linew, color='green', label='p=2')
        plt.plot(log10gammas, epsilons[2], linestyle='-', marker='o',
                 markersize=markerss, linewidth=linew, color='lightpink', label='p=4')
        plt.plot(log10gammas, epsilons[3], linestyle='--', marker='v',
                 markersize=markerss, linewidth=linew, color='gray', label='p=8')

        my_plot_parameters(r'$\log_{10}\gamma$', 'p', i)


@record_time
def test_boldness(args):
    for i, p in enumerate([4, 2, 1]):
        args.p = p
        epsilons = []
        for k in [1, 4, 5 * args.grid_l]:
            args.k = k
            epsilons.append(get_gamma_score_list(args, gammas))

        plt.plot(log10gammas, epsilons[0], linestyle='-', marker='+',
                 markersize=markerss, linewidth=linew, color='blue', label='cautious')
        plt.plot(log10gammas, epsilons[1], linestyle='--', marker='*',
                 markersize=markerss, linewidth=linew, color='green', label='normal')
        plt.plot(log10gammas, epsilons[2], linestyle='-', marker='o',
                 markersize=markerss, linewidth=linew, color='gray', label='bold')
        my_plot_parameters(r'$\log_{10}\gamma$', 'boldness', i)


@record_time
def test_reaction(args):
    args.k = 5 * args.grid_l
    for i, p in enumerate([2, 1]):
        args.p = p
        epsilons = []
        for reaction_strategy in ["blind", "disapper", "any_hole", "nearer_hole"]:
            args.reaction_strategy = reaction_strategy
            epsilons.append(get_gamma_score_list(args, gammas))

        plt.plot(log10gammas, epsilons[0], linestyle='-', marker='+',
                 markersize=markerss, linewidth=linew, color='blue', label='blind')
        plt.plot(log10gammas, epsilons[1], linestyle='--', marker='*',
                 markersize=markerss, linewidth=linew, color='green', label='notices target disappearance')
        plt.plot(log10gammas, epsilons[2], linestyle='-', marker='o',
                 markersize=markerss, linewidth=linew, color='lightpink', label='target dis. or any new hole')
        plt.plot(log10gammas, epsilons[3], linestyle='--', marker='v',
                 markersize=markerss, linewidth=linew, color='gray', label='target dis. or near hole')

        my_plot_parameters(r'$\log_{10}\gamma$', 'reaction', i)

@record_time
def test_disappear(args):
    args.p = 1
    args.reaction_strategy = "disapper"
    epsilons = []
    for k in [1, 4, 5 * args.grid_l]:
        args.k = k
        epsilons.append(get_gamma_score_list(args, gammas))

    plt.plot(log10gammas, epsilons[0], linestyle='-', marker='+',
             markersize=markerss, linewidth=linew, color='blue', label='cautious')
    plt.plot(log10gammas, epsilons[1], linestyle='--', marker='*',
             markersize=markerss, linewidth=linew, color='green', label='normal')
    plt.plot(log10gammas, epsilons[2], linestyle='-', marker='o',
             markersize=markerss, linewidth=linew, color='gray', label='bold')
    my_plot_parameters(r'$\log_{10}\gamma$', 'reaction', '2')
