# -*- coding: UTF-8 -*- #
"""
@filename:utils.py
@author:201300086
@time:2023-04-11
"""
import copy
from matplotlib import colors, pyplot as plt
import matplotlib
import time
from init import init_args, reset_args, run

def get_gamma_score_list(args, gammas):
    a=[]
    for gamma in gammas:
        test_args = copy.deepcopy(args)
        test_args.gamma = gamma
        test_args = reset_args(test_args)
        a.append(run(test_args))
    return a


def my_plot_parameters(xlabel, l, i):
    plt.legend(fancybox=True, framealpha=0, fontsize=17, handletextpad=0.1, handlelength=1, columnspacing=0.5,
               markerscale=1.3, labelspacing=0.3)
    plt.ylabel(r'$\epsilon$', fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylim([0, 1.1])
    plt.grid()
    #plt.tight_layout()
    plt.savefig('figure/test-{}-{}.png'.format(l,int(i) + 1), bbox_inches='tight', dpi=150)
    plt.close()

def record_time(func):
    def wrapper(*args, **kwargs):  # 包装带参函数
        start_time = time.perf_counter()
        a = func(*args, **kwargs)  # 包装带参函数
        end_time = time.perf_counter()
        print('time=', end_time - start_time)
        return a  # 有返回值的函数必须给个返回

    return wrapper
