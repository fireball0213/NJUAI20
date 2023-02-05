# -*- coding: UTF-8 -*- #
"""
@filename:plot.py
@author:201300086
@time:2023-02-01
"""
from model.dataset import load_h5
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def plot_scatters(y, pos, lable):
    colors = ['darkred', 'green', 'pink', 'yellow',
              'blueviolet', 'deepskyblue', 'grey', 'orange',
              'deeppink', 'gold', 'lightblue', 'lightyellow']
    assert (len(colors) >= len(lable))
    for i in range(len(lable)):
        plt.scatter(pos[y == lable[i], 0],  # 横坐标
                    pos[y == lable[i], 1],  # 纵坐标
                    c=colors[i],
                    label=lable[i])
    # plt.legend()


if __name__ == "__main__":
    file_path = "spca_dat/sample_151510.h5"

    # 真实标记
    X, y, pos, lable = load_h5(file_path)
    plot_scatters(y, pos, lable)
    plt.show()
