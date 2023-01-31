# -*- coding: UTF-8 -*- #
"""
@filename:main.py
@author:201300086
@time:2023-01-31
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from model.POSS import POSS_cover, POSS_regression
from model.NSGA_II import NSGA_cover, NSGA_regression, MOEAD_cover, MOEAD_regression

NSGA_regression(300, 10)
MOEAD_regression(300, 10)
POSS_regression(2000, 10)
# NSGA_cover(800,10)
# MOEAD_cover(800,10)
# POSS_cover(5000,10)
plt.show()
