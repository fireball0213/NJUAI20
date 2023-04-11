# -*- coding: UTF-8 -*- #
"""
@filename:environment.py
@author:201300086
@time:2023-04-10
"""
from copy import deepcopy

import numpy as np
from numpy import random
from enum import Enum
from queue import Queue


class ACTION(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PLANNING = 4


class Hole():
    def __init__(self, args, x, y) -> None:
        self.life_expectancy = random.randint(args.l_min, args.l_max+1)
        self.score = random.randint(args.s_min, args.s_max+1)
        self.age = 0
        self.x = x
        self.y = y


class TileWorld():
    def __init__(self, args) -> None:
        self.args = args
        self.grid_l = args.grid_l
        self.holes = []
        self.map = np.zeros((args.grid_l, args.grid_l), dtype=int).tolist()
        self.obstacles = args.obstacles
        self.g_min = args.g_min
        self.g_max = args.g_max
        self.a_x, self.a_y = None, None
        self.total_score = 0
        for x, y in self.obstacles:
            self.map[x][y] = 1
        self.g_time = 0

    def update(self, action) -> int:
        if self.g_time == 0:
            self._gen_holes()
        else:
            self.g_time -= 1

        if action == ACTION.DOWN:
            self.a_x += 1
        elif action == ACTION.UP:
            self.a_x -= 1
        elif action == ACTION.LEFT:
            self.a_y -= 1
        elif action == ACTION.RIGHT:
            self.a_y += 1
        else:
            pass
        assert self.a_x < self.args.grid_l and self.a_x >= 0 and self.a_y < self.args.grid_l and self.a_y >= 0

        score = 0
        for hole in self.holes[:]:
            if hole.age == hole.life_expectancy:
                self.holes.remove(hole)
        for hole in self.holes[:]:
            if hole.x == self.a_x and hole.y == self.a_y:
                score = hole.score
                self.holes.remove(hole)

        for hole in self.holes:
            hole.age += 1

        self.map = np.zeros(
            (self.args.grid_l, self.args.grid_l), dtype=int).tolist()

        for x, y in self.obstacles:
            self.map[x][y] = 1
        for hole in self.holes:
            self.map[hole.x][hole.y] = 2
        self.map[self.a_x][self.a_y] = 3
        return score


    def _gen_holes(self):
        x = random.randint(0, self.grid_l)
        y = random.randint(0, self.grid_l)
        while (x, y) in self.obstacles:
            x = random.randint(0, self.grid_l)
            y = random.randint(0, self.grid_l)
        self.map[x][y] = 2
        self.holes.append(Hole(self.args, x, y))
        self.total_score += self.holes[-1].score
        self.g_time = random.randint(self.g_min, self.g_max+1)


class Agent():
    def __init__(self, args, env) -> None:
        self.args = args
        self.actions = []
        self.m = args.m
        self.k = args.k
        self.p = args.p
        self.time = 0
        self.step = 0
        self._born(env)
        self.score = 0
        self.target = None
        self.memory_hole = None
        self.reaction_strategy = args.reaction_strategy
        assert self.reaction_strategy == "blind" or self.reaction_strategy == "disapper" or self.reaction_strategy == "any_hole" or self.reaction_strategy == "nearer_hole"

    def update(self, env) -> ACTION:
        if self.time != 0:
            self.time -= 1
            return ACTION.PLANNING
        else:
            if self.step == self.args.k or len(self.actions) == 0 or (self.reaction_strategy != "blind" and self._check(env)):
                self._planning(env)
                return ACTION.PLANNING
            else:
                self.time = self.m-1
                self.step += 1
                return self.actions.pop(0)

    def _check(self, env):
        if env.map[self.target[0]][self.target[1]] != 2:
            return True
        if self.reaction_strategy == "any_hole":
            for hole in env.holes:
                hole = (hole.x, hole.y)
                if hole not in self.memory_hole:
                    return True
        if self.reaction_strategy == "nearer_hole":
            for hole in env.holes:
                hole = (hole.x, hole.y)
                if hole not in self.memory_hole and self._manhattan_distance(hole, (env.a_x, env.a_y)) < self._manhattan_distance(self.target, (env.a_x, env.a_y)):
                    return True
        return False

    def _manhattan_distance(self, x, y):
        return np.sum(np.abs(np.array(x)-np.array(y)))


    def _planning(self, env):
        if len(env.holes) == 0:
            return ACTION.PLANNING
        self.time = self.p-1
        self.step = 0
        self.actions, self.target = self._find_way(env)
        self.memory_hole = []
        for hole in env.holes:
            self.memory_hole.append((hole.x, hole.y))

    def _find_way(self, env):
        mmap = deepcopy(env.map)
        id2actions = [ACTION.UP, ACTION.DOWN, ACTION.RIGHT, ACTION.LEFT]

        holes = []

        h, w = len(mmap), len(mmap[0])

        class Node:
            def __init__(self, x, y, parent, dis, action) -> None:
                self.x = x
                self.y = y
                self.parent = parent
                self.dis = dis
                self.action = action
        node = Node(env.a_x, env.a_y, None, 0, None)
        queue = Queue()
        queue.put(node)
        while not queue.empty():
            node = queue.get()
            for i, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, 1), (0, -1)]):
                x, y = node.x+dx, node.y+dy
                if x >= 0 and x < h and y >= 0 and y < w and mmap[x][y] != 1:
                    new_node = Node(x, y, node, node.dis+1, id2actions[i])
                    queue.put(new_node)
                    if mmap[x][y] == 2:
                        holes.append(new_node)
                    mmap[x][y] = 1

        max_utility = -np.inf
        node = None
        for hole in holes:
            utility = self._utility(hole, env)
            if utility > max_utility:
                max_utility = utility
                node = hole
        target = (node.x, node.y)
        actions = []
        while node.x != env.a_x or node.y != env.a_y:
            actions.append(node.action)
            node = node.parent
        return actions[::-1], target

    def _utility(self, node, env):
        for hole in env.holes:
            if node.x == hole.x and node.y == hole.y:
                utility = self.args.d_coef*node.dis+self.args.s_coef * \
                    hole.score+self.args.a_coef*hole.age
                return utility

    def _born(self, env):

        x = random.randint(0, env.grid_l)
        y = random.randint(0, env.grid_l)
        while (x, y) in env.obstacles:
            x = random.randint(0, self.grid_l)
            y = random.randint(0, self.grid_l)
        env.a_x = x
        env.a_y = y
        env.map[x][y] = 3

