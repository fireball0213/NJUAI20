import numpy as np
import random
from torchvision import datasets, transforms


class Make_Env(object):
    def __init__(self, env_mode=2):
        simple_image = False

        if env_mode == 1:
            pos_man = [0, 0]
            pos_key = [3, 3]
            pos_door = [3, 0]
            self.grid_num = 4
            self.image_size = 32
            self.grid_size = self.image_size // self.grid_num
            self.margin = 0
            self.blocks = [20,
                           21,
                           23]
        elif env_mode == 2:
            pos_man = [0, 0]
            pos_key = [0, 7]
            pos_door = [7, 7]
            self.grid_num = 8
            self.image_size = 64
            self.grid_size = self.image_size // self.grid_num
            self.margin = 0
            self.blocks = [3, 4,

                           20, 22, 25, 26,
                           32, 35,
                           40, 41, 42, 44, 45, 47,

                           61, 63, 64, 65,

                           71, ]

        self.observation_shape = (3, self.image_size, self.image_size)
        self.state_shape = 2
        self.action_shape = 4
        self.use_key = True
        self.get_key = False
        self.get_door = False
        self.done = False
        if simple_image:
            self.image_man = np.zeros((3, self.grid_size, self.grid_size))
            self.image_man[:, :, :] = [[[1]], [[0]], [[0]]]
            self.image_key = np.zeros((3, self.grid_size, self.grid_size))
            self.image_key[:, :, :] = [[[0]], [[1]], [[0]]]
            self.image_door = np.zeros((3, self.grid_size, self.grid_size))
            self.image_door[:, :, :] = [[[0]], [[0]], [[1]]]
            self.image_wall = np.ones((3, self.grid_size, self.grid_size))
            self.image_wall[:] = 0.5
            self.image_init = np.zeros((3, self.image_size + 2 * self.margin, self.image_size + 2 * self.margin))
            self.image_init[:, self.margin:(-self.margin), self.margin:(-self.margin)] = 1

        else:
            self.image_man = np.zeros((3, self.grid_size, self.grid_size))
            self.image_man_block = [[1], [1], [1]]
            self.image_man[:, 2, :] = self.image_man_block
            self.image_man[:, 7, 0:3] = self.image_man_block
            self.image_man[:, 7, 5:8] = self.image_man_block
            self.image_man[:, 2:8, 2] = self.image_man_block
            self.image_man[:, 2:8, 5] = self.image_man_block
            self.image_man[:, 0:6, 3] = self.image_man_block
            self.image_man[:, 0:6, 4] = self.image_man_block

            self.image_key = np.ones((3, self.grid_size, self.grid_size))
            self.image_key_block = [[0], [0.25], [0.5]]
            self.image_key[:, 0, 2:6] = self.image_key_block
            self.image_key[:, 3, 2:6] = self.image_key_block
            self.image_key[:, 1:3, 2] = self.image_key_block
            self.image_key[:, 1:3, 5] = self.image_key_block
            self.image_key[:, 4:8, 2] = self.image_key_block
            self.image_key[:, 4:8, 3] = self.image_key_block
            self.image_key[:, 6:8, 4] = self.image_key_block
            self.image_key[:, 6:8, 5] = self.image_key_block

            self.image_door = np.ones((3, self.grid_size, self.grid_size))
            self.image_door_block = [[0.5], [0.25], [0]]
            self.image_door[:, 0, :] = self.image_door_block
            self.image_door[:, 7, :] = self.image_door_block
            self.image_door[:, :, 0] = self.image_door_block
            self.image_door[:, :, 3] = self.image_door_block
            self.image_door[:, :, 4] = self.image_door_block
            self.image_door[:, :, 7] = self.image_door_block
            self.image_wall = np.ones((3, self.grid_size, self.grid_size))
            self.image_wall[:, 2, :] = 0.5
            self.image_wall[:, 5, :] = 0.5
            self.image_wall[:, 0:2, 4] = 0.5
            self.image_wall[:, 5:, 4] = 0.5
            self.image_wall[:, 2:5, 0] = 0.5
            self.image_init = np.zeros((3, self.image_size + 2 * self.margin, self.image_size + 2 * self.margin))
            self.image_init[:, self.margin:(-self.margin), self.margin:(-self.margin)] = 1

        self.action_set = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        self.length = 0
        self.score = 0

        self.pos_man = pos_man
        self.pos_key_init = pos_key
        self.pos_key = pos_key
        self.pos_door_init = pos_door
        self.pos_door = pos_door

        for pos in self.blocks:
            x = pos // 10
            y = pos % 10
            self.image_init[:,
            (self.margin + x * self.grid_size):(self.margin + (x + 1) * self.grid_size),
            (self.margin + y * self.grid_size):(
                    self.margin + (y + 1) * self.grid_size)] = self.image_wall

    # print(self.pos_man, self.pos_key, self.pos_door)

    def not_wall_position(self, pos_man):
        if pos_man[0] < 0 or pos_man[0] == self.grid_num or pos_man[1] < 0 or pos_man[1] == self.grid_num:
            return 0
        elif pos_man[0] * 10 + pos_man[1] in self.blocks:
            return 0
        else:
            return 1

    def action_sample(self):
        return random.randint(0, self.action_shape - 1)

    def trans(self, action):
        reward = 0
        if self.not_wall_position(self.pos_man + self.action_set[action]):
            self.pos_man = self.pos_man + self.action_set[action]
            if self.get_key:
                self.pos_key = self.pos_man
            else:
                if (not self.use_key):
                    self.get_key = True
                else:
                    if (self.pos_man == self.pos_key_init).all():
                        self.get_key = True
                        reward += 10
            if self.get_key and (self.pos_man == self.pos_door).all():
                self.get_door = True

        reward += -1
        if self.get_door:
            reward += 100
        self.length = self.length + 1
        self.done = self.get_door or (self.length == 50)
        self.score = self.score + reward
        return reward, self.done, self.score

    def get_ob(self, state=-1):
        observ = self.image_init.copy()
        observ[:,
        (self.margin + self.pos_man[0] * self.grid_size):(self.margin + (self.pos_man[0] + 1) * self.grid_size),
        (self.margin + self.pos_man[1] * self.grid_size):(
                    self.margin + (self.pos_man[1] + 1) * self.grid_size)] = self.image_man
        if self.use_key:
            observ[:,
            (self.margin + self.pos_key[0] * self.grid_size):(self.margin + (self.pos_key[0] + 1) * self.grid_size),
            (self.margin + self.pos_key[1] * self.grid_size):(
                        self.margin + (self.pos_key[1] + 1) * self.grid_size)] = self.image_key
        observ[:,
        (self.margin + self.pos_door[0] * self.grid_size):(self.margin + (self.pos_door[0] + 1) * self.grid_size),
        (self.margin + self.pos_door[1] * self.grid_size):(
                    self.margin + (self.pos_door[1] + 1) * self.grid_size)] = self.image_door
        return observ.astype(np.float32)

    def get_state(self):
        return np.append(self.pos_man.astype(np.float32), self.get_key).astype(np.int32)

    def R(self, s, a, s_):
        if (s[2] == 1 or not self.use_key) and (s_[:2] == self.pos_door_init).all():
            r = 99
        elif (s[2] == 0 and self.use_key) and (s_[:2] == self.pos_key_init).all():
            r = 9
        else:
            r = -1
        return r

    def D(self, s, a, s_):
        if (s[2] == 1 or not self.use_key) and (s_[:2] == self.pos_door_init).all():
            done = True
        else:
            done = False
        return done
    def reset(self):
        self.pos_key = self.pos_key_init
        self.pos_door = self.pos_door_init
        self.pos_man = np.random.randint(0, self.grid_num, 2)
        while self.wrong_position() or (not self.not_wall_position(self.pos_man)):
            self.pos_man = np.random.randint(0, self.grid_num, 2)
        self.get_key = False
        self.get_door = False
        self.done = False
        self.length = 0
        self.score = 0

        # observ = self.get_ob()
        observ = self.get_state()
        return observ

    def wrong_position(self):
        if (self.pos_man == self.pos_key).all():
            return True
        elif (self.pos_man == self.pos_door).all():
            return True
        else:
            return False

    def step(self, action):
        reward, done, score = self.trans(action)
        info = self.get_ob()
        observ_next = self.get_state()
        return observ_next, reward, done, info
