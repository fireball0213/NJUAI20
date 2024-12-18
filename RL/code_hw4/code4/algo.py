import numpy as np

from abc import abstractmethod
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Dense

class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


class Myagent:
    def __init__(self, lr, gamma):
        super(Myagent, self).__init__()
        self.lr = lr
        self.gamma = gamma
        self.Qtable = np.zeros((8, 8, 2, 4))

    def update(self, ob, action, ob_next, reward, done):
        x, y, key = ob
        x1, y1, key1 = ob_next

        Q_predict = self.Qtable[x, y, key, action]

        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * \
                np.max(self.Qtable[x1, y1, key1, :])

        self.Qtable[x, y, key, action] += self.lr * (Q_target - Q_predict)
        # # '''#改进二：限制Q值的范围
        self.Qtable[x, y, key, action] = np.clip(self.Qtable[x, y, key, action], -100, 100)
        # '''

    def select_action(self, ob):
        x, y, key = ob
        return np.argmax(self.Qtable[x, y, key, :])


class Model:
    def __init__(self, width, height, policy):
        self.width = width
        self.height = height
        self.policy = policy
        pass

    @abstractmethod
    def store_transition(self, s, a, r, s_):
        pass

    @abstractmethod
    def sample_state(self):
        pass

    @abstractmethod
    def sample_action(self, s):
        pass

    @abstractmethod
    def predict(self, s, a):
        pass


class DynaModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.Model = {}

    def store_transition(self, s, a, r, s_):
        s = tuple(s)
        if s not in self.Model.keys():
            self.Model[s] = [[] for _ in range(4)]
            self.Model[s][a] = [r, s_]
        else:
            self.Model[s][a] = [r, s_]

    def sample_state(self):
        idx = np.random.randint(0, len(self.Model.keys()))
        s = list(self.Model.keys())[idx]
        return list(s)

    def sample_action(self, s):
        sample = []
        s = tuple(s)
        for idx in range(4):
            if self.Model[s][idx] != []:
                sample.append(idx)
        return np.random.choice(sample)

    def predict(self, s, a):
        s = tuple(s)
        return self.Model[s][a][1]

    def train_transition(self):
        pass


class NetworkModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
        self.x_next_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x_next')
        self.a_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='a')
        self.r_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        # h1 = tf.layers.dense(tf.concat([self.x_ph, self.a_ph], axis=-1), units=256, activation=tf.nn.relu)
        h1 = Dense(256, activation='relu')(tf.concat([self.x_ph, self.a_ph], axis=-1))
        # h2 = tf.layers.dense(h1, units=256, activation=tf.nn.relu)
        h2 = Dense(256, activation='relu')(h1)
        # self.next_x = tf.layers.dense(h2, units=3, activation=tf.nn.tanh) * 1.3 + self.x_ph
        self.next_x = Dense(3, activation='relu')(h2)* 1.3 + self.x_ph
        self.x_mse = tf.reduce_mean(tf.square(self.next_x - self.x_next_ph))
        self.opt_x = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(self.x_mse)
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.variables_initializer(tf.global_variables()))
        self.buffer = []
        self.sensitive_index = []

    def norm_s(self, s):
        return s

    def de_norm_s(self, s):
        s = np.clip(np.round(s), 0, self.width - 1).astype(np.int32)
        s[2] = np.clip(s[2], 0, 1).astype(np.int32)
        return s

    def store_transition(self, s, a, r, s_):
        s = self.norm_s(s)
        s_ = self.norm_s(s_)
        self.buffer.append([s, a, r, s_])
        if s[-1] - s_[-1] != 0:
            self.sensitive_index.append(len(self.buffer) - 1)

    def train_transition(self, batch_size):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            s, a, r, s_ = self.buffer[idx]
            s_list.append(s)
            a_list.append([a])
            r_list.append(r)
            s_next_list.append(s_)
        # 新增部分
        # '''
        if len(self.sensitive_index) > 0:
            for _ in range(batch_size):
                idx = np.random.randint(0, len(self.sensitive_index))
                idx = self.sensitive_index[idx]
                s, a, r, s_ = self.buffer[idx]
                s_list.append(s)
                a_list.append([a])
                r_list.append(r)
                s_next_list.append(s_)
        # '''


        x_mse = self.sess.run([self.x_mse,  self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]
        return x_mse

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        s, a, r, s_ = self.buffer[idx]
        return self.de_norm_s(s), idx

    def sample_action(self, s):
        return self.policy.select_action(s)

    def predict(self, s, a):
        s_ = self.sess.run(self.next_x, feed_dict={self.x_ph: [s], self.a_ph: [[a]]})
        return self.de_norm_s(s_[0])
