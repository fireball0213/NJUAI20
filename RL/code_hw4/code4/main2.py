from arguments import get_args
from algo import *
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from env import Make_Env
from gym_minigrid.wrappers import *
import scipy.misc
import os
t = str(time.time())


def plot(record, info):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    ax.fill_between(record['steps'], record['min'], record['max'],
                    color='blue', alpha=0.2)
    ax.set_xlabel('number of steps')
    ax.set_ylabel('Average score per episode')
    #显示图像
    plt.show()
    os.makedirs(t + '-{}'.format(info), exist_ok=True)
    fig.savefig(t + '-{}/performance.png'.format(info))
    plt.close()
def main():
    n = 250
    start_planning = 8  # 开始使⽤model based 提⾼样本利⽤率
    h = 0  # ⼀条轨迹执⾏的⻓度
    m = 12  # 转移训练的频率


    # load hyper parameters
    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {'steps': [0],
              'max': [0],
              'mean': [0],
              'min': [0]}

    # environment initial
    envs = Make_Env(env_mode=2)
    action_shape = envs.action_shape
    observation_shape = envs.state_shape
    print(action_shape, observation_shape)

    epsilon = 0.2
    alpha = 0.2
    gamma = 0.99

    agent = Myagent(alpha, gamma)
    dynamics_model = NetworkModel(8, 8, policy=agent)
    count = 0
    print('n:', n, 'start_planning:', start_planning, 'h:', h, 'm:', m, end=' ')
    # start to train your agent
    for i in range(num_updates * 100):
        # an example of interacting with the environment
        obs = envs.reset()
        obs = obs.astype(int)
        for step in range(args.num_steps):
            # Sample actions with epsilon greedy policy

            if np.random.rand() < epsilon:
                action = envs.action_sample()
            else:
                action = agent.select_action(obs)

            # interact with the environment
            obs_next, reward, done, info = envs.step(action)
            obs_next = obs_next.astype(int)
            # add your Q-learning algorithm

            agent.update(obs, action, obs_next, reward, done)

            dynamics_model.store_transition(obs, action, reward, obs_next)
            obs = obs_next

            if done:
                obs = envs.reset()
        if i > start_planning:
            for _ in range(n):
                s, idx = dynamics_model.sample_state()
                # buf_tuple = dynamics_model.buffer[idx]
                for _ in range(h):
                    if np.random.rand() < epsilon:
                        a = envs.action_sample()
                    else:
                        a = agent.select_action(s)
                    s_ = dynamics_model.predict(s, a)
                    r = envs.R(s, a, s_)
                    done = envs.D(s, a, s_)
                    # add your Q-learning algorithm
                    agent.update(s, a, s_, r, done)
                    s = s_
                    if done:
                        break

        for _ in range(m):
            dynamics_model.train_transition(32)

        if (i + 1) % (args.log_interval) == 0:
            total_num_steps = (i + 1) * args.num_steps

            obs = envs.reset()
            obs = obs.astype(int)
            reward_episode_set = []
            reward_episode = 0.
            for step in range(args.test_steps):
                action = agent.select_action(obs)
                obs_next, reward, done, info = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0.
                    obs = envs.reset()

            end = time.time()
            print("TIME {} Updates {}, num timesteps {}, FPS {}  avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                i, total_num_steps, int(total_num_steps / (end - start)),
                np.mean(reward_episode_set),
                np.min(reward_episode_set),
                np.max(reward_episode_set)))
            record['steps'].append(total_num_steps)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            if np.mean(reward_episode_set)>93 and np.min(reward_episode_set)>85:
                count += 1
            else:
                count = 0
            if count==2:
                print("Time: {:.1f}, Step: {}".format(time.time() - start,total_num_steps))
                break
        plot(record, args.info)



def search_main():
    # 初始化变量
    n_values = []  # 用于记录 n
    total_steps_values = []  # 用于记录 total_num_steps
    #遍历参数v从0到500，间隔逐渐增加
    v=0
    step1 = 0  # 变量初始值
    step2 = 1  # 初始间隔
    while v <= 500:
        v = step1  # 增加当前步长
        step1 += 2  # 步长逐渐增大
        step1 += step2  # 步长逐渐增大
        step2+=1
        n=v
        start_planning=v
        m=v
        # n = 250
        start_planning = 8  # 开始使⽤model based 提⾼样本利⽤率
        h = 0  # ⼀条轨迹执⾏的⻓度
        m = 12  # 转移训练的频率


        # load hyper parameters
        args = get_args()
        num_updates = int(args.num_frames // args.num_steps)
        start = time.time()
        record = {'steps': [0],
                  'max': [0],
                  'mean': [0],
                  'min': [0]}

        # environment initial
        envs = Make_Env(env_mode=2)
        action_shape = envs.action_shape
        observation_shape = envs.state_shape
        # print(action_shape, observation_shape)

        # agent initial
        # you should finish your agent with QAgent
        # e.g. agent = myQAgent()
        epsilon = 0.2
        alpha = 0.2
        gamma = 0.99

        agent = Myagent(alpha, gamma)
        dynamics_model = NetworkModel(8, 8, policy=agent)
        count = 0
        print('v:', v, 'n:', n, 'start_planning:', start_planning, 'h:', h, 'm:', m, end=' ')
        # start to train your agent
        for i in range(num_updates * 100):
            # an example of interacting with the environment
            obs = envs.reset()
            obs = obs.astype(int)
            for step in range(args.num_steps):
                # Sample actions with epsilon greedy policy

                if np.random.rand() < epsilon:
                    action = envs.action_sample()
                else:
                    action = agent.select_action(obs)

                # interact with the environment
                obs_next, reward, done, info = envs.step(action)
                obs_next = obs_next.astype(int)
                # add your Q-learning algorithm

                agent.update(obs, action, obs_next, reward, done)

                dynamics_model.store_transition(obs, action, reward, obs_next)
                obs = obs_next

                if done:
                    obs = envs.reset()
            if i > start_planning:
                for _ in range(n):
                    s, idx = dynamics_model.sample_state()
                    # buf_tuple = dynamics_model.buffer[idx]
                    for _ in range(h):
                        if np.random.rand() < epsilon:
                            a = envs.action_sample()
                        else:
                            a = agent.select_action(s)
                        s_ = dynamics_model.predict(s, a)
                        r = envs.R(s, a, s_)
                        done = envs.D(s, a, s_)
                        # add your Q-learning algorithm
                        agent.update(s, a, s_, r, done)
                        s = s_
                        if done:
                            break

            for _ in range(m):
                dynamics_model.train_transition(32)

            if (i + 1) % (args.log_interval) == 0:
                total_num_steps = (i + 1) * args.num_steps

                obs = envs.reset()
                obs = obs.astype(int)
                reward_episode_set = []
                reward_episode = 0.
                for step in range(args.test_steps):
                    action = agent.select_action(obs)
                    obs_next, reward, done, info = envs.step(action)
                    reward_episode += reward
                    obs = obs_next
                    if done:
                        reward_episode_set.append(reward_episode)
                        reward_episode = 0.
                        obs = envs.reset()

                end = time.time()
                # print("TIME {} Updates {}, num timesteps {}, FPS {}  avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
                #     time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                #     i, total_num_steps, int(total_num_steps / (end - start)),
                #     np.mean(reward_episode_set),
                #     np.min(reward_episode_set),
                #     np.max(reward_episode_set)))
                record['steps'].append(total_num_steps)
                record['mean'].append(np.mean(reward_episode_set))
                record['max'].append(np.max(reward_episode_set))
                record['min'].append(np.min(reward_episode_set))
                if np.mean(reward_episode_set)>93 and np.min(reward_episode_set)>85:
                    count += 1
                else:
                    count = 0
                if count==2:
                    print("Time: {:.1f}, Step: {}".format(time.time() - start,total_num_steps))
                    n_values.append(v)  # 记录当前 n
                    total_steps_values.append(total_num_steps)  # 记录当前步数
                    break
        plot(record, args.info)

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, total_steps_values, marker='o', linestyle='-', color='b')
    plt.xlabel('n')
    plt.ylabel('total_num_steps')
    # plt.title('Total Steps vs. n ')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # search_main()
    main()
