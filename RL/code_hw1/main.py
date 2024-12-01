from arguments import get_args
from Dagger import DaggerAgent, ExampleAgent, MyAgent
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
from Expert import *

# Traing process with an expert_model guide

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot(record):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    ax.fill_between(record['steps'], record['min'], record['max'],
                    color='blue', alpha=0.2)
    ax.set_xlabel('number of steps')
    ax.set_ylabel('Average score per episode')
    ax1 = ax.twinx()
    ax1.plot(record['steps'], record['query'],
             color='red', label='query')
    ax1.set_ylabel('queries')
    reward_patch = mpatches.Patch(
        lw=1, linestyle='-', color='blue', label='score')
    query_patch = mpatches.Patch(
        lw=1, linestyle='-', color='red', label='query')
    patch_set = [reward_patch, query_patch]
    ax.legend(handles=patch_set)
    fig.savefig('performance.png')


def get_action(action_shape):
    # 检查输入的动作是否是数字，如果不是则提示重新输入
    # 使用一个循环来确保输入是合法的动作
    while True:
        action = input('Please input action: '.format(action_shape - 1))
        try:
            # 尝试将输入转换为整数
            action = int(action)
            # 检查动作是否在有效范围内
            if 0 <= action < action_shape:
                break  # 输入合法，退出循环
            else:
                print("Action out of range! Please input a number between 0 and {}.".format(action_shape - 1))
        except ValueError:
            # 捕获转换失败的情况（输入非整数）
            print("Invalid input! Please enter a valid integer.")
    return action


def collect_expert_labels(data_set, action_shape):
    with open('label.txt', 'a') as f:  # 使用'a'模式来追加写入文件
        for i, obs in enumerate(data_set['data']):
            im = Image.fromarray(obs)
            plt.figure()
            plt.imshow(im)
            plt.show()
            action = get_action(action_shape)  # 让专家手动输入动作
            data_set['label'].append(action)  # 将动作添加到标签列表中
            f.write(str(action) + '\n')  # 将动作写入label.txt文件
            print(f"Action {action} saved for image {i}.")


# the wrap is mainly for speed up the game
# the agent will act every num_stacks frames instead of one frame
class Env(object):
    def __init__(self, env_name, num_stacks):
        self.env = gym.make(env_name)
        # num_stacks: the agent acts every num_stacks frames
        # it could be any positive integer
        self.num_stacks = num_stacks
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        reward_sum = 0
        for stack in range(self.num_stacks):
            obs_next, reward, done, _, info = self.env.step(action)
            reward_sum += reward
            if done:
                self.env.reset()
                return obs_next, reward_sum, done, info
        return obs_next, reward_sum, done, info

    def reset(self):
        return self.env.reset()[0]


def main():
    # load hyper parameters
    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {'steps': [0],
              'max': [0],
              'mean': [0],
              'min': [0],
              'query': [0]}
    # query_cnt counts queries to the expert
    query_cnt = 0

    # environment initial
    envs = make_atari(args.env_name, 4500)
    # action_shape is the size of the discrete action set, here is 18
    # Most of the 18 actions are useless, find important actions
    # in the tips of the homework introduction document
    action_shape = envs.action_space.n
    # observation_shape is the shape of the observation
    # here is (210,160,3)=(height, weight, channels)
    observation_shape = envs.observation_space.shape
    print(action_shape, observation_shape)

    # Expert Model load
    expert = PolicyModel((4, 84, 84), action_shape).to(device)
    checkpoint = torch.load("Expert/params.pth")
    expert.load_state_dict(checkpoint["current_policy_state_dict"])

    # agent initial
    # you should finish your agent with DaggerAgent
    agent = MyAgent()

    # You can play this game yourself for fun
    if args.play_game:
        obs = envs.reset()
        while True:
            im = Image.fromarray(obs)
            im.save('imgs/' + str('screen') + '.jpeg')
            action = int(input('input action'))
            while action < 0 or action >= action_shape:
                action = int(input('re-input action'))
            obs_next, reward, done, _ = envs.step(action)
            obs = obs_next
            if done:
                obs = envs.reset()

    data_set = {'data': [], 'label': []}

    # start train your agent
    stacked_states = np.zeros((84, 84, 4), dtype=np.uint8)

    for i in range(num_updates):
        print("NUM_UPDATES: {}.".format(i + 1))
        if len(data_set['data']) == 10000:
            indice = np.random.choice( len(data_set['label']), 1000, replace=False)
            data_set['data'] = np.delete(data_set['data'], indice, axis=0)

            data = data_set['data']
            data_set['data'] = []
            for arr in data:
                data_set['data'].append(arr)

            data_set['label'] = np.delete(data_set['label'], indice, axis=0).tolist()

        # an example of interacting with the environment
        # we init the environment and receive the initial observation
        obs = envs.reset()
        # we get a trajectory with the length of args.num_steps

        for step in range(args.num_steps):
            # Sample actions
            epsilon = 0.05
            stacked_states = stack_states(stacked_states, obs, True)

            if np.random.rand() < epsilon:
                # we choose a random action
                action = envs.action_space.sample()
                # print(action)
            else:
                # we choose a special action according to our model
                action = agent.select_action(obs)
                if action > 5:
                    action += 5

            expert_action = get_actions_and_values(expert, stacked_states)
            query_cnt += 1

            data_set['data'].append(obs)
            data_set['label'].append(expert_action)

            # interact with the environment
            obs_next, reward, done, _ = envs.step(action)
            # we view the new observation as current observation
            obs = obs_next
            # if the episode has terminated, we need to reset the environment.
            if done:
                envs.reset()

        # design how to train your model with labeled data
        agent.update(data_set['data'], data_set['label'])

        if (i + 1) % args.log_interval == 0:
            total_num_steps = (i + 1) * args.num_steps
            obs = envs.reset()
            reward_episode_set = []
            reward_episode = 0
            # evaluate your model by testing in the environment
            for step in range(args.test_steps):
                action = agent.select_action(obs)
                if action > 5:
                    action += 5
                # you can render to get visual results
                # envs.render()
                obs_next, reward, done, _ = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if step == args.test_steps - 1:
                    reward_episode_set.append(reward_episode)
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0
                    envs.reset()

            end = time.time()
            print(
                "TIME {} Updates {}, num timesteps {}, FPS {} \n query {}, avrage/min/max reward {:.1f}/{:.1f}/{:.1f}"
                .format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(
                        time.time() - start)),
                    i, total_num_steps,
                    int(total_num_steps / (end - start)),
                    query_cnt,
                    np.mean(reward_episode_set),
                    np.min(reward_episode_set),
                    np.max(reward_episode_set)
                ))
            record['steps'].append(total_num_steps)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            record['query'].append(query_cnt)
            plot(record)


if __name__ == "__main__":
    main()
