from arguments import get_args
from algo import QAgent, MyQAgent
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from env import Make_Env
from gym_minigrid.wrappers import *
import scipy.misc
def plot(record):
	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(record['steps'], record['mean'],
	        color='blue', label='reward')
	ax.fill_between(record['steps'], record['min'], record['max'],
	                 color='blue', alpha=0.2)
	ax.set_xlabel('number of steps')
	ax.set_ylabel('Average score per episode')
	reward_patch = mpatches.Patch(lw=1, linestyle='-', color='blue', label='score')
	patch_set = [reward_patch]
	ax.legend(handles=patch_set)
	# 显示图像
	fig.show()
	fig.savefig('performance.png')

class Env(object):
	def __init__(self, env_name, num_stacks):
		self.env = gym.make(env_name)
		# num_stacks: the agent acts every num_stacks frames
		self.num_stacks = num_stacks
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space

	def step(self, action):
		reward_sum = 0
		for stack in range(self.num_stacks):
			obs_next, reward, done, info = self.env.step(action)
			reward_sum += reward
			if done:
				self.env.reset()
				return obs_next, reward_sum, done, info
		return obs_next, reward_sum, done, info

	def reset(self):
		return self.env.reset()

def main():
	# load hyper parameters
	args = get_args()
	num_updates = int(args.num_frames // args.num_steps)
	start = time.time()
	record = {'steps':[0],
	          'max':[0],
	'mean': [0],
	'min': [0]}

	# environment initial
	envs = Make_Env(env_mode=2)
	action_shape = envs.action_shape
	observation_shape = envs.state_shape
	print(action_shape, observation_shape)


	# agent initial
	# you should finish your agent with QAgent
	# e.g. agent = myQAgent()
	# agent = QAgent()
	agent = MyQAgent()


	# start to train your agent
	for i in range(num_updates):
		# an example of interacting with the environment
		obs = envs.reset()
		for step in range(args.num_steps):
			# Sample actions with epsilon greedy policy
			epsilon = 1 - i / 100.0
			if np.random.rand() < epsilon:
				action = envs.action_sample()
			else:
				action = agent.select_action(obs)

			# interact with the environment
			obs_next, reward, done, info = envs.step(action)
			agent.learn(obs, action, reward, obs_next)
			obs = obs_next
			if done:
				envs.reset()

			# an example of saving observations
			if args.save_img:
				scipy.misc.toimage(info, cmin=0.0, cmax=1).save('imgs/example.jpeg')

		# you should finish your Q-learning algorithm here


		if (i+1) % args.log_interval == 0:
			total_num_steps = (i + 1) * args.num_steps
			obs = envs.reset()
			reward_episode_set = []
			reward_episode = 0
			for step in range(args.test_steps):
				action = agent.select_action(obs)
				# you can render to get visual results
				# envs.render()
				obs_next, reward, done, info = envs.step(action)
				reward_episode += reward
				obs = obs_next
				if done:
					reward_episode_set.append(reward_episode)
					reward_episode = 0
					envs.reset()

			end = time.time()
			print(
				"TIME {} Updates {}, num timesteps {}, FPS {} \n avrage/min/max reward {:.1f}/{:.1f}/{:.1f}"
					.format(
				            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
				            i, total_num_steps,
				            int(total_num_steps / (end - start)),
				            np.mean(reward_episode_set),
				            np.min(reward_episode_set),
				            np.max(reward_episode_set)
				            ))
			record['steps'].append(total_num_steps)
			record['mean'].append(np.mean(reward_episode_set))
			record['max'].append(np.max(reward_episode_set))
			record['min'].append(np.min(reward_episode_set))
			plot(record)

if __name__ == "__main__":
	main()


