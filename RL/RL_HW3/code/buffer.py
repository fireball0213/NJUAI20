import random
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler

'''
需要调整/重新定⼀个sample函数来
实现Prioritized Replay Buffer。
'''
class RolloutStorage(object):
    def __init__(self, config):
        self.obs = torch.zeros([config.max_buff,  *config.state_shape], dtype=torch.uint8)
        self.next_obs = torch.zeros([config.max_buff,  *config.state_shape], dtype=torch.uint8)
        self.rewards = torch.zeros([config.max_buff,  1])
        self.actions = torch.zeros([config.max_buff, 1])
        self.actions = self.actions.long()
        self.masks = torch.ones([config.max_buff,  1])
        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.num_steps = config.max_buff
        self.step = 0
        self.current_size = 0

    def add(self, obs, actions, rewards, next_obs, masks):
        self.obs[self.step].copy_(torch.tensor(obs[None,:], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.next_obs[self.step].copy_(torch.tensor(next_obs[None,:], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.actions[self.step].copy_(torch.tensor(actions, dtype=torch.float))
        self.rewards[self.step].copy_(torch.tensor(rewards, dtype=torch.float))
        self.masks[self.step].copy_(torch.tensor(masks, dtype=torch.float))
        self.step = (self.step + 1) % self.num_steps
        self.current_size = min(self.current_size + 1, self.num_steps)

    def sample(self, mini_batch_size=None):
        indices = np.random.randint(0, self.current_size, mini_batch_size)
        obs_batch = self.obs[indices]
        obs_next_batch = self.next_obs[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        masks_batch = self.masks[indices]
        return obs_batch, obs_next_batch, actions_batch, rewards_batch, masks_batch


class PrioritizedStorage(object):
    def __init__(self, config):
        self.config = config
        self.state_buffer = torch.zeros([config.max_buff, *config.state_shape], dtype=torch.uint8)
        self.next_state_buffer = torch.zeros([config.max_buff, *config.state_shape], dtype=torch.uint8)
        self.reward_buffer = torch.zeros([config.max_buff, 1])

        self.priority_buffer = torch.zeros(config.max_buff)

        self.action_buffer = torch.zeros([config.max_buff, 1])
        self.action_buffer = self.action_buffer.long()
        self.mask_buffer = torch.ones([config.max_buff, 1])
        self.max_buffer_size = config.max_buff
        self.current_step = 0
        self.buffer_size = 0

    def add(self, state, action, reward, next_state, mask):
        self.state_buffer[self.current_step].copy_(
            torch.tensor(state[None, :], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.next_state_buffer[self.current_step].copy_(
            torch.tensor(next_state[None, :], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.action_buffer[self.current_step].copy_(torch.tensor(action, dtype=torch.float))
        self.reward_buffer[self.current_step].copy_(torch.tensor(reward, dtype=torch.float))
        self.mask_buffer[self.current_step].copy_(torch.tensor(mask, dtype=torch.float))

        self.priority_buffer[self.current_step].copy_(torch.tensor(reward + 100, dtype=torch.float))
        self.current_step = (self.current_step + 1) % self.max_buffer_size
        self.buffer_size = min(self.buffer_size + 1, self.max_buffer_size)

    def sample(self, batch_size=None):
        sorted_priority, indices = torch.sort(self.priority_buffer, descending=True)
        selected_indices = indices[:batch_size]

        state_batch = self.state_buffer[selected_indices]
        next_state_batch = self.next_state_buffer[selected_indices]
        action_batch = self.action_buffer[selected_indices]
        reward_batch = self.reward_buffer[selected_indices]
        mask_batch = self.mask_buffer[selected_indices]

        return state_batch, next_state_batch, action_batch, reward_batch, mask_batch