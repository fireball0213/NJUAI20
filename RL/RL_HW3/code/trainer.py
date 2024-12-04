import math
import matplotlib.pyplot as plt
import os
import numpy as np
from config import Config
# from core.logger import TensorBoardLogger
from core.util import get_output_folder
import time
import imageio
from PIL import Image
import csv
class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)
        print(self.outputdir)

    def train(self, pre_fr=0):
        losses = []
        all_rewards = []

        rewards = []
        step = []
        losss = []

        episode_reward = 0
        ep_num = 0
        is_win = False
        start = time.time()
        state = self.env.reset()

        for fr in range(pre_fr + 1, self.config.frames + 1):
            if fr % self.config.gif_interval >= 1 and fr % self.config.gif_interval <= 200:
                if fr % self.config.gif_interval == 1:
                    frames = []
                img = state[0, 0:3].transpose(1, 2, 0).astype('uint8')
                frames.append(Image.fromarray(img).convert('RGB'))
                if fr % self.config.gif_interval == 200:
                    imageio.mimsave('record.gif', frames, 'GIF', duration=0.1)

            epsilon = self.epsilon_by_frame(fr)
            action = self.agent.act(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)

            reward = np.nan_to_num(reward)  # 将任何NaN或无穷大的奖励转换为0

            self.agent.buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            if np.isnan(episode_reward):
                print(f"Warning: episode_reward is NaN at frame {fr}")

            loss = 0
            if fr > self.config.init_buff and fr % self.config.learning_interval == 0:
                loss = self.agent.learning(fr)
                losses.append(loss)

            if fr % self.config.print_interval == 0:
                if len(all_rewards) >= 10:
                    rewards.append(np.mean(all_rewards[-10:]))
                else:
                    rewards.append(np.mean(all_rewards))
                losss.append(loss)
                step.append(fr // 1000)

                print(
                    "TIME {}  num timesteps {}, FPS {} Loss {:.3f}, average reward {:.1f}"
                    .format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                            fr,
                            int(fr / (time.time() - start)),
                            loss, np.mean(all_rewards[-10:])))

            if fr % self.config.gif_interval == 0:
                self.plot_progress(step, rewards, losss)

            if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                self.agent.save_checkpoint(fr, self.outputdir)

            if done:
                state = self.env.reset()
                if episode_reward == 0:
                    print(f"Warning: episode_reward is 0 at step {fr}")
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[
                    -1] > self.config.win_reward:
                    is_win = True
                    self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (
                    ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')

        self.save_final_results(step, rewards, losss)

    def plot_progress(self, step, rewards, losss):
        """绘制并显示奖励和损失图像"""
        # 绘制奖励曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.xlabel('Frame')
        plt.ylabel('Reward')
        plt.title('Reward vs. Frame')
        plt.plot(step, rewards,
                 label=f"Dueling:{self.config.dueling_dqn}  Double:{self.config.double_dqn}  Prioritized:{self.config.prioritized}",
                 color='purple')
        plt.legend()

        # 绘制损失曲线
        plt.subplot(1, 2, 2)
        plt.xlabel('Frame')
        plt.ylabel('Loss')
        plt.title('Loss vs. Frame')
        plt.plot(step, losss,
                 label=f"Dueling:{self.config.dueling_dqn}  Double:{self.config.double_dqn}  Prioritized:{self.config.prioritized}",
                 color='purple')
        plt.legend()

        # 显示图像
        plt.tight_layout()
        plt.show()

    def save_final_results(self, step, rewards, losss):
        """保存最终的图像和CSV文件"""
        self.save_plot(step, rewards, 'Reward')
        self.save_plot(step, losss, 'Loss')
        self.save_csv(step, rewards, losss)

    def save_plot(self, step, values, ylabel):
        """保存图像为PNG文件"""
        plt.figure()
        plt.xlabel('Frame')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs. Frame')
        plt.plot(step, values,
                 label=f"Dueling:{self.config.dueling_dqn}  Double:{self.config.double_dqn}  Prioritized:{self.config.prioritized}",
                 color='purple')
        plt.legend()
        plt.savefig(
            f'{ylabel.lower()}_performance_Duelilng_{self.config.dueling_dqn}_Double_{self.config.double_dqn}_Prioritized_{self.config.prioritized}.png')
        plt.close()

    def save_csv(self, step, rewards, losss):
        """保存结果到CSV文件"""
        with open(
                f'output_Duelilng_{self.config.dueling_dqn}_Double_{self.config.double_dqn}_Prioritized_{self.config.prioritized}.csv',
                mode='w', newline='') as file:
            writer = csv.writer(file)
            for s, r, l in zip(step, rewards, losss):
                writer.writerow([s, r, l])