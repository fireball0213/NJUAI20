import argparse
import os
import random
import torch
from torch.optim import Adam
from tester import Tester
from buffer import RolloutStorage,PrioritizedStorage
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from config import Config
from core.util import get_class_attr_val
from model import *
from trainer import Trainer
import numpy as np
import ale_py
roms_path = os.path.join(os.path.dirname(ale_py.__file__), "roms")
os.environ['ALE_PY_ROM_DIR'] = roms_path
class CnnDDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True

        self.double_dqn = config.double_dqn
        self.dueling_dqn = config.dueling_dqn
        self.prioritized = config.prioritized
        if self.prioritized:
            self.buffer = PrioritizedStorage(config)
        else:
            self.buffer = RolloutStorage(config)

        if self.dueling_dqn:
            self.model = DuelingDQN(self.config.state_shape, self.config.action_dim)
            if self.double_dqn:
                self.target_model = DuelingDQN(self.config.state_shape, self.config.action_dim)
                self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.model = CnnDQN(self.config.state_shape, self.config.action_dim)
            if self.double_dqn:
                self.target_model = CnnDQN(self.config.state_shape, self.config.action_dim)
                self.target_model.load_state_dict(self.model.state_dict())


        # self.model_optim = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate,
        #                                        eps=1e-5, weight_decay=0.95, momentum=0, centered=True)
        self.model_optim = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,betas=[0.9,0.999],eps=1e-5)
        self.loss = torch.nn.MSELoss(reduction='mean')
        if self.config.use_cuda:
            self.cuda()
            
    #代表这个agent的策略（即在观测下做出何种动作）
    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float32)/255.0
            if self.config.use_cuda:
                state = state.to(self.config.device)
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    #个agent如何学习策略
    '''
    需要在函数learning中利⽤batch=(s0,s1,a,r,done)来计算loss并更新⽹络参数。'''
    def learning(self, fr):
        s0, s1, a, r, done = self.buffer.sample(self.config.batch_size)
        if self.config.use_cuda:
            s0 = s0.to(torch.float32).to(self.config.device) / 255.0
            s1 = s1.to(torch.float32).to(self.config.device) / 255.0

            a = a.to(self.config.device).long()  # 动作应该是long类型
            r = r.to(self.config.device).float()  # 奖励转换为float类型
            done = done.to(self.config.device).float()  # done标志转换为float类型


        # How to calculate Q(s,a) for all actions
        # q_values is a vector with size (batch_size, action_shape, 1)
        # each dimension i represents Q(s0,a_i)
        q_values = self.model(s0).to(config.device).gather(1, a)

        with torch.no_grad():
            if self.double_dqn:
                q_values_target = self.target_model(s1)
            else:
                q_values_target = self.model(s1)
            max_target_q_values = q_values_target.max(1,keepdim=True).values
            q_target = r + self.config.gamma* max_target_q_values*(1-done)

        # How to calculate argmax_a Q(s,a)
        # actions = q_values.max(1)[1]

        # Tips: function torch.gather may be helpful
        # You need to design how to calculate the loss
        loss = self.loss(q_values,q_target)

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % self.config.update_tar_interval == 0 and self.double_dqn:
            self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    def cuda(self):
        self.model.to(self.config.device)
        if self.double_dqn:
            self.target_model.to(self.config.device)

    def load_weights(self, model_path):
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)

    def save_model(self, output, name=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        return fr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', default=True, help='train model')  # action='store_true',
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument('--cuda_id', type=str, default='0', help='if test or retrain, import the model')
    # parser.add_argument('--double_dqn',type=str,default=False,help='use double dqn')
    # parser.add_argument('--dueling_dqn',type=str,default=False,help='use dueling dqn')
    # parser.add_argument('--prioritized',type=str,default=False,help='use prioritized storage')
    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.cuda = True

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.05
    config.eps_decay = 30000
    config.frames = 500000
    config.use_cuda = args.cuda
    config.learning_rate = 1e-4
    config.init_buff = 10000
    config.max_buff = 100000
    config.learning_interval = 4
    config.update_tar_interval = 1000
    config.batch_size = 128
    config.gif_interval = 5000
    config.print_interval = 1000
    config.log_interval = 1000
    config.checkpoint = True
    config.checkpoint_interval = 500000
    config.win_reward = 18
    config.win_break = True
    config.device = torch.device("cuda:"+args.cuda_id if args.cuda else "cpu")


    config.double_dqn = True
    config.dueling_dqn = True
    config.prioritized = True
    # handle the atari env
    env = make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape
    print(config.action_dim, config.state_shape)
    agent = CnnDDQNAgent(config)


    if config.use_cuda:
        print("CUDA is available and being used. Device:", config.device)
    else:
        print("CUDA is not being used.")


    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, args.model_path)
        tester.test()

    elif args.retrain:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)

        fr = agent.load_checkpoint(args.model_path)
        trainer = Trainer(agent, env, config)
        trainer.train(fr)
