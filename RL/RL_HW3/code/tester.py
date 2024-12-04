class Tester(object):

    def __init__(self, agent, env, model_path, num_episodes=50, max_ep_steps=400, test_ep_steps=100):
        self.num_episodes = num_episodes
        self.max_ep_steps = max_ep_steps
        self.test_ep_steps = test_ep_steps
        self.agent = agent
        self.env = env
        self.agent.is_training = False
        self.agent.load_weights(model_path)
        self.policy = lambda x: agent.act(x)

    def test(self, debug=False, visualize=True):
        avg_reward = 0
        for episode in range(self.num_episodes):
            s0 = self.env.reset()
            episode_steps = 0
            episode_reward = 0.

            done = False
            while not done:
                if visualize:
                    self.env.render()

                action = self.policy(s0)
                s0, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_steps += 1

                if episode_steps + 1 > self.test_ep_steps:
                    done = True

            if debug:
                print('[Test] episode: %3d, episode_reward: %5f' % (episode, episode_reward))

            avg_reward += episode_reward
        avg_reward /= self.num_episodes
        print("avg reward: %5f" % (avg_reward))




