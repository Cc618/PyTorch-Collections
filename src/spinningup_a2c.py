# A2C inspired by https://spinningup.openai.com/en/latest/algorithms/vpg.html

import gym
import torch as T
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, optim
from utils import save_agent, try_load_agent, models_dir


class Buffer:
    def __init__(self, size, batch_size, learn):
        self.size = size
        self.batch_size = batch_size
        self.learn = learn

        self.data = []
        
    def add(self, state, action, reward, log_prob):
        self.data.append((state, action, reward, log_prob))

        # if len(self.data) >= self.size:
        #     self.learn(self.data)
        #     self.data.clear()
    
    def finish_game(self):
        for i in range(0, self.size, self.batch_size):
            batch = self.data[i : min(len(self.data), i + self.batch_size)]
            self.learn(batch)

        self.data.clear()


def discounted_rewards(rewards):
    dis_rwd = T.empty([len(rewards)])
    current_dis_rwd = 0

    i = len(rewards)
    for rwd in reversed(rewards):
        i -= 1
        current_dis_rwd = current_dis_rwd * discount_factor + rwd
        dis_rwd[i] = current_dis_rwd

    return dis_rwd


def learn(batch):
    states = T.cat([T.Tensor(b[0]).unsqueeze(0) for b in batch])
    # actions = T.Tensor([b[1] for b in batch])
    rewards = [b[2] for b in batch]
    log_probs = T.cat([b[3] for b in batch])

    rewards = discounted_rewards(rewards)
    advantages = rewards - critic(states)


    # loss_actor = -(log_probs * advantages.detach()).mean()
    loss_actor = (-1 / len(buf.data)) * (log_probs * advantages.detach()).sum()
    opti_actor.zero_grad()
    loss_actor.backward()
    opti_actor.step()

    # loss_critic = .5 * advantages.pow(2).mean()
    loss_critic = (1 / (len(batch) * len(buf.data))) * advantages.pow(2).sum()
    opti_critic.zero_grad()
    loss_critic.backward()
    opti_critic.step()

    # print(loss_actor.item() + loss_critic.item())


train = False
n_hidden = 128
batch_size = 20
lr = 1e-3
discount_factor = .98
print_freq = 20
save_freq = 100
seed = 161831415
path = models_dir + '/vpg'
path_actor = path + "_actor"
path_critic = path + "_critic"

env = gym.make('LunarLander-v2')
env.seed(seed)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

T.manual_seed(seed)
actor = nn.Sequential(
    nn.Linear(n_state, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_action),
    nn.Softmax(0)
)

critic = nn.Sequential(
    nn.Linear(n_state, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, 1),
)

opti_actor = optim.Adam(actor.parameters(), lr=lr)
opti_critic = optim.Adam(critic.parameters(), lr=lr)

try_load_agent(actor, path_actor)
try_load_agent(critic, path_critic)

buf = Buffer(200, batch_size, learn)

e = 0
avg_reward = 0

if train:
    while True:
        e += 1

        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            state = T.Tensor(state)

            # Choose action
            action_probs = actor(state)
            dis = D.Categorical(action_probs)
            action = dis.sample()
            # entropy += dis.entropy()
            log_prob = dis.log_prob(action).view(1)
            action = action.detach().item()

            # Update game
            new_state, reward, done, _ = env.step(action)

            # Memorize
            buf.add(state, action, reward, log_prob)

            total_reward += reward
            state = new_state

        # TODO
        buf.learn(buf.data)
        buf.data.clear()

        env.close()

        avg_reward += total_reward

        if e % print_freq == 0 and e != 0:
            print(f'Epoch {e:4d} Avg reward {avg_reward / print_freq:3.1f}')
            avg_reward = 0

        if e % save_freq == 0 and e != 0:
            save_agent(actor, path_actor)
            save_agent(critic, path_critic)
            print('Model saved')
else:
    # To create a video :
    env = gym.wrappers.Monitor(env, './video')
    
    # Render
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        state = T.Tensor(state)

        # Choose action
        action_probs = actor(state)
        dis = D.Categorical(action_probs)
        action = dis.sample()
        action = action.detach().item()

        # Update game
        new_state, reward, done, _ = env.step(action)

        env.render()

        total_reward += reward
        state = new_state

    env.close()

    print(total_reward)
