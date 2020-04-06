# Basic DQN trained on CartPole-v0
# Scores > 100 after 1000 epochs in average
# Uses a simple replay buffer

import random
from itertools import count
from random import randint, shuffle
import torch as T
from torch import nn, optim
import torch.nn.functional as F
import gym
from utils import models_dir, save_agent, try_load_agent


class ReplayBuffer:
    '''
        A basic memory, when 'size' steps are memorized,
    the functor on_learn is called with a batch as parameter
    '''
    def __init__(self, n_state, size, on_learn):
        super().__init__()

        self.size = size
        self.on_learn = on_learn
        self.sample_i = 0

        self.states = T.empty([size, n_state])
        self.next_states = T.empty([size, n_state])
        self.actions = T.empty([size], dtype=T.long)
        self.rewards = T.empty([size])
        self.dones = T.empty([size])

    def add(self, state, next_state, action, reward, done):
        self.states[self.sample_i] = state
        self.next_states[self.sample_i] = next_state
        self.actions[self.sample_i] = action
        self.rewards[self.sample_i] = reward
        self.dones[self.sample_i] = done

        self.sample_i += 1
        if self.sample_i >= self.size:
            # Shuffle data
            idx = [i for i in range(self.size)]
            shuffle(idx)

            self.on_learn(self.states[idx], self.next_states[idx], self.actions[idx], self.rewards[idx], self.dones[idx])

            # 'Clear' data
            self.sample_i = 0


class DQN(nn.Module):
    def __init__(self, n_state, n_action, n_hidden):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_action)

    def forward(self, state):
        y = F.relu(self.fc1(state))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return y


def learn(states, next_states, actions, rewards, dones):
    global dqn, n_action, opti, discount, exploration, exploration_decay, min_exploration, avg_loss

    # Update exploration
    exploration = max(exploration * exploration_decay, min_exploration)

    # Q values for these states
    q = (dqn(states) * F.one_hot(actions, n_action)).sum(1)

    # Next Q values
    best_q = T.max(dqn(next_states), 1)[0]

    # Target Q values
    q_target = rewards + (1 - dones) * discount * best_q

    loss = F.mse_loss(q, q_target.detach()).mean()
    # loss = F.smooth_l1_loss(q, q_target.detach()).mean()

    avg_loss += loss.item()

    opti.zero_grad()
    loss.backward()
    opti.step()


def act(state, exploration):
    if random.random() < exploration:
        return randint(0, n_action - 1)

    rewards = dqn(state)
    return T.argmax(rewards).detach().item()


def save():
    global path, dqn

    save_agent(dqn, path)
    print('Model saved')


lr = 14e-4
exploration_decay = .99
min_exploration = .1
n_hidden = 128
discount = .99
mem_size = 40
epochs = 2000
print_freq = 100
save_freq = 1000

path = models_dir + '/dqn'

env = gym.make('CartPole-v1')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

seed = 3141618
env.seed(seed)
T.manual_seed(seed)
random.seed(seed)

exploration = 1

mem = ReplayBuffer(n_state, mem_size, learn)
dqn = DQN(n_state, n_action, n_hidden)
opti = optim.Adam(dqn.parameters(), lr=lr)

# Load agent if possible
try_load_agent(dqn, path)

# Train
avg_reward = 0
avg_loss = 0
steps = 0
sync_step = 0
for e in range(1, epochs + 1):
    state = T.from_numpy(env.reset()).to(T.float32)
    done = False
    while not done:
        action = act(state, exploration)

        new_state, reward, done, _ = env.step(action)
        new_state = T.from_numpy(new_state).to(T.float32)

        mem.add(state, new_state, action, reward, float(done))

        state = new_state
        steps += 1
        avg_reward += reward

    if e % print_freq == 0:
        print(f'Epoch {e:5d} Reward {avg_reward / print_freq:<3.0f} Loss {avg_loss / steps:<6.4f} Exploration {exploration:<4.2f}')
        avg_reward = avg_loss = steps = 0

    if e % save_freq == 0:
        save()
    
save()

# Play
for i in count():
    total_reward = 0
    state = T.from_numpy(env.reset()).to(T.float32)
    done = False
    while not done:
        action = act(state, 0)

        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        new_state = T.from_numpy(new_state).to(T.float32)

        env.render()

        state = new_state
    
    env.close()

    print(f'Test {i:3d} Reward : {total_reward}')
