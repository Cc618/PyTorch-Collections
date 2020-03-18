# Double Deep Q Network

import os
from random import random, randint, shuffle
import gym
import torch as T
from torch import nn
from torch import optim
import torch.nn.functional as F
from utils import models_dir


class Net(nn.Module):
    def __init__(self, n_state, n_action, n_hidden, lr=1e-3, discount_rate=.96, exploration_decay=.98):
        super().__init__()

        self.n_action = n_action
        self.discount_rate = discount_rate
        self.exploration_decay = exploration_decay
        self.exploration_rate = 1

        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_action)
        
        self.opti = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, state):
        '''
            Predicts the state action value for each action in this state
        '''
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        state = self.fc3(state)

        return state

    def guess_action(self, state):
        '''
            Guesses the next action to take
        '''
        values = self(state)

        return T.argmax(values).detach().cpu().item()

    def take_action(self, state):
        '''
            Either takes a random action or guesses the next action
        '''
        # Explore
        if random() < self.exploration_rate:
            return randint(1, self.n_action) - 1

        # Guess
        return self.guess_action(state)

    def learn(self, step):
        '''
            Learns from an env step
        - step : (action, prev_state, state, reward, done)
        '''
        action, prev_state, state, reward, done = step

        # Values for this state
        values = self(prev_state)

        # self(prev_state)[action] should be == reward + gamma * max(self(state)) * (1 - done)
        action_value = T.tensor(reward, dtype=T.float32, device=device)

        if not done:
            action_value += self.discount_rate * T.max(self(state))

        loss = self.criterion(action_value, values[action])

        self.opti.zero_grad()
        loss.backward()
        self.opti.step()


def train_game(env, max_steps=-1):
    '''
        Train on one game
    - return : steps, total_reward
    '''
    # Memorize #
    # Memory
    env_steps = []
    total_reward = 0
    state = T.from_numpy(env.reset()).to(device).to(T.float32)
    done = False
    while not done:
        prev_state = state
        action = net.take_action(state)

        # Update game
        state, reward, done, _ = env.step(action)
        state = T.from_numpy(state).to(device).to(T.float32)
        total_reward += reward

        # Memorize
        env_steps.append((action, prev_state, state, reward, done))

        if len(env_steps) == max_steps:
            break

    env.close()

    # Learn #
    shuffle(env_steps)

    for step in env_steps:
        net.learn(step)

    # Update exploration rate
    net.exploration_rate *= net.exploration_decay

    return len(env_steps), total_reward


def train_batch(env, epochs):
    avg_steps = 0
    avg_reward = 0
    for e in range(1, epochs + 1):
        if e % print_freq == 0:
            print(f'Epoch {e:4d}\tAverage steps : {avg_steps / print_freq:.1f}\tAverage total reward : {avg_reward / print_freq:.1f}')
            avg_steps = 0
            avg_reward = 0

        steps, reward = train_game(env, max_steps=max_steps)
        avg_steps += steps
        avg_reward += reward

    # Save
    if save:
        T.save(net.state_dict(), path)
        print('Model trained and saved')


def test_game(env, render=False, max_steps=-1):
    '''
        Test on one game
    - return : (steps, total_reward)
    '''
    steps = 0
    total_reward = 0
    state = env.reset()
    done = False
    with T.no_grad():
        while not done:
            state = T.tensor(state, dtype=T.float32, device=device)

            # Guess action
            action = net.guess_action(state)

            # Update game
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if render:
                env.render()

            steps += 1

            if steps == max_steps:
                break

    env.close()

    return steps, total_reward


def test_batch(env, games=20):
    '''
        Tests the agent on multiple games,
    displays the results
    '''
    avg_steps, avg_reward = 0, 0
    for _ in range(games):
        steps, reward = test_game(env, max_steps=max_steps)
        avg_steps += steps
        avg_reward += reward

    avg_steps /= games
    avg_reward /= games

    print('Test ended :')
    print(f'- Average steps : {avg_steps:.1f}')
    print(f'- Average total reward : {avg_steps:.1f}')


# Params
train = True
test = True
n_display = 3
save = False
epochs = 200
print_freq = 10
max_steps = 200
device = T.device('cpu') # CPU is better for minibatches T.device('cuda:0' if T.cuda.is_available() else 'cpu')
path = models_dir + '/ddqn'
T.manual_seed(314159265)

# Env
env_name = 'CartPole-v1'
env = gym.make(env_name)
env.seed(314159265)

# Net
net = Net(env.observation_space.shape[0], env.action_space.n, 128, lr=1e-3, discount_rate=.98, exploration_decay=.98)
net.to(device)

# !!! Be careful, the exploration rate is reset when the model is loaded
if save and os.path.exists(path):
    net.load_state_dict(T.load(path))
    print('Model loaded')

# Training
if train:
    train_batch(env, epochs)

# Test
if test:
    test_batch(env)

# Display
for _ in range(n_display):
    print(test_game(env, True, max_steps=1000)[1])

