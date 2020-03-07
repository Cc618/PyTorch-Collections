# REINFORCE algorithm on CartPole env

import os
from random import random, shuffle
import gym
import numpy as np
import torch as T
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch import distributions
from utils import models_dir, eps


class Policy(nn.Module):
    '''
        Simple fully connected network which outputs the policy (probabilities for each action)
    '''
    def __init__(self, n_x, n_h, n_y):
        super().__init__()

        self.fc1 = nn.Linear(n_x, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, n_y)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)

        return x


def train_game(env, max_steps=-1):
    '''
        Train on one game
    - return : steps, total_reward
    '''
    # Used to learn after
    # List of [log_prob, reward]
    env_steps = []

    # Explore #
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        state = T.from_numpy(state).to(device).to(T.float32)
        
        # Compute the probabilities for each action
        # (This is a distribution, the sum is 1)
        action_probs = policy(state)
        dis = distributions.Categorical(action_probs)
        # Sample an action
        action = dis.sample()
        # Compute log(pi(action|state))
        log_prob = dis.log_prob(action)
        action = action.detach().cpu().item()

        # Update game
        state, reward, done, _ = env.step(action)
        total_reward += reward

        # Memorize
        env_steps.append([log_prob, reward])

        if len(env_steps) == max_steps:
            break

    env.close()

    # Learn #
    # Compute the state value function = Sum for i : gamma ** i * reward[i]
    state_val = 0
    # Now advantages are just state value functions
    advantages = []
    for _, reward in reversed(env_steps):
        state_val = reward + discount_rate * state_val
        advantages.append(state_val)

    # advantages was reversed
    advantages = advantages[::-1]
    advantages = T.tensor(advantages, device=device)
    # Compute advantages and normalize
    advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

    # Back prop
    opti.zero_grad()

    loss = T.tensor(0, dtype=T.float32, device=device)
    for (log_prob, _), advantage in zip(env_steps, advantages):
        loss += log_prob * advantage

    loss = -loss

    loss.backward()
    opti.step()

    return len(env_steps), total_reward


def take_action(state):
    '''
        Returns the next action to take
    '''
    state = T.from_numpy(state).to(device).to(T.float32)

    # Compute the probabilities for each action
    # (This is a distribution, the sum is 1)
    action_probs = policy(state)
    dis = distributions.Categorical(action_probs)

    # Sample an action
    return dis.sample().detach().cpu().item()


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
    T.save(policy.state_dict(), path)
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
    while not done:
        # Guess action
        action = take_action(state)

        # Update game
        state, _, done, _ = env.step(action)

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
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
env_id = 'CartPole-v0'
train = False
test = False
epochs = 40
n_display_games = 4
tests = 20
lr = 1e-3
discount_rate = .96
path = models_dir + '/reinforce'
max_steps = 200
print_freq = 10

# Agent and env
env = gym.make(env_id)
policy = Policy(env.observation_space.shape[0], 64, env.action_space.n).to(device)
opti = optim.Adam(policy.parameters(), lr=lr, betas=(.9, .999))

if os.path.exists(path):
    policy.load_state_dict(T.load(path))

# Train
if train:
    print('> Training')
    train_batch(env, epochs)

# Test
if test:
    print('> Testing')
    test_batch(env, games=tests)

# Display
for _ in range(n_display_games):
    test_game(env, True)
