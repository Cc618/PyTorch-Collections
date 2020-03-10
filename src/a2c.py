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
from utils import models_dir, eps, DenseLayer


class Net(nn.Module):
    '''
        Simple fully connected network which outputs the policy (probabilities for each action)
    '''
    def __init__(self, n_state, n_action, n_hidden_actor, n_hidden_critic):
        super().__init__()

        self.actor = DenseLayer(n_state, n_hidden_actor, n_action)
        self.critic = DenseLayer(n_state, n_hidden_critic, 1)

    def forward(self, x):
        # Returns the action distribution and the value
        return F.softmax(self.actor(x), 0), self.critic(x)


def compute_value_fun(rewards):
    '''
        Returns the state value functions : E(Sum(i, gamma ** i * reward))
    '''
    values = []
    value = 0
    for reward in reversed(rewards):
        value = discount_rate * value + reward
        values.append(value)
    
    return values[::-1]


def train_game(env, max_steps=-1):
    '''
        Train on one game
    - return : steps, total_reward
    '''
    # Used to learn after
    log_probs = []
    rewards = []
    values = []

    # Explore #
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        state = T.from_numpy(state).to(device).to(T.float32)
        
        # Compute the probabilities for each action and the value
        # (This is a distribution, the sum is 1)
        action_probs, value = net(state)

        # Sample an action and compute log(pi(action|state))
        dis = distributions.Categorical(action_probs)
        action = dis.sample()
        log_prob = dis.log_prob(action).view(1)
        action = action.detach().cpu().item()

        # Update game
        state, reward, done, _ = env.step(action)
        total_reward += reward

        # Memorize
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)

        if len(log_probs) == max_steps:
            break

    env.close()

    # Learn #
    # Cast outputs
    values = T.cat(values)
    log_probs = T.cat(log_probs)
    value_fun = T.tensor(compute_value_fun(rewards), dtype=T.float32, device=device)
    advantage = value_fun.detach() - values

    # Loss
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = .5 * advantage.pow(2).mean()
    loss = critic_loss + actor_loss

    # Back prop
    opti.zero_grad()
    loss.backward()
    opti.step()

    return len(log_probs), total_reward


def take_action(state):
    '''
        Returns the next action to take
    '''
    state = T.from_numpy(state).to(device).to(T.float32)

    # Compute the probabilities for each action
    # (This is a distribution, the sum is 1)
    action_probs, _ = net(state)
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
train = True
test = True
epochs = 150
n_display_games = 3
tests = 20
# Train at 1e-3 and then 5e-4 when there 150+ average steps
lr = 5e-4
discount_rate = .98
path = models_dir + '/a2c'
max_steps = 300
print_freq = 10

# Agent and env
env = gym.make(env_id)
env._max_episode_steps = max_steps
net = Net(env.observation_space.shape[0], env.action_space.n, 256, 1024).to(device)
opti = optim.Adam(net.parameters(), lr=lr, betas=(.9, .999))

if os.path.exists(path):
    net.load_state_dict(T.load(path))

# Train
if train:
    print('> Training')
    train_batch(env, epochs)

# Test
if test:
    print('> Testing')
    test_batch(env, games=tests)

# Display
# To create a video : env = gym.wrappers.Monitor(env, './video')
env._max_episode_steps = 1500
for _ in range(n_display_games):
    test_game(env, True)