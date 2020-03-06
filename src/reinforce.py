# REINFORCE algorithm on CartPole env

from random import random, shuffle
import gym
import numpy as np
import torch as T
import torch.nn.functional as F
from torch import nn
from torch import optim
from rlutils import IAi, show_game, test_game, memorize_game, sample_discrete_distribution, RandomAgent


def test(env, ai, tests):
    ai.exploration_rate = 0
    avg_steps, avg_reward = 0, 0
    for _ in range(tests):
        # TODO : Max steps
        steps, reward = test_game(env, ai)
        avg_steps += steps
        avg_reward += reward

    avg_steps /= tests
    avg_reward /= tests

    print(f'Average steps in tests : {avg_steps:.1f}')
    print(f'Average total reward in tests : {avg_reward:.2f}')


class ReinforceAgentPolicy(nn.Module):
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
        # TODO : Sigmoid
        x = F.softmax(F.sigmoid(self.fc3(x)), dim=0)

        return x


class ReinforceAgent(IAi):
    def __init__(self, env, device, lr=.0002, n_hidden=24, discount_factor=.96, exploration_decay=.96):
        super().__init__(env, exploration_decay=.96)

        self.discount_factor = discount_factor
        self.device = device

        self.policy = ReinforceAgentPolicy(env.observation_space.shape[0], n_hidden, env.action_space.n)
        self.policy = self.policy.to(device)
        self.opti = optim.Adam(self.policy.parameters(), lr=lr, betas=(.9, .999))

    def learn(self, batch):
        # !!! Reversed
        parsed_batch = []

        # This is the state value function
        # <=> E[Sum(k from 0 to +inf, discount_factor ** k * reward_k) given the state s]
        acc_reward = 0

        # Iterate backwards the batch to have the accumulated reward
        for action, _, state, reward, _ in reversed(batch):
            state = T.tensor(state, dtype=T.float32, device=self.device)

            # Update the accumulated reward
            acc_reward *= self.discount_factor
            acc_reward += reward

            # Accumulate rewards
            parsed_batch.append([action, state, acc_reward])

        # Components to make the loss and learn
        # [[policy(state), advantage(action, state)]]
        components = []

        # Compute policies and advantages
        for action, state, value_fun in reversed(parsed_batch):
            # Guess action probabilities and the next action
            action_probs = self.policy(state)

            # The advantage is how much this action is better
            # than the value function
            advantage = action_probs[action] - value_fun

            components.append([action_probs[action], advantage])

        # Learn
        shuffle(components)
        for policy, advantage in components:
            self.opti.zero_grad()

            # - to make gradient ascent
            loss = -T.log(policy) * advantage

            # Update weights
            loss.backward()
            self.opti.step()

    def guess(self, state):
        '''
            Exploitation
        '''
        with T.no_grad():
            state = T.tensor(state, dtype=T.float32, device=self.device)

            # Guess probabilities for each action
            # This is like a distribution, the sum
            # of each prob is 1
            action_probs = self.policy(state)

            # Sample an action index from the distribution
            action_i = sample_discrete_distribution(action_probs)

        # Return the index of the action
        return action_i


# Params
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
env_id = 'CartPole-v0'
epochs = 40
tests = 20

# Agent and env
env = gym.make(env_id)
ai = ReinforceAgent(env, device, lr=1e-3, n_hidden=64)

# Train
for e in range(epochs):
    print(f'Epoch {e + 1}')
    game = memorize_game(env, ai)
    ai.learn(game)

# Test
print('--- Random ---')
test(env, RandomAgent(env), tests)
print('--- Ai ---')
test(env, ai, tests)

