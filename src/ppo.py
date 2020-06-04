# PPO using clipped objective
# Around 200 epochs to have > 150 reward

import os
import gym
import torch as T
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, optim
from utils import save_agent, try_load_agent, models_dir


class Buffer:
    '''
        Stores experiences to learn from minibatches
    '''
    def __init__(self, batch_size, learn):
        self.batch_size = batch_size
        self.learn = learn

        self.data = []
        
    def add(self, *args):
        self.data.append(args)
    
    def finish_game(self):
        # Learn from minibatches
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i : min(len(self.data), i + self.batch_size)]
            self.learn(batch)

        self.data.clear()


def save():
    # TODO : Verify
    T.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
    }, path)

    print('Model saved')


def try_load():
    global actor, critic

    if os.path.exists(path):
        data = T.load(path)
        actor.load_state_dict(data['actor'])
        critic.load_state_dict(data['critic'])

        print('Model loaded')


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
    rewards = [b[1] for b in batch]
    log_probs = T.cat([b[2] for b in batch])
    actions = T.Tensor([b[3] for b in batch])

    rewards = discounted_rewards(rewards)
    values = critic(states).squeeze(1)
    advantages = rewards - values

    old_log_probs = log_probs.detach()
    for _ in range(ppo_epochs):
        new_dis = D.Categorical(probs=actor(states))
        new_log_probs = new_dis.log_prob(actions)

        entropy = new_dis.entropy()
        ratio = (new_log_probs - old_log_probs).exp()

        surr1 = ratio * advantages.detach()
        surr2 = T.clamp(ratio, 1 - ppo_ratio, 1 + ppo_ratio) * advantages.detach()
        # L^CLIP
        actor_obj = -T.min(surr1, surr2).mean()
        
        loss_actor = actor_obj - entropy_ratio * entropy.mean()
        opti_actor.zero_grad()
        loss_actor.backward()
        opti_actor.step()

        value = critic(states)
        loss_critic = critic_ratio * F.mse_loss(value, rewards.view(-1, 1)).mean()
        opti_critic.zero_grad()
        loss_critic.backward()
        opti_critic.step()


lr = 5e-4
n_hidden = 128
discount_factor = .99
batch_size = 100
ppo_epochs = 4
ppo_ratio = .2
entropy_ratio = 1e-2
critic_ratio = 1

print_freq = 20
save_freq = 5000
seed = 161831415
path = models_dir + '/ppo'

env = gym.make('CartPole-v1')
env.seed(seed)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

buf = Buffer(batch_size, learn)

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

try_load()

e = 0
avg_reward = 0

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
        log_prob = dis.log_prob(action).view(1)
        action = action.detach().item()

        # Update game
        new_state, reward, done, _ = env.step(action)

        # Display
        # env.render()

        # Memorize
        buf.add(state, reward, log_prob, action)

        total_reward += reward
        state = new_state

    buf.finish_game()

    env.close()

    avg_reward += total_reward

    if e % print_freq == 0:
        print(f'Epoch {e:4d} Avg reward {avg_reward / print_freq:3.1f}')
        avg_reward = 0

    if e % save_freq == 0:
        save()




