# Functions and classes for reinforcement learning algorithms

from random import random


def show_game(env, agent):
    '''
        Renders a game from the env played by agent
    '''
    state = env.reset()
    done = False
    while not done:
        action = agent.take_action(state)
        state, _, done, _ = env.step(action)
        env.render()

    env.close()


def test_game(env, agent):
    '''
        Returns how many steps the agent weathered and
    the total reward
    '''
    steps = 0
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        steps += 1
        action = agent.take_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    env.close()

    return steps, total_reward


def memorize_game(env, agent):
    '''
        Create a batch of trajectories
    - trajectory : (action, prev_state, state, reward, done)
    - return : List of trajectories
    '''
    batch = []
    state = env.reset()
    done = False
    while not done:
        action = agent.take_action(state)
        new_state, reward, done, _ = env.step(action)

        batch.append((action, state, new_state, reward, done))

        state = new_state

    env.close()

    agent.exploration_rate *= agent.exploration_decay

    return batch


def sample_discrete_distribution(dis):
    '''
        Returns a random index in the distribution
    following this ditribution.
    - dis : One dimensional tensor, the sum is one
    '''
    # Sample index
    i = 0
    
    # Sample the distribution
    v = random()
    while True:
        v -= dis[i].detach().cpu().item()

        if v <= .01:
            break

        i += 1
    
    return i


class IAgent:
    '''
        Base class for an agent
    '''
    def __init__(self):
        pass

    def take_action(self, state):
        '''
            Returns the next action to take from this state
        '''
        raise NotImplementedError()


class RandomAgent(IAgent):
    '''
        Agent which takes random moves
    '''
    def __init__(self, env):
        super().__init__()

        self.action_space = env.action_space

    def take_action(self, state):
        return self.action_space.sample()


class IAi(IAgent):
    '''
        Agent which learn
    '''
    def __init__(self, env, exploration_decay):
        super().__init__()

        self.action_space = env.action_space
        self.exploration_decay = .96
        self.exploration_rate = 1

    def learn(self, batch):
        '''
            Learns from a batch of trajectories
        - batch : [(action, prev_state, state, reward, done)]
        * The batch contains only one game and is not shuffled
        '''
        raise NotImplementedError()

    def take_action(self, state):
        if random() > self.exploration_rate:
            return self.action_space.sample()
        return self.guess(state)

    def guess(self, state):
        '''
            Exploitation
        '''
        raise NotImplementedError()

