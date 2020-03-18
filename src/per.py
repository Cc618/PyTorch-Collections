# Prioritized Experience Replay

from random import random, randint
import torch as T


# !!! This method is less efficient than the method in the paper
# !!! which use a sum tree, get : O(n), add : O(1)
class PERMemory:
    def __init__(self, size, on_full, min_prob, a=.6, b=.4):
        '''
        - a : [0, 1], 0 = pure randomness, 1 = get the exp with most priority
        '''
        # List (exp, error) with error = (abs(td) + eps) ** a
        self.min_prob = min_prob
        self.on_full = on_full
        self.size = size
        self.last_index = 0
        self.data = [None for _ in range(self.size)]
        self.a = a
        self.b = b
        self.sm = 0

    def __get_i(self):
        '''
            Samples an index
        '''
        # P(i) = Pi ^ a / sum(Pk ^ a)
        rand = random() * self.sm
        i = 0
        for _, p in self.data:
            rand -= p
            if rand <= 0:
                return i

            i += 1

        return 0

    def clear(self):
        self.last_index = 0

    def add(self, exp, td):
        '''
            Stores an experience
        '''
        p = T.max(self.min_prob, T.abs(td).pow(self.a))
        self.sm += p
        self.data[self.last_index] = (exp, p)
        self.last_index += 1

        if self.last_index >= self.size:
            self.on_full()
            self.clear()

    def get(self):
        '''
            Samples an experience
        '''
        return self.data[self.__get_i()][0]

    # TODO : Test
    # (WIP) Not tested section
    # def corr(self):
    #     '''
    #         The correction error
    #     '''
    #     n = len(self.data)
    #     w = T.empty([n])
    #     for i in range(n):
    #         w[i] = self.data[i][1]

    #     w = (n * self.sm).pow_(-self.b)

    #     return w
    
    def iter(self, n):
        '''
        Samples 'size' * n experiences
        '''
        return (self.get() for _ in range(self.size))
