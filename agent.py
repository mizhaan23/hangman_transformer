import torch
import torch.nn as nn
from Models import PGN
from utils import MyTokenizer, MyMasker
from Batch import create_masks
from torch.autograd import Variable
import numpy as np

class Agent:        # src_vocab=28, d_model=128, max_seq_len=max_len, N=12, heads=8, dropout=0.1
    def __init__(self, model, lr=0.001, gamma=1, max_seq_len=32):
        self.lr = lr
        self.gamma = gamma

        self.log_probs = torch.empty(0)
        if torch.cuda.is_available():
            self.log_probs = self.log_probs.to('cuda')

        self.reward_memory = []
        self.policy = model  # Our PGN class goes here
        self.tokenizer = MyTokenizer(max_seq_len)
        self.mat = self._get_special_matrix(gamma, 26)

    def make_guess(self, observation, left, mask):
        # observation e.g. : ['__a_ge']
        state = self.tokenizer.encode(observation)

        if torch.cuda.is_available():
            state = state.to('cuda')
            mask = mask.to('cuda')
            left = left.to('cuda')

        probs = self.policy(state, mask)

        # returns an int : 4
        c = torch.distributions.Categorical(probs=probs)

        # behavioural policy
        '''
        b_probs = torch.mul(probs, left)
        b_probs = b_probs / torch.sum(b_probs)
        b = torch.distributions.Categorical(probs=b_probs)
        '''

        action = c.sample()

        if self.log_probs.dim() != 0:
            self.log_probs = torch.cat([self.log_probs, c.log_prob(action)])
        else:
            self.log_probs = c.log_prob(action)

        # returns a chr : 'd'
        guess = self.tokenizer.reverse_map[action.item()]
        return guess, action.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward) # list of ints

    # will learn after Monte-Carlo Episode
    def learn(self, optimizer):
        episode_len = len(self.reward_memory)
        mat = self.mat[:episode_len, :episode_len]

        rewards = torch.tensor(self.reward_memory, dtype=torch.float)
        if torch.cuda.is_available():
            mat = mat.to('cuda')
            rewards = rewards.to('cuda')
        G = torch.matmul(rewards, mat)
        G = (G - G.mean()) / (G.std() + 1e-9)

        # Calculating loss
        loss = torch.dot(self.log_probs, -G)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def clear_memory(self):
        # After update clear memory
        del self.log_probs
        del self.reward_memory

        self.log_probs = torch.empty(0)
        if torch.cuda.is_available():
            self.log_probs = self.log_probs.to('cuda')
        # self.state_memory = []
        # self.action_memory = []
        self.reward_memory = []



    def _get_special_matrix(self, gamma, n):
        mat = torch.zeros((n, n), dtype=torch.float)
        for i in range(n):
            for j in range(i, n):
                m_ij = gamma ** (j - i)
                mat[j, i] = m_ij
        return mat



    def make_random_guess(self):
        r = np.random.randint(1, 26)
        guess = self.tokenizer.reverse_map[r]
        return guess, r




