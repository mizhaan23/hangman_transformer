import torch
import random
import time
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self):
        with open('./utils/words_250000_train.txt', 'r') as f:
            words = f.read().splitlines()
        f.close()
        self.n_samples = len(words)
        self.words = words

    def __getitem__(self, index):
        return self.words[index]

    def __len__(self):
        return self.n_samples


'''
dataset = TextDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

num_epochs = 10
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, src in enumerate(dataloader):
        if i == 0:
            print(src)
        break
'''
