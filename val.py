import torch
import torch.nn as nn
from utils import MyMasker, MyTokenizer
from data import TextDataset
from torch.utils.data import random_split
from Models import Transformer

max_len = 32
model = Transformer(src_vocab=28, d_model=128, max_seq_len=max_len, N=12, heads=8, dropout=0.1)

model.load_state_dict(torch.load('./weights/model_weights'))
model.eval()

masker = MyMasker()
tokenizer = MyTokenizer(32)

# Loading data
bs=128
dataset = TextDataset()
train_size = int(0.8*len(dataset))
test_size = len(dataset)-train_size
train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

sample = [val_dataset[53]]
sample_len = len(sample[0])
print(sample)
sample = masker.mask(sample, 0.7)
print(sample)
sample = tokenizer.encode(sample)
sample = sample.to('cuda')
output = model(sample, None)

P = nn.Softmax(dim=-1)
prob = P(output)
out_id = torch.argmax(prob, dim=-1)
uncut = tokenizer.decode(out_id)
cut = uncut[0][:sample_len]

print(cut)