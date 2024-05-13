# import argparse
import time
import torch
import torch.nn as nn
from model.Models import Transformer
# from Process import *
import torch.nn.functional as F
from model.Batch import create_masks
# import dill as pickle
from utils.utils import MyTokenizer, MyMasker
from utils.data import TextDataset
from torch.utils.data import DataLoader, random_split

# Loading data
bs = 128
dataset = TextDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

trainloader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=0)
valloader = DataLoader(dataset=val_dataset, batch_size=bs, shuffle=True, num_workers=0)

# Loading Transformer model from scratch
max_len = 32
model = Transformer(src_vocab=28, d_model=128, max_seq_len=max_len, N=12, heads=8, dropout=0.1)
if torch.cuda.is_available():
    model.to('cuda')
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.98), eps=1e-9)


def train_model(model, bs=32, epochs=1000, printevery=100):
    masker = MyMasker()
    tokenizer = MyTokenizer(32)

    print("training model...")
    start = time.time()
    if torch.cuda.is_available():
        print('gpu detected!')
    else:
        print('no gpu detected')
        return 0

    model.train()
    for epoch in range(epochs):

        total_loss = 0

        for i, trg in enumerate(trainloader):

            perc = None
            src = masker.mask(trg, perc)  # e.g. [m_zh__n, _s, _w_so_e]
            src = tokenizer.encode(src)  # e.g. [[], [], []]

            # trg is the complete word
            trg = tokenizer.encode(trg)

            # our src_mask is the same as trg_mask = mask
            mask, _ = create_masks(src)  # e.g. [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]]

            # Converting to cuda
            if torch.cuda.is_available():
                src = src.to('cuda')
                mask = mask.to('cuda')
                trg = trg.to('cuda')

            preds = model(src, mask)
            # ys = trg[:, 1:].contiguous().view(-1)

            optim.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), trg.contiguous().view(-1), ignore_index=0)
            loss.backward()
            optim.step()

            '''
            if opt.SGDR == True: 
            opt.sched.step()
            '''
            total_loss += loss.item()

            # print(i+1)
            if (i + 1) % printevery == 0:
                p = int(100 * (i + 1) / len(trainloader.dataset) * bs)
                avg_loss = total_loss / printevery
                print("\r   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))),
                       p, avg_loss), end='')

            total_loss = 0

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), './weights/model_weights_1')

        print("\r   %dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
              ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
               avg_loss, epoch + 1, avg_loss))


train_model(model, bs=bs, printevery=1)
