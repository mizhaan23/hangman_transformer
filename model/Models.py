import torch
import torch.nn as nn
from model.Layers import EncoderLayer
from model.Embed import Embedder, PositionalEncoder
from model.Sublayers import Norm
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


'''
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
'''


class Transformer(nn.Module):
    def __init__(self, src_vocab, d_model, max_seq_len, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, max_seq_len, N, heads, dropout)
        # self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, src_vocab)

    def forward(self, src, src_mask=None):
        if src_mask is None:
            src_mask = (src != 0).unsqueeze(-2)
        e_outputs = self.encoder(src, src_mask)
        # d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(e_outputs)
        return output


class Transformer2(nn.Module):
    def __init__(self, src_vocab, d_model, max_seq_len, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, max_seq_len, N, heads, dropout)
        # self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, src_vocab)

    def forward(self, src, src_mask=None):
        if src_mask is None:
            src_mask = (src != 0).unsqueeze(-2)
        e_outputs = self.encoder(src, src_mask)
        # d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(e_outputs)
        y = torch.where(src_mask, 1., 0.)
        output = torch.matmul(y, output).squeeze(1)
        return output


class PGN(nn.Module):
    def __init__(self, src_vocab=28, d_model=128, max_seq_len=32, N=12, heads=8, dropout=0.1):
        super().__init__()
        self.transformer = Transformer(src_vocab, d_model, max_seq_len, N, heads, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        if mask is None:
            mask = (x != 0).unsqueeze(-2)
        self.x = self.transformer(x, mask)
        self.x = self.softmax(self.x)

        self.pi = torch.matmul(1. * mask, self.x)  # effectively adds the probs row-wise for each action / character

        # self.x = torch.mul(self.x.squeeze(1), left)
        self.pi = self.pi.squeeze(1)
        self.pi = self.pi / torch.sum(self.pi)

        return self.pi


class PGN2(nn.Module):
    def __init__(self, src_vocab=28, d_model=128, max_seq_len=32, N=12, heads=8, dropout=0.1):
        super().__init__()
        self.transformer = Transformer2(src_vocab, d_model, max_seq_len, N, heads, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        out = self.transformer(x, mask)
        pi = self.softmax(out)
        return pi


def get_model(opt, src_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if opt.device == 0:
        model = model.cuda()

    return model
