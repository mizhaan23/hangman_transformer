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
