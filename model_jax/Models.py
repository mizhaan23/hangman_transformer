from pathlib import Path
from typing import Union
import jax.numpy as jnp
from flax import nnx

from .Layers import EncoderLayer
from .Embed import Embedder, PositionalEncoder
from .Sublayers import Norm


def load_npz_weights(model: nnx.Module, npz_path: Union[str, Path]):
    """Load weights from a pre-converted Flax NNX .npz file."""
    data = jnp.load(npz_path)
    for key in data.files:
        val = data[key]
        parts = key.split(".")
        curr = model
        for p in parts[:-1]:
            curr = curr[int(p)] if p.isdigit() else getattr(curr, p)
        target = getattr(curr, parts[-1])
        if hasattr(target, "__setitem__"):
            target[...] = val
        else:
            setattr(curr, parts[-1], val)


class Encoder(nnx.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, N, heads, dropout, *, rngs):
        self.N = N
        self.embed = Embedder(vocab_size, d_model, rngs=rngs)
        self.pe = PositionalEncoder(d_model, max_seq_len, dropout=dropout, rngs=rngs)
        self.layers = nnx.List([EncoderLayer(d_model, heads, dropout, rngs=rngs) for _ in range(N)])
        self.norm = Norm(d_model, rngs=rngs)

    def __call__(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Transformer(nnx.Module):
    def __init__(self, src_vocab, d_model, max_seq_len, N, heads, dropout, *, rngs):
        self.encoder = Encoder(src_vocab, d_model, max_seq_len, N, heads, dropout, rngs=rngs)
        self.out = nnx.Linear(d_model, src_vocab, rngs=rngs)

    def __call__(self, src, src_mask=None):
        if src_mask is None:
            src_mask = (src != 0)[..., None, :]
        e_outputs = self.encoder(src, src_mask)
        return self.out(e_outputs)

    def load_weights(self, npz_path: Union[str, Path]):
        """Load pre-trained weights from a .npz file."""
        load_npz_weights(self, npz_path)
