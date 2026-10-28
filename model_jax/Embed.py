import math
import jax.numpy as jnp
from flax import nnx


class Embedder(nnx.Module):
    def __init__(self, vocab_size, d_model, rngs):
        self.d_model = d_model
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)

    def __call__(self, x):
        return self.embed(x) * (x != 0)[..., None]


class PositionalEncoder(nnx.Module):
    def __init__(self, d_model, max_seq_len, dropout, rngs):
        self.d_model = d_model
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

        position = jnp.arange(0, max_seq_len)[..., None].astype(float)  # (max_seq_len, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2).astype(float) * -(math.log(10000.0) / d_model))  # (d_model//2,)

        pe = jnp.zeros((max_seq_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        self.pe = nnx.Variable(pe[None, ...])  # (1, max_seq_len, d_model)

    def __call__(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
