import math
import jax.numpy as jnp
from flax import nnx


class Norm(nnx.Module):
    def __init__(self, d_model, eps=1e-6, *, rngs):
        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nnx.Param(jnp.ones(self.size))
        self.bias = nnx.Param(jnp.zeros(self.size))

        self.eps = eps

    def __call__(self, x):
        std = x.std(axis=-1, keepdims=True, ddof=1)
        x = (x - x.mean(axis=-1, keepdims=True)) / (std + self.eps)
        return self.alpha * x + self.bias


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = q @ k.swapaxes(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        mask = mask[:, None, ...]
        scores = jnp.where(mask == 0, -1e9, scores)

    scores = nnx.softmax(scores, axis=-1)

    if dropout is not None:
        scores = dropout(scores)

    return scores @ v


class MultiHeadAttention(nnx.Module):
    def __init__(self, heads, d_model, dropout=0.1, *, rngs):

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_linear = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_linear = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, q, k, v, mask=None):
        bs = q.shape[0]

        # perform linear operation and split into N heads
        k = self.k_linear(k).reshape(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).reshape(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).reshape(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.swapaxes(1, 2)
        q = q.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.swapaxes(1, 2).reshape(bs, -1, self.d_model)
        return self.out(concat)


class FeedForward(nnx.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1, *, rngs):
        # We set d_ff as a default to 2048
        self.linear_1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.linear_2 = nnx.Linear(d_ff, d_model, rngs=rngs)

    def __call__(self, x):
        x = self.dropout(nnx.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
