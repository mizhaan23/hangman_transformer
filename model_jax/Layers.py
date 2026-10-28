from flax import nnx
from .Sublayers import FeedForward, MultiHeadAttention, Norm


class EncoderLayer(nnx.Module):
    def __init__(self, d_model, heads, dropout=0.1, *, rngs):
        self.norm_1 = Norm(d_model, rngs=rngs)
        self.norm_2 = Norm(d_model, rngs=rngs)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout, rngs=rngs)
        self.ff = FeedForward(d_model, dropout=dropout, rngs=rngs)
        self.dropout_1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout_2 = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
