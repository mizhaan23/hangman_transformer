"""Pure JAX utilities for tokenization, Hangman masking, and serialization.

Mirrors utils.py with GPU-accelerated array primitives using jax.vmap and JIT compilation.
"""

from pathlib import Path
from typing import List, Sequence, Union, Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


# ==============================================================================
# 1. GPU-Native Masking Functions (Functional Kernels)
# ==============================================================================

def mask_single_word(
    word_ids: jax.Array,
    key: jax.Array,
    perc: Optional[Union[float, jax.Array]] = None,
) -> jax.Array:
    """Mask characters of a single word according to Hangman rules in pure JAX.

    All occurrences of selected unique letters are masked simultaneously to 27 ('_'),
    while padding tokens (0) and unmasked letters (1..26) are preserved.

    Args:
        word_ids: (M,) int32 array of character token IDs.
        key: jax.random.PRNGKey for stochastic sampling.
        perc: Optional float in [0, 1]. If None, uniformly sampled per word.

    Returns:
        (M,) int32 array representing the partially masked word.
    """
    letters = jnp.arange(1, 27)
    # Identify unique letters present in the word
    present = jnp.any(word_ids[:, None] == letters[None, :], axis=0)  # (26,) bool
    num_unique = jnp.sum(present)

    key_perc, key_rand = jax.random.split(key)
    if perc is None:
        perc = jax.random.uniform(key_perc)

    # Number of distinct letters to mask: k = min(U, max(1, floor(perc * U)))
    k = jnp.maximum(1, jnp.floor(perc * num_unique).astype(jnp.int32))
    k = jnp.minimum(k, jnp.maximum(1, num_unique))

    # Score present letters with uniform noise and rank
    rand_scores = jax.random.uniform(key_rand, shape=(26,))
    scores = jnp.where(present, rand_scores, -1.0)
    ranks = jnp.argsort(jnp.argsort(-scores))  # rank 0 is highest random score
    is_masked_letter = (ranks < k) & present   # (26,) bool

    is_letter = (word_ids >= 1) & (word_ids <= 26)
    letter_idx = jnp.clip(word_ids - 1, 0, 25)
    masked = is_masked_letter[letter_idx]

    # Token 27 is '_', padding (0) remains 0
    return jnp.where(is_letter & masked, 27, word_ids)


# Batched masking via jax.vmap
vmap_mask = jax.vmap(mask_single_word, in_axes=(0, 0, None))


# ==============================================================================
# 2. Tokenizer & Masker Classes (Object-Oriented Interface Matching utils.py)
# ==============================================================================

class JaxTokenizer:
    """Character-level tokenizer producing and decoding JAX integer tensors."""

    def __init__(self, max_length: int = 32):
        self.max_length = max_length
        self.map = {'_': 27}
        self.reverse_map = {27: '_', 0: ''}
        for i in range(26):
            ch = chr(ord('a') + i)
            self.map[ch] = i + 1
            self.reverse_map[i + 1] = ch

    def encode(self, words: Sequence[str]) -> jax.Array:
        """Encode a batch of word strings into a contiguous (B, max_length) JAX int32 array."""
        arr = np.zeros((len(words), self.max_length), dtype=np.int32)
        for i, word in enumerate(words):
            cleaned = word.strip().lower()
            length = min(len(cleaned), self.max_length)
            arr[i, :length] = [self.map.get(ch, 0) for ch in cleaned[:length]]
        return jnp.array(arr)

    def decode(self, token_ids: Union[jax.Array, np.ndarray]) -> List[str]:
        """Decode a (B, max_length) array of integer tokens back into a list of strings."""
        tokens_np = np.asarray(token_ids)
        decoded = []
        for row in tokens_np:
            word = ''.join(self.reverse_map.get(int(t), '') for t in row if int(t) != 0)
            decoded.append(word)
        return decoded


class JaxMasker:
    """GPU-accelerated Hangman character masker."""

    def __init__(self):
        pass

    def mask(
        self,
        word_ids: jax.Array,
        key: jax.Array,
        percentage: Optional[float] = None,
    ) -> jax.Array:
        """Mask a batch of encoded words (B, max_length) on GPU using vmap."""
        batch_size = word_ids.shape[0]
        keys = jax.random.split(key, batch_size)
        return vmap_mask(word_ids, keys, percentage)


# ==============================================================================
# 3. Model Weight Serialization Helper
# ==============================================================================

def save_npz_weights(model: nnx.Module, path: Union[str, Path]):
    """Save model parameters in flat dictionary format compatible with load_weights."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}

    def walk(tree, prefix):
        for k, v in tree.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                walk(v, name)
            else:
                arr = np.asarray(v)
                if np.issubdtype(arr.dtype, np.floating) or arr.dtype.name == "bfloat16":
                    arr = arr.astype(np.float32)
                flat[name] = arr

    walk(nnx.state(model, nnx.Not(nnx.RngState)).to_pure_dict(), "")
    np.savez(str(target_path), **flat)
