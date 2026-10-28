"""Pure JAX implementation of the Hangman Environment.

Designed for native GPU acceleration with jax.vmap and jax.lax.scan.
Supports standard vectorized RL environments and GRPO group-relative rollouts.
"""

from pathlib import Path
from typing import NamedTuple, Sequence, Union
import numpy as np
import jax
import jax.numpy as jnp


# ==============================================================================
# 1. State Container (Pytree)
# ==============================================================================

class EnvState(NamedTuple):
    """Immutable state container for a single Hangman game."""
    word: jax.Array        # (32,) int32: target word (0=PAD, 1..26='a'..'z')
    guessed: jax.Array     # (26,) bool: True if letter i has been guessed
    lives: jax.Array       # () int32: remaining incorrect attempts allowed
    done: jax.Array        # () bool: True if episode has terminated (win or loss)
    won: jax.Array         # () bool: True if target word was completely solved


# ==============================================================================
# 2. Pure Functional Kernels (Single-Game Transitions)
# ==============================================================================

def env_reset(word: jax.Array, max_lives: int = 6) -> EnvState:
    """Initialize a single game from an encoded word tensor.

    Args:
        word: (32,) int32 array where letters are 1..26 and padding is 0.
        max_lives: Maximum incorrect guesses allowed (standard Hangman = 6).

    Returns:
        EnvState with all 26 letters un-guessed and full lives.
    """
    return EnvState(
        word=word,
        guessed=jnp.zeros(26, dtype=jnp.bool_),
        lives=jnp.array(max_lives, dtype=jnp.int32),
        done=jnp.array(False, dtype=jnp.bool_),
        won=jnp.array(False, dtype=jnp.bool_),
    )


def env_observe(state: EnvState) -> jax.Array:
    """Project hidden state onto what the agent observes.

    Returns:
        (32,) int32 array: 0 for PAD, 1..26 for revealed characters, 27 for '_'.
    """
    is_letter = (state.word >= 1) & (state.word <= 26)
    letter_idx = jnp.clip(state.word - 1, 0, 25)
    is_revealed = state.guessed[letter_idx]
    return jnp.where(is_letter & (~is_revealed), 27, state.word)


def env_action_mask(state: EnvState) -> jax.Array:
    """Return valid action mask where True = available (not yet guessed) letter."""
    return ~state.guessed


def env_step(state: EnvState, action: jax.Array) -> tuple[EnvState, jax.Array]:
    """Execute one letter guess transition.

    Args:
        state: Current EnvState
        action: scalar int32 in [0, 25] ('a'..'z')

    Returns:
        next_state: Updated EnvState (frozen if game was already done)
        reward: scalar float32 (1.0 on winning guess, 0.0 otherwise)
    """
    already_guessed = state.guessed[action]
    in_word = jnp.any(state.word == action + 1)

    # 1. Update lives (deduct only on new incorrect guess when game is active)
    lose_life = (~already_guessed) & (~in_word) & (~state.done)
    new_lives = state.lives - lose_life.astype(jnp.int32)
    new_guessed = state.guessed.at[action].set(True)

    # 2. Check win / loss terminal conditions
    is_letter = (state.word >= 1) & (state.word <= 26)
    letter_idx = jnp.clip(state.word - 1, 0, 25)
    all_revealed = jnp.all(jnp.where(is_letter, new_guessed[letter_idx], True))
    won = state.won | (all_revealed & (new_lives > 0))
    lost = (new_lives <= 0)
    done = state.done | won | lost

    # 3. Reward (1.0 on transition to won, 0.0 otherwise)
    reward = jnp.where(won & (~state.done), 1.0, 0.0)

    # 4. Freeze state if already done before this step
    next_state = EnvState(
        word=state.word,
        guessed=jnp.where(state.done, state.guessed, new_guessed),
        lives=jnp.where(state.done, state.lives, new_lives),
        done=done,
        won=won,
    )
    return next_state, reward


# ==============================================================================
# 3. Vectorized Primitives via jax.vmap
# ==============================================================================

vmap_reset = jax.vmap(env_reset)
vmap_step = jax.vmap(env_step)
vmap_observe = jax.vmap(env_observe)
vmap_action_mask = jax.vmap(env_action_mask)


# ==============================================================================
# 4. Dataset Environment Wrapper (Holds VRAM Buffer & Sampling)
# ==============================================================================

class JaxHangmanEnv:
    """Vectorized Hangman Environment holding the pre-encoded dataset on GPU."""

    def __init__(self, dataset: Union[Sequence[str], Path, str], max_seq_len: int = 32):
        self.max_seq_len = max_seq_len

        # 1. Ingest text dataset
        if isinstance(dataset, (str, Path)):
            with open(dataset, "r") as f:
                raw_words = [line.strip().lower() for line in f if line.strip()]
        else:
            raw_words = [w.strip().lower() for w in dataset]

        self.num_words = len(raw_words)

        # 2. Pre-encode entire dataset once into contiguous GPU integer tensor (~32 MB)
        encoded = np.zeros((self.num_words, max_seq_len), dtype=np.int32)
        for i, w in enumerate(raw_words):
            length = min(len(w), max_seq_len)
            encoded[i, :length] = [ord(c) - ord('a') + 1 for c in w[:length]]

        self.words = jnp.array(encoded)

    def reset_batch(self, key: jax.Array, batch_size: int) -> EnvState:
        """Sample distinct random words directly on GPU without replacement."""
        idx = jax.random.choice(key, self.num_words, shape=(batch_size,), replace=False)
        return vmap_reset(self.words[idx])

    # Convenience alias: env.reset(key, batch_size)
    reset = reset_batch

    def reset_grpo(self, key: jax.Array, num_prompts: int, group_size: int) -> EnvState:
        """Sample distinct prompt words and tile each group_size times for GRPO rollouts.

        Total batch size = num_prompts * group_size.
        """
        idx = jax.random.choice(key, self.num_words, shape=(num_prompts,), replace=False)
        prompt_words = self.words[idx]
        grpo_words = jnp.repeat(prompt_words, group_size, axis=0)
        return vmap_reset(grpo_words)

    def reset_indices(self, indices: jax.Array) -> EnvState:
        """Deterministic reset by dataset indices (for validation / testing)."""
        return vmap_reset(self.words[indices])

    @staticmethod
    def step(states: EnvState, actions: jax.Array) -> tuple[EnvState, jax.Array]:
        """Step all batched environments in parallel."""
        return vmap_step(states, actions)

    @staticmethod
    def observe(states: EnvState) -> jax.Array:
        """Return (B, 32) observations for all batched environments."""
        return vmap_observe(states)

    @staticmethod
    def action_mask(states: EnvState) -> jax.Array:
        """Return (B, 26) boolean action masks for all batched environments."""
        return vmap_action_mask(states)

