import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.utils import shuffle


class Hangman:
    def __init__(self, penalty=-1, reward=0, n_episode=6):
        self.penalty = penalty
        self.reward = reward
        self.n_episode = n_episode

    def reset(self, code):
        self.timer = 0
        self.done = False
        self.state = ['_'] * len(code)
        self.answer = self._get_counter(list(code))

    def _get_counter(self, code):
        dict_code = {}
        for i, ch in enumerate(code):
            if ch in dict_code:
                dict_code[ch].append(i)
            else:
                dict_code[ch] = [i]
        return dict_code

    def get_state(self):
        return self.state

    def step(self, guess):
        state = self.state
        counter = self.answer

        if self.done:
            return state, 0, self.done

        if self.timer == self.n_episode:
            self.done = True

        if guess not in counter:
            self.timer += 1
            return state, self.penalty, self.done

        r = self.reward
        for pos in counter[guess]:
            state[pos] = guess

        del counter[guess]
        if not bool(counter):
            self.done = True

        self.answer = counter
        self.state = state
        return state, r, self.done


class HangmanEnv(gym.Env):
    def __init__(self, dataloader, max_seq_len=32, init_counter=0):
        super(HangmanEnv, self).__init__()

        self.dataset = shuffle(dataloader.dataset)
        self.counter = init_counter
        self.max_seq_len = max_seq_len
        self.action_space = spaces.Discrete(28)  # 26 possible actions (a-z) + '' + '_'
        self.observation_space = spaces.Box(low=0, high=27, shape=(self.max_seq_len,), dtype=int)

        self.hidden_word = None
        self.word_length = None
        self._reset_attributes()
    
    def _reset_attributes(self):
        self.guessed_letters = set()
        self.remaining_attempts = 6  # Maximum attempts
        self.current_state = np.zeros(self.max_seq_len, dtype=int)  # Initial state
        self.game_over = False

    def reset(self, *, seed=0, options=None):
        self.hidden_word = self.dataset[self.counter % len(self.dataset)]
        self.word_length = len(self.hidden_word)
        self._reset_attributes()
        
        # Increment reset counter
        self.counter += 1

        current_word = ''.join([char if char in self.guessed_letters else '_' for char in self.hidden_word])
        self.current_state = self.word2state(current_word)
        return self.current_state, \
            {
            'word': current_word, 
            'hidden_word': self.hidden_word, 
            'guessed_letters': self.guessed_letters,
            'remaining_attempts': self.remaining_attempts,
        }

    def generate_random_word(self):
        # Replace this with your logic for generating random words
        word_list = self.dataset
        idx = self.counter % len(word_list)
        self.counter += 1
        return word_list[idx]

    def step(self, action):
        if action in self.guessed_letters:
            print("You have already guessed that letter.")
        else:
            self.guessed_letters.add(action)
            if action in self.hidden_word:
                reward = 0
            else:
                reward = 0
                self.remaining_attempts -= 1

        if set(self.hidden_word) <= self.guessed_letters or self.remaining_attempts == 0:
            reward = 1 if set(self.hidden_word) <= self.guessed_letters else 0
            self.game_over = True

        current_word = ''.join([char if char in self.guessed_letters else '_' for char in self.hidden_word])
        self.current_state = self.word2state(current_word)
        return self.current_state, reward, self.game_over, self.game_over, \
            {
            'word': current_word, 
            'hidden_word': self.hidden_word, 
            'guessed_letters': self.guessed_letters,
            'remaining_attempts': self.remaining_attempts,
        }

    def word2state(self, word):
        state = [27 if char == '_' else ord(char) - ord('a') + 1 for char in word]
        while len(state) < self.max_seq_len:
            state.append(0)
        return state
