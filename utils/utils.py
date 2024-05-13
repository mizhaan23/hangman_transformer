import torch
import random
import numpy as np


class MyTokenizer:
    def __init__(self, max_length=64):
        start = ord('a')
        alphabets = {'_': 27}
        ids = {27: '_', 0: ''}
        for i in range(26):
            ch = chr(start)
            alphabets[ch] = i + 1
            ids[i + 1] = ch
            start += 1

        self.map = alphabets.copy()
        self.reverse_map = ids.copy()
        self.max_length = max_length

    def encode(self, src):
        """
        Takes in the list (batch) of masked words and encodes them according to their id.
        """
        src_ids = torch.zeros((len(src), self.max_length), dtype=int)
        for k, word in enumerate(src):
            for i, ch in enumerate(word):
                src_ids[k][i] = self.map[ch]
        return src_ids

    def decode(self, token):
        words = []
        bs, seq_len = tuple(token.shape)
        for i in range(bs):
            word = ''
            for token_id in token[i].tolist():
                word += self.reverse_map[token_id]
            words.append(word)
        return words


class MyMasker:
    def __init__(self):
        pass

    def mask(self, src, percentage=None):
        """
        Takes in a list (batch) of words and applies the masking to each.
        """
        return [self._get_mask(word, percentage) for word in src]

    @staticmethod
    def _get_mask(word, perc=None):
        """
        Takes in a words and masks it acc. to the Hangman rules.
        """
        if perc is None:
            perc = np.random.uniform()

        counter = {}
        for i, ch in enumerate(word):
            if ch in counter:
                counter[ch].append(i)
            else:
                counter[ch] = [i]

        for key in random.sample(list(counter.keys()), max(1, int(perc * len(counter)))):
            del counter[key]

        word = ['_'] * len(word)
        for ch in counter:
            for pos in counter[ch]:
                word[pos] = ch

        return ''.join(word)

    # def _get_counter(self, word):
    #     n = len(word)
    #     counter = {}
    #     for i, ch in enumerate(word):
    #         if ch in counter:
    #             counter[ch].append(i)
    #         else:
    #             counter[ch] = [i]
    #     return counter, n

    # def _counter_to_word(self, counter, n=0):
    #     word = ['_']*n
    #     for ch in counter:
    #         for pos in counter[ch]:
    #             word[pos] = ch
    #     return ''.join(word)
    #
    # def _rand_get_mask(self, word, percentage):
    #     if percentage is None:
    #         percentage = np.random.uniform()
    #
    #     n = len(word)
    #     n_masks = max(1, int(percentage*n))
    #     temp = list(range(n))
    #     random.shuffle(temp)
    #     i_masks = set(temp[:n_masks])
    #     masked_word = ''
    #     for i, ch in enumerate(word):
    #         masked_word += ch if i not in i_masks else '_'
    #     return masked_word

    # def _get_mask(self, word, perc=None):
    #     if perc is None:
    #         perc = np.random.uniform()
    #
    #     counter, m = self._get_counter(word)
    #     n = len(counter)
    #     n_masks = max(1, int(perc*n))
    #     for key in random.sample(list(counter.keys()), n_masks):
    #         del counter[key]
    #     return self._counter_to_word(counter, m)
