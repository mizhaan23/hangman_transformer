import torch
import random
import numpy as np

class MyTokenizer:
    def __init__(self, max_length=64):
        start = ord('a')
        alphabets = {'_': 27}
        ids = {27:'_', 0:''}
        for i in range(26):
            ch = chr(start)
            alphabets[ch] = i+1
            ids[i+1] = ch
            start += 1

        self.map = alphabets.copy()
        self.reverse_map = ids.copy()
        self.max_length = max_length

    def encode(self, src):
        '''
        src is a list of words in a batch.
        '''
        n = self.max_length
        tokenid = torch.zeros((len(src), n), dtype=int)
        for k, word in enumerate(src):
            for i, ch in enumerate(word):
                ix = self.map[ch]
                tokenid[k][i] = ix
        return tokenid

    def decode(self, token, len_word=None):
        words = []
        bs, seq_len = tuple(token.shape)
        for i in range(bs):
            word=''
            for id in token[i].tolist():
                word += self.reverse_map[id]
            words.append(word)
        return words

class MyMasker():
    def __init__(self):
        pass

    def mask(self, src, percentage=None):
        masked = []
        for word in src:
            mask_word = self._get_mask(word, percentage)
            masked.append(mask_word)
        return masked

    def _get_counter(self, word):
        n = len(word)
        counter = {}
        for i, ch in enumerate(word):
            if ch in counter:
                counter[ch].append(i)
            else:
                counter[ch] = [i]
        return counter, n

    def _counter_to_word(self, counter, n=0):
        word = ['_']*n
        for ch in counter:
            for pos in counter[ch]:
                word[pos] = ch
        return ''.join(word)

    def _rand_get_mask(self, word, percentage):
        if percentage is None:
            percentage = np.random.uniform()

        n = len(word)
        n_masks = max(1, int(percentage*n))
        temp = list(range(n))
        random.shuffle(temp)
        i_masks = set(temp[:n_masks])
        masked_word = ''
        for i, ch in enumerate(word):
            masked_word += ch if i not in i_masks else '_'
        return masked_word

    def _get_mask(self, word, perc=None):
        if perc is None:
            perc = np.random.uniform()

        counter, m = self._get_counter(word)
        n = len(counter)
        n_masks = max(1, int(perc*n))
        for key in random.sample(counter.keys(), n_masks):
            del counter[key]
        return self._counter_to_word(counter, m)
