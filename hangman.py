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
