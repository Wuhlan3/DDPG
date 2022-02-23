import numpy as np

class Replay_memory():
    def __init__(self, capacity):
        self.mem = []
        self.capacity = capacity
        self.pos = 0

    def push(self, data):
        if len(self.mem) < self.capacity:
            self.mem.append(data)
        else:
            self.mem[int(self.pos)] = data
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        state = []
        next_state = []
        action = []
        reward = []
        done = []
        pos = np.random.randint(0, len(self.mem), size=batch_size)
        
        for i in pos:
            s, next_s, a, r, d = self.mem[i]
            state.append(np.array(s, copy=False))
            next_state.append(np.array(next_s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(next_state), np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)
