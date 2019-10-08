import numpy as np
import random
import time

class Env():
    def __init__(self):
        self.map = ['o', '-', '-', '-', '-', '*']
        self.current_pos = 0
        self.end_pos = 5
        self.done = False

    def reset(self):
        self.done = False
        self.current_pos = 0
        self.map = ['o', '-', '-', '-', '-', '*']

    def update(self, action):
        if action == 1:
            self.current_pos += 1
            if self.current_pos == self.end_pos:
                self.map = ['-', '-', '-', '-', '-', 'o']
            else:
                self.map[self.current_pos], self.map[self.current_pos-1] = self.map[self.current_pos-1], self.map[self.current_pos]
        elif action == 0 and self.current_pos != 0:
            self.current_pos -= 1
            self.map[self.current_pos], self.map[self.current_pos+1] = self.map[self.current_pos+1], self.map[self.current_pos]
        print(self.map)
        return self.done

    def is_done(self):
        if self.current_pos == self.end_pos:
            self.done = True

class RL():
    def __init__(self, env):
        self.q = np.zeros((6, 2))
        self.r = np.array([0, 0, 0, 0, 0, 5])
        self.alpha = 0.1
        self.gamma = 0.1
        self.eposide = 10
        self.env = env

    def get_action(self, current_pos):
        if self.q[current_pos][0] == self.q[current_pos][1]:
            self.action = random.randint(0, 1)
        else:
            self.action = np.argmax(self.q[current_pos])

    def update_q(self, current_pos):
        new_pos = self.get_pos(current_pos)
        self.q[current_pos][self.action] += self.alpha * (self.r[new_pos] + self.gamma * max(self.q[new_pos]) - self.q[current_pos][self.action])

    def get_pos(self, current_pos):
        if current_pos == 0 and self.action == 0:
            return 0
        else:
            return current_pos + 1 if self.action == 1 else current_pos - 1

if __name__ == "__main__":
    env = Env()
    rl = RL(env)
    for i in range(rl.eposide):
        env.reset()
        print("iteration " + str(i))
        print(rl.env.map)
        while rl.env.current_pos != rl.env.end_pos:
            rl.get_action(rl.env.current_pos)
            rl.update_q(rl.env.current_pos)
            done = rl.env.update(rl.action)
            if done:
                break
