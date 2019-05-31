import gym
import numpy as np

ENV_NAME = "FrozenLake-v0"
S_TERMINAL = 15

GAMMA = 1
EPSILON = 0.1
ALPHA = 0.1

EPISODES = 300

class TabularQLearning:

    def __init__ (self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.Q = np.random.randn(observation_space, action_space)
        self.Q[15,:] = 0

    def learn(self, state, action, reward, state2):
        self.Q[state, action] += ALPHA * (reward + GAMMA * np.argmax(self.Q[state2,:]))

    def act(self, state):
        if self.take_random_action() == True:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.Q[state,:])
        return action

    def take_random_action(self):
        if np.random.rand() < EPSILON:
            return True
        else:
            return False



def frozen_lake():
    env = gym.make(ENV_NAME)
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    q_solver = TabularQLearning(action_space, observation_space)
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        t = 0
        while not done:
            action = q_solver.act(state)
            state2, reward, done, _ = env.step(action)
            q_solver.learn(state, action, reward, state2)
            state = state2
            if done:
                print("Episode: " + str(episode) + " done after " + str(t) + " steps")
            t += 1



if __name__ == "__main__":
    frozen_lake()