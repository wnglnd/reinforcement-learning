import numpy as np

GAMMA = 1
EPSILON_INIT = 1.0
EPSILON_MIN = 0.1
EPSILON_MIN_EPISODE = 29000
ALPHA = 0.1

class TabularQLearning:

    def __init__ (self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.Q = np.zeros((observation_space, action_space))
        self.epsilon = EPSILON_INIT

    def learn(self, state, action, reward, state2):
        target = reward + GAMMA * np.max(self.Q[state2, :])
        predict = self.Q[state, action]
        self.Q[state, action] = predict + ALPHA * (target - predict)

    def act(self, state, exploring = True):
        if self.take_random_action(exploring) == True:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.Q[state,:])
        return action

    def take_random_action(self, exploring):
        if exploring == True:
            if np.random.rand() < self.epsilon:
                return True
            else:
                return False
        else:
            return False

    def update_epsilon(self):
        self.epsilon = np.max([self.epsilon - (EPSILON_INIT - EPSILON_MIN)/EPSILON_MIN_EPISODE, EPSILON_MIN])