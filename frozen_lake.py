"""
Tabular Q-learning algorithm for solving the 4x4 Frozen Lake environment
from Open AI Gym. 

Using the standard environment where the ice is slippery it receives about
0.08 average reward per episode. Oddly enough the performance drops to 0 if
EPSILON_MIN is set to 0..

"""
import gym
import numpy as np
import matplotlib.pyplot as plt

ENV_NAME = "FrozenLake-v0"

GAMMA = 1
EPSILON_INIT = 1.0
EPSILON_MIN = 0.1
ALPHA = 0.1

EVALUATION_EPISODE = 1000
EPSILON_MIN_EPISODE = 29000
EPISODES = 60000

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

def frozen_lake():
    env = gym.make(ENV_NAME)
    #env = gym.make(ENV_NAME, is_slippery = False)
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    q_solver = TabularQLearning(action_space, observation_space)
    rewards = 0
    episodes = np.zeros((1,1))
    avg_reward = np.zeros((1,1))

    for episode in range(EPISODES):
        done = False
        t = 0
        if episode % EVALUATION_EPISODE == 0:
            if episode > 0:
                print("Average reward (t = " + str(episode) + "): " + str(rewards / EVALUATION_EPISODE))
                episodes = np.append(episodes, episode)
                avg_reward = np.append(avg_reward, rewards / EVALUATION_EPISODE)
                rewards = 0
        q_solver.update_epsilon()
        state = env.reset()
        while not done:
            action = q_solver.act(state)
            state2, reward, done, _ = env.step(action)
            q_solver.learn(state, action, reward, state2)
            state = state2
            t += 1
        rewards += reward
    plt.plot(episodes, avg_reward)
    plt.title("Frozen Lake 4x4 Tabular Q-learning\nAlpha = " + str(ALPHA))
    plt.xlabel("Episodes")
    plt.ylabel("Avg reward")
    plt.show()


if __name__ == "__main__":
    frozen_lake()