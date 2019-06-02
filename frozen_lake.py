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
import algorithm.q_learning as q_learning

ENV_NAME = "FrozenLake-v0"
EVALUATION_EPISODE = 1000
EPISODES = 60000



def frozen_lake():
    env = gym.make(ENV_NAME)
    #env = gym.make(ENV_NAME, is_slippery = False)
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    q_solver = q_learning.TabularQLearning(action_space, observation_space)
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
    plt.title("Frozen Lake 4x4 Tabular Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Avg reward")
    plt.show()


if __name__ == "__main__":
    frozen_lake()