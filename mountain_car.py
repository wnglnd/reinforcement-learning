"""
Solving mountain car using discretized state space
and Q-learning

"""
import gym
import numpy as np
import algorithm.q_learning as ql

ENV_NAME = "MountainCar-v0"
POS_MIN = -1.2
POS_MAX = 0.6
POS_BIN_STEP_SIZE = 0.1
VEL_MIN = -0.07
VEL_MAX = 0.07
VEL_BIN_STEP_SIZE = 0.01
EPISODES = 30000

def discretize_observation_space(continuous_observation):
    pos_bins = np.arange(POS_MIN, POS_MAX, POS_BIN_STEP_SIZE)
    pos_bin = np.digitize(continuous_observation[0], pos_bins)
    pos_bin = np.min([pos_bin, len(pos_bins) - 1])
    vel_bins = np.arange(VEL_MIN, VEL_MAX, VEL_BIN_STEP_SIZE)
    vel_bin = np.digitize(continuous_observation[1], vel_bins)
    vel_bin = np.min([vel_bin, len(vel_bins) - 1])
    return pos_bin, vel_bin

def get_observation_space():
    n_pos = len(np.arange(POS_MIN, POS_MAX, POS_BIN_STEP_SIZE))
    n_vel = len(np.arange(VEL_MIN, VEL_MAX, VEL_BIN_STEP_SIZE))
    return (n_pos, n_vel)

def mountain_car():
    env = gym.make(ENV_NAME)
    action_space = env.action_space.n
    observation_space = get_observation_space()
    q_solver = ql.TabularQLearning(action_space, np.prod(observation_space))

    for episode in range(EPISODES):
        q_solver.update_epsilon()
        state_continuous = env.reset()
        state_discrete = discretize_observation_space(state_continuous)
        state_1d = np.ravel_multi_index(state_discrete, observation_space)
        done = False
        rewards = 0
        while not done:
            action = q_solver.act(state_1d)
            state2_continuous, reward, done, _ = env.step(action)
            state2_discrete = discretize_observation_space(state2_continuous)
            state2_1d = np.ravel_multi_index(state2_discrete, observation_space)
            q_solver.learn(state_1d, action, reward, state2_1d)
            state_1d = state2_1d
            rewards += reward
            if episode % 1000 == 0:
                env.render()
        if episode % 1000 == 0:
            print("Episode: " + str(episode) + ", return = " + str(rewards))




if __name__ == "__main__":
    mountain_car()

