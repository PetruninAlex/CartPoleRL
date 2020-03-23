import gym
import numpy as np
import matplotlib.pyplot as plt


def generate_random_weigths():
    return np.random.rand(4) * 2 - 1


def choose_action(agent_weights, observation):
    agent_decision = np.dot(agent_weights, observation)
    if agent_decision >= 0:
        action = 0
    else:
        action = 1
    return action


def run_episode(env, agent_weights, episode_length):
    total_reward = 0
    observation = env.reset()
    for _ in range(episode_length):
        action = choose_action(agent_weights, observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            return total_reward
    return total_reward


def random_search_train(env, episode_length, num_episodes):
    for episode in range(num_episodes):
        agent_weights = generate_random_weigths()
        reward = run_episode(env, agent_weights, episode_length)
        if reward == episode_length:
            return episode
    return num_episodes


episode_length = 200
num_of_episodes = 10000
num_of_search = 1000

env = gym.make('CartPole-v0')
num_episodes_per_search = []
for _ in range(num_of_search):
    num_episodes = random_search_train(env, episode_length, num_of_episodes)
    num_episodes_per_search.append(num_episodes)
env.close()

print("The average number of episodes required until the score is 200 is: ", np.average(num_episodes_per_search))
plt.hist(num_episodes_per_search)
plt.savefig('hist_of_num_episodes_per_search.png', dpi=100)
print(
    "The histogram of number of episodes per search was saved in the local folder with the name hist_of_num_episodes_per_search.png")
