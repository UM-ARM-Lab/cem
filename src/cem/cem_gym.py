import heapq
import os

import gym
from cem.envs import setup_env
from cem.policies import setup_policy


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_elite_indicies(num_elite, rewards):
    return heapq.nlargest(num_elite, range(len(rewards)), rewards.take)


def evaluate_theta(theta, theta_index, render, monitor, epoch, monitor_directory, env_name):
    env, _, _ = setup_env(env_name)

    if monitor:
        trial_identifier = "{}-{}".format(theta_index, epoch)
        full_monitor_dir = os.path.join(monitor_directory, trial_identifier)
        env = gym.wrappers.Monitor(env, full_monitor_dir, force=False, video_callable=False)

    policy = setup_policy(env, theta)

    done = False
    observation = env.reset()
    if render:
        env.render()
    rewards = []

    while not done:
        action = policy.act(observation)
        next_observation, reward, done, info = env.step(action)

        rewards.append(reward)
        observation = next_observation
        if render:
            env.render()

    env.close()
    return sum(rewards)
