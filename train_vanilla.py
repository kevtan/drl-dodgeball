"""
This training script uses vanilla Q-learning without function approximation.
"""
import hashlib
import os
import random
import statistics

import tqdm
from mlagents.envs.environment import UnityEnvironment
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils
from experience_replay import ExperienceReplayMemory
from networks import *

# training configuration
ENVIRONMENT = "environments/Basic.app"
EPOCHS = 100
EPISODES = 10
DISCOUNT_RATE = 0.95
EXPLORATION_RATE = 0.25
EXPLORATION_RATE_DECAY = 1
RANDOM_STATES = 10
CHECKPOINT_EPOCHS = 5
COLLECT_DATA = True
TRAIN = False
LEARNING_RATE = 0.05

# initialize environment simulation
env = UnityEnvironment(ENVIRONMENT)
env.reset()

# extract environment information
BRAIN_NAME = env.external_brain_names[0]        # TODO: add support for >1 brain
brain_parameters = env.external_brains[BRAIN_NAME]
STATE_SPACE_SIZE = brain_parameters.vector_observation_space_size #* brain_parameters.num_stacked_vector_observations
ACTION_SPACE_SIZE = brain_parameters.vector_action_space_size[0] # TODO: add support for >1 action space dimension

# sample some random states for tracking training progress
# inspired by https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
random_states = []
for _ in range(RANDOM_STATES):
    action = random.choice(range(ACTION_SPACE_SIZE))
    braininfo = env.step(action)[BRAIN_NAME]
    state = torch.from_numpy(braininfo.vector_observations[0]).float()
    random_states.append(state)

# create Q-table
Q = torch.zeros((5, 3))

# training progress logger
writer = SummaryWriter() if COLLECT_DATA else None

# training loop
for epoch in tqdm.tqdm(range(EPOCHS), "Epochs"):
    episode_rewards, episode_lengths = [], []
    for _ in range(EPISODES):
        braininfo = env.reset()[BRAIN_NAME]
        episode_reward, episode_length = 0, 0
        while not (braininfo.local_done[0] or braininfo.max_reached[0]):
            # select action
            state = braininfo.vector_observations[0]
            state_encoding = state.nonzero()[0].item()
            if TRAIN and random.random() < EXPLORATION_RATE:
                action = random.randint(1, 2)
            else:
                action = torch.argmax(Q[state_encoding])
            # execute action
            new_braininfo = env.step({BRAIN_NAME: np.array([action])})[BRAIN_NAME]
            # learn from experience
            new_state = new_braininfo.vector_observations[0]
            new_state_encoding = new_state.nonzero()[0].item()
            reward = new_braininfo.rewards[0]
            if TRAIN:
                prediction = Q[state_encoding]
                target = reward + DISCOUNT_RATE * max(Q[new_state_encoding])
                Q[state_encoding] += LEARNING_RATE * (target - prediction)
            # update training metrics
            episode_reward += reward
            episode_length += 1
            # advance to next state
            braininfo = new_braininfo
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    # exploration rate decay
    EXPLORATION_RATE *= EXPLORATION_RATE_DECAY
    # log training metrics
    if COLLECT_DATA:
        writer.add_scalar("Average_Reward_per_Episode", statistics.mean(episode_rewards), epoch)
        writer.add_scalar("Average_Length_per_Episode", statistics.mean(episode_lengths), epoch)
env.close()
