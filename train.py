import hashlib
import os
import random
import statistics

import torch
import tqdm
from mlagents.envs.environment import UnityEnvironment
from torch.utils.tensorboard import SummaryWriter

import utils
from experience_replay import ExperienceReplayMemory
from networks import *

# training configuration
ENVIRONMENT = "environments/Lesson1Small-Internal.app"
EPOCHS = 100
EPISODES = 5
DISCOUNT = 0.9
EXPLORATION = 0.4
RANDOM_STATES = 100
CHECKPOINT_EPOCHS = 5
REPLAY_MEMORY = 100
MINIBATCH = 10
COLLECT_DATA = True

# initialize environment simulation
env = UnityEnvironment(ENVIRONMENT)
env.reset()

# extract environment information
BRAIN_NAME = env.external_brain_names[0]
brain_parameters = env.external_brains[BRAIN_NAME]
STATE_SPACE_SIZE = brain_parameters.vector_observation_space_size * brain_parameters.num_stacked_vector_observations
ACTION_SPACE_SIZE = brain_parameters.vector_action_space_size[0]

# sample some random states for tracking training progress
# inspired by https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
random_states = []
for _ in range(RANDOM_STATES):
    action = random.choice(range(ACTION_SPACE_SIZE))
    braininfo = env.step(action)[BRAIN_NAME]
    state = torch.from_numpy(braininfo.vector_observations[0]).float()
    random_states.append(state)

# setup network and optimizer
qnet = Network3(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
optimizer = torch.optim.SGD(qnet.parameters(), 0.005, 0.9)
identifier = hashlib.md5(f"{ENVIRONMENT}{DISCOUNT}{str(qnet)}{str(optimizer)}".encode("utf-8")).hexdigest()
path = f"models/{identifier}.pt"
if os.path.exists(path):
   qnet.load_state_dict(torch.load(path))

# initialize agent's experience replay memory
erm = ExperienceReplayMemory(REPLAY_MEMORY)

# training progress logger
writer = SummaryWriter() if COLLECT_DATA else None

# training loop
for epoch in tqdm.tqdm(range(EPOCHS), "Epochs"):
    episode_rewards, episode_lengths = [], []
    for _ in range(EPISODES):
        braininfo = env.reset()[BRAIN_NAME]
        episode_reward, episode_length = 0, 0
        while not (braininfo.local_done[0] or braininfo.max_reached[0]):
            # epsilon-greedy exploration strategy
            observation = torch.from_numpy(braininfo.vector_observations[0]).float()
            predicted_qs = qnet(observation)
            action = random.choice(range(ACTION_SPACE_SIZE)) if random.random() < EXPLORATION else torch.argmax(predicted_qs).item()
            # execute chosen action
            new_braininfo = env.step(action)[BRAIN_NAME]
            # add experience to memory
            new_observation = torch.from_numpy(new_braininfo.vector_observations[0]).float()
            reward = new_braininfo.rewards[0]
            erm.add((observation, action, reward, new_observation))
            # sample minibatch of memories for parameter update
            minibatch = erm.sample(MINIBATCH)
            loss = utils.minibatch_loss(minibatch, qnet, DISCOUNT)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # update training metrics
            episode_reward += reward
            episode_length += 1
            # advance to next state
            braininfo = new_braininfo
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    # log training metrics after epoch finishes
    if COLLECT_DATA:
        writer.add_scalar("Average_Reward_per_Episode", statistics.mean(episode_rewards), epoch)
        writer.add_scalar("Average_Length_per_Episode", statistics.mean(episode_lengths), epoch)
        writer.add_scalar("Average_State_Value", utils.average_state_value(qnet, random_states), epoch)
    # save network parameters at checkpoints
    if epoch % CHECKPOINT_EPOCHS == 0:
        torch.save(qnet.state_dict(), path)
env.close()

# save network parameters
torch.save(qnet.state_dict(), path)
