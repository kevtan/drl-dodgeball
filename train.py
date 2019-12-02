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
ENVIRONMENT = "environments/Lesson3.app"
EPOCHS = 100
EPISODES = 5
DISCOUNT_RATE = 0.95
EXPLORATION_RATE = 0.3
EXPLORATION_RATE_DECAY = 0.999
RANDOM_STATES = 100
TARGET_NETWORK_UPDATE_INTERVAL = 100
CHECKPOINT_EPOCHS = 5
REPLAY_MEMORY = 100
MINIBATCH_SIZE = 10
COLLECT_DATA = True
TRAIN = True
NETWORK = Network3

# initialize environment simulation
env = UnityEnvironment(ENVIRONMENT)
env.reset()

# extract environment information
BRAIN_NAME = env.external_brain_names[0]        # TODO: add support for >1 brain
brain_parameters = env.external_brains[BRAIN_NAME]
STATE_SPACE_SIZE = brain_parameters.vector_observation_space_size * brain_parameters.num_stacked_vector_observations
ACTION_SPACE_SIZE = brain_parameters.vector_action_space_size[0] # TODO: add support for >1 action space dimension

# sample some random states for tracking training progress
# inspired by https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
random_states = []
for _ in range(RANDOM_STATES):
    action = random.choice(range(ACTION_SPACE_SIZE))
    braininfo = env.step(action)[BRAIN_NAME]
    state = torch.from_numpy(braininfo.vector_observations[0]).float()
    random_states.append(state)

# initialize policy/target network and optimizer
qnet = NETWORK(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
optimizer = torch.optim.SGD(qnet.parameters(), 0.005, 0.9)
identifier = hashlib.md5(f"{ENVIRONMENT}{DISCOUNT_RATE}{REPLAY_MEMORY}{MINIBATCH_SIZE}{str(qnet)}{str(optimizer)}".encode("utf-8")).hexdigest()
path = f"models/{identifier}.pt"
if os.path.exists(path):
    print("Policy network parameters loaded from previous training session.")
    qnet.load_state_dict(torch.load(path))
tnet = NETWORK(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
tnet.load_state_dict(qnet.state_dict())

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
            # select action
            action = utils.decide_action(qnet, braininfo, TRAIN, EXPLORATION_RATE, ACTION_SPACE_SIZE)
            # execute action
            new_braininfo = env.step(action)[BRAIN_NAME]
            # add experience to memory
            observation = torch.from_numpy(braininfo.vector_observations[0]).float()
            new_observation = torch.from_numpy(new_braininfo.vector_observations[0]).float()
            reward = new_braininfo.rewards[0]
            erm.add((observation, action, reward, new_observation))
            if TRAIN:
                # sample minibatch of memories for parameter update
                minibatch = erm.sample(MINIBATCH_SIZE)
                optimizer.zero_grad()
                loss = utils.minibatch_loss(minibatch, qnet, tnet, DISCOUNT_RATE)
                loss.backward()
                optimizer.step()
            # update training metrics
            episode_reward += reward
            episode_length += 1
            # advance to next state
            braininfo = new_braininfo
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    # update target network parameters
    tnet.load_state_dict(qnet.state_dict())
    # exploration rate decay
    EXPLORATION_RATE *= EXPLORATION_RATE_DECAY
    # log training metrics
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
