import random
import statistics

import torch
import tqdm
from mlagents.envs.environment import UnityEnvironment
from torch.utils.tensorboard import SummaryWriter

import networks
import utils
from experience_replay import ExperienceReplayMemory

# training configuration
ENVIRONMENT = "environments/Basic.app"
EPOCHS = 50
EPISODES = 10
TRAIN = False

# agent hyperparameters
NETWORK = networks.DNN
LEARNING_RATE = 0.005
DISCOUNT_RATE = 0.95
EXPLORATION_RATE = 0.25
EXPLORATION_RATE_DECAY = 0.99
TARGET_NETWORK_UPDATE_INTERVAL = 100
REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 10

# progress tracking and saving
COLLECT_DATA = True
RANDOM_STATES = 10
CHECKPOINT_EPOCHS = 5

# initialize simulation
env = UnityEnvironment(ENVIRONMENT)
bi = env.reset()
BRAIN_NAME = env.external_brain_names[0]
brain_parameters = env.external_brains[BRAIN_NAME]
STATE_SPACE_SIZE = brain_parameters.vector_observation_space_size
ACTION_SPACE_SIZE = brain_parameters.vector_action_space_size[0]

# sample states
random_states = utils.sample_states(env, BRAIN_NAME, ACTION_SPACE_SIZE, RANDOM_STATES)

# initialize policy network, target network, and optimizer
qnet = NETWORK(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
optimizer = torch.optim.SGD(qnet.parameters(), LEARNING_RATE)
tnet = NETWORK(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
tnet.load_state_dict(qnet.state_dict())
tnet.requires_grad_(False)
if not TRAIN:
    qnet.requires_grad_(False)

# initialize agent's experience replay memory
erm = ExperienceReplayMemory(REPLAY_MEMORY_SIZE)

if COLLECT_DATA:
    writer = SummaryWriter()

for epoch in tqdm.tqdm(range(EPOCHS), "Epochs"):
    episode_rewards, episode_lengths = [], []
    for _ in range(EPISODES):
        braininfo = env.reset()[BRAIN_NAME]
        episode_reward, episode_length = 0, 0
        while not (braininfo.local_done[0] or braininfo.max_reached[0]):
            action = utils.decide_action(qnet, braininfo, TRAIN, EXPLORATION_RATE, ACTION_SPACE_SIZE)
            new_braininfo = env.step(action)[BRAIN_NAME]
            observation = torch.from_numpy(braininfo.vector_observations[0]).float()
            new_observation = torch.from_numpy(new_braininfo.vector_observations[0]).float()
            reward = new_braininfo.rewards[0]
            erm.add((observation, action, reward, new_observation))
            if TRAIN:
                minibatch = erm.sample(MINIBATCH_SIZE)
                optimizer.zero_grad()
                loss = utils.minibatch_loss(minibatch, qnet, tnet, DISCOUNT_RATE)
                loss.backward()
                optimizer.step()
            episode_reward += reward
            episode_length += 1
            braininfo = new_braininfo
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    tnet.load_state_dict(qnet.state_dict())
    EXPLORATION_RATE *= EXPLORATION_RATE_DECAY
    if COLLECT_DATA:
        writer.add_scalar("Average_Reward_per_Episode", statistics.mean(episode_rewards), epoch)
        writer.add_scalar("Average_Length_per_Episode", statistics.mean(episode_lengths), epoch)
        writer.add_scalar("Average_State_Value", utils.average_state_value(qnet, random_states), epoch)
    if epoch % CHECKPOINT_EPOCHS == 0:
        torch.save(qnet.state_dict(), path)

env.close()
torch.save(qnet.state_dict(), "qnet_parameters.pt")