import functools
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
ENVIRONMENT = "environments/Lesson1SmallLimited.app"
EPOCHS = 50
EPISODES = 10
TRAIN = False

# agent hyperparameters
NETWORK = networks.Network3
LEARNING_RATE = 0.005
DISCOUNT_RATE = 0.95
EXPLORATION_RATE = 1
EXPLORATION_RATE_DECAY = 1
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
qnet.load_state_dict(torch.load("qnet_parameters.pt"))
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

# epsilon-greedy exploration strategy
decide_action = functools.partial(
    utils.decide_action,
    train=TRAIN,
    exploration=EXPLORATION_RATE,
    action_space_size=ACTION_SPACE_SIZE
)

for epoch in tqdm.tqdm(range(EPOCHS), "Epochs"):
    episode_rewards, episode_lengths = [], []
    for _ in range(EPISODES):
        braininfo = env.reset()[BRAIN_NAME]
        episode_reward, episode_length = 0, 0
        while True:
            observation, reward = utils.unpack_braininfo(braininfo)
            with torch.no_grad():
                action, q_values = decide_action(qnet, observation)
            new_braininfo = env.step(action)[BRAIN_NAME]
            new_observation, new_reward = utils.unpack_braininfo(new_braininfo)
            if TRAIN:
                erm.add((observation, action, new_reward, new_observation))
                minibatch = erm.sample(MINIBATCH_SIZE)
                for s, a, r, s_prime in minibatch:
                    prediction = qnet(s)[a]
                    with torch.no_grad():
                        target = r + DISCOUNT_RATE * max(tnet(s_prime)).item()
                    loss = (target - prediction) ** 2
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            episode_reward += reward
            episode_length += 1
            if braininfo.local_done[0] or braininfo.max_reached[0]:
                print(reward)
                break
            braininfo = new_braininfo
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    tnet.load_state_dict(qnet.state_dict())
    EXPLORATION_RATE *= EXPLORATION_RATE_DECAY
    if COLLECT_DATA:
        with torch.no_grad():
            print("V([1, 0, 0, 0, 0, 0]) = ", max(qnet(torch.tensor([1., 0, 0, 0, 0, 0]))))
            print("V([0, 1, 0, 0, 0, 0]) = ", max(qnet(torch.tensor([0., 1, 0, 0, 0, 0]))))
            print("V([0, 0, 1, 0, 0, 0]) = ", max(qnet(torch.tensor([0., 0, 1, 0, 0, 0]))))
            print("V([0, 0, 0, 1, 0, 0]) = ", max(qnet(torch.tensor([0., 0, 0, 1, 0, 0]))))
            print("V([0, 0, 0, 0, 1, 0]) = ", max(qnet(torch.tensor([0., 0, 0, 0, 1, 0]))))
            print("V([0, 0, 0, 0, 0, 1]) = ", max(qnet(torch.tensor([0., 0, 0, 0, 0, 1]))))
        writer.add_scalar("Average_Reward_per_Episode", statistics.mean(episode_rewards), epoch)
        writer.add_scalar("Average_Length_per_Episode", statistics.mean(episode_lengths), epoch)
        writer.add_scalar("Average_State_Value", utils.average_state_value(qnet, random_states), epoch)

env.close()
torch.save(qnet.state_dict(), "qnet_parameters.pt")
