import functools
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

# initialize policy and target network
qnet = NETWORK(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
qnet.load_state_dict(torch.load("networkparams.pt"))
optimizer = torch.optim.SGD(qnet.parameters(), LEARNING_RATE)
tnet = NETWORK(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
tnet.load_state_dict(qnet.state_dict())
tnet.requires_grad_(False)
if not TRAIN:
    qnet.requires_grad_(False)

# experience replay memory
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
            action, q_values = decide_action(qnet, observation)
            new_braininfo = env.step(action)[BRAIN_NAME]
            if TRAIN:
                new_observation, new_reward = utils.unpack_braininfo(new_braininfo)
                with torch.no_grad():
                    ve_action = torch.argmax(qnet(new_observation)).item()
                    target = new_reward + DISCOUNT_RATE * tnet(new_observation)[ve_action]
                loss = (target - q_values[action]) ** 2
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
    temp = qnet.state_dict()
    qnet.load_state_dict(tnet.state_dict())
    tnet.load_state_dict(temp)
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

env.close()
torch.save(qnet.state_dict(), "qnet_parameters.pt")