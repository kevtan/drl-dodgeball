import random
import statistics

import torch
import torch.nn.functional as F


def average_state_value(qnet, states):
    """
    Finds the average value of a set of states given a particular policy
    as determined by a specific Q-network.
    """
    return statistics.mean(max(qnet(state)).item() for state in states)

def minibatch_loss(minibatch, qnet, tnet, discount):
    """Calculates the MSE for all experiences in a minibatch of experiences."""
    predictions = [qnet(experience[0])[1] for experience in minibatch]
    targets = [experience[2] + discount * max(tnet(experience[3]).detach()) for experience in minibatch]
    losses = [(prediction - target) ** 2 for prediction, target in zip(predictions, targets)]
    return sum(losses) / len(losses)

def decide_action(qnet, braininfo, train, exploration, action_space_size):
    """
    Choose an action to take given a policy network and BrainInfo object.
    If |train| is True, then 100% exploitation, else epsilon-greedy
    """
    observation = torch.from_numpy(braininfo.vector_observations[0]).float()
    best_action = torch.argmax(qnet(observation)).item()
    if train and random.random() < exploration:
        return random.choice(range(action_space_size))
    return best_action
