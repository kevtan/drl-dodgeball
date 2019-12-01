import statistics
import torch
import torch.nn.functional as F


def average_state_value(qnet, states):
    """
    Finds the average value of a set of states given a particular policy
    as determined by a specific Q-network.
    """
    return statistics.mean(max(qnet(state)).item() for state in states)

def minibatch_loss(minibatch, qnet, discount):
    """Calculates the MSE for all experiences in a minibatch of experiences."""
    predictions = [qnet(experience[0])[1] for experience in minibatch]
    targets = [experience[2] + discount * max(qnet(experience[3])) for experience in minibatch]
    losses = [(prediction - target) ** 2 for prediction, target in zip(predictions, targets)]
    return sum(losses) / len(losses)