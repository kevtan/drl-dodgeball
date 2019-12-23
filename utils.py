import random
import statistics

import torch
import torch.nn.functional as F

def sample_states(env, brain_name, action_space_size, n_states):
    """
    Sample some random states for tracking training progress.
    Inspiration: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    random_states = []
    for _ in range(n_states):
        action = random.choice(range(action_space_size))
        braininfo = env.step(action)[brain_name]
        state = torch.from_numpy(braininfo.vector_observations[0]).float()
        random_states.append(state)
    return random_states


def unpack_braininfo(braininfo):
    """
    Unpacks a BrainInfo object into its vector observation and scalar reward.

    Returns:
    - observation (torch.FloatTensor)
    - reward (float)
    """
    observation = torch.from_numpy(braininfo.vector_observations[0]).float()
    reward = braininfo.rewards[0]
    return observation, reward

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

def decide_action(qnet, observation, train, exploration, action_space_size):
    """
    Choose an action to take given a policy network and observation.
    If |train| is True, then 100% exploitation, else epsilon-greedy

    Returns a tuple action, best_action in order to save computation effort
    """
    q_values = qnet(observation)
    best_action = torch.argmax(q_values).item()
    if train and random.random() < exploration:
        return random.choice(range(action_space_size)), q_values
    return best_action, q_values
