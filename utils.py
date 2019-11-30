import statistics

def average_state_value(qnet, states):
    """
    Finds the average value of a set of states given a particular policy
    as determined by a specific Q-network.
    """
    return statistics.mean(max(qnet(state)).item() for state in states)