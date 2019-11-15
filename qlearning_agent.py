import function_approximators as fa
import itertools
import numpy as np
import random

class QLearningAgent():
    
    def __init__(self, actions, ftn_approximator):
        """
        Creates Q-learning agent.
        ---------------------
        Parameters:
        - actions (list)
        - ftn_approximator (FunctionApproximator)
        - discount (float)
        """
        self.actions = actions
        self.ftn_approximator = ftn_approximator
        # short term memory
        self.last_state = None
        self.last_action = None
    
    def get_action(self, state):
        """
        Asks Q-learning agent to give an action suggestion given state/observation.
        ---------------------
        Parameters:
        - state (np.ndarray): complete or partial observation of the world.
        
        Returns:
        - action (np.ndarray): action suggestion given agent's understanding of the world.
        """
        if random.random() < 0.2:
            return random.choice(self.actions)
        best_action = None
        best_value = -float("inf")
        for action in self.actions:
            value = self.ftn_approximator.predict_q(state, action)
            if value > best_value:
                best_action = action
                best_value = value
            elif value == best_value:
                if random.random() < 0.5:
                    best_action = action
                    best_value = value
        self.last_action = best_action
        self.last_state = state
        return best_action
        
    
    def give_feedback(self, reward, new_state):
        """
        Gives Q-learning agent feedback on the previous action based on environment reward.
        ---------------------
        Parameters:
        - reward (float): reward granted by environment based on last action.
        - new_state (np.ndarray): complete or partial observation of the world.
        """
        if self.last_state is None:
            return
        self.ftn_approximator.update_weights(
            self.last_state,
            self.last_action,
            reward,
            new_state
        )

if __name__ == "__main__":
    actions = set(itertools.product((-1, 0, 1), (-1, 0, 1)))
    approximator = fa.LinearFunctionApproximator(3, 2, actions)
    agent = QLearningAgent(actions, approximator)
    print(f"Before: {agent.ftn_approximator.weights}")
    state = [1, 1, 1]
    action = agent.get_action(state)
    print(f"State: {state}")
    print(f"Action: {action}")
    reward = 1
    new_state = [0, 0, 0]
    print(f"Reward: {reward}")
    agent.give_feedback(reward, new_state)
    print(f"After: {agent.ftn_approximator.weights}")