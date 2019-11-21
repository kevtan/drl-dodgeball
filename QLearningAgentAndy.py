import function_approximators as fa
import itertools
import numpy as np
import random
import math
from collections import defaultdict

class QLearningAgent():
    
    def __init__(self, actions, discount, featureExtractor, explorationProb = 0.2):
        """
        Creates Q-learning agent.
        ---------------------
        Parameters:
        - actions (list)
        - discount (float)
        - featureExtractor (observations)
        - exploration Probability (for epsilon greedy policy)
        - weights (defaultDict(float))
        - numIteration (int)
        """
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0
    
    def getQ(self,state,action):
        score = 0
        for f,v in self.featureExtractor(state,action):
            score += self.weights[f] * v
        return score
    
    def get_action(self, state):
        """
        Asks Q-learning agent to give an action suggestion given state/observation.
        ---------------------
        Parameters:
        - state (np.ndarray): complete or partial observation of the world.
        
        Returns:
        - action (np.ndarray): action suggestion given agent's understanding of the world.
        """
        self.numIters += 1 
        if(random.random() < self.explorationProb):
            return random.choice(self.actions)
        else:
            maxQ = -float('inf')
            bestAction = None
            for action in self.actions:
                if(self.getQ(state,action) > maxQ):
                    maxQ = self.getQ(state,action)
                    bestAction = action
            return bestAction
        
    def getStepSize(self):
        import math
        return 1.0 / math.sqrt(self.numIters)
    
    def give_feedback(self, state, action, reward, newState):
        """
        Gives Q-learning agent feedback on the previous action based on environment reward.
        ---------------------
        Parameters:
        - reward (float): reward granted by environment based on last action.
        - new_state (np.ndarray): complete or partial observation of the world.
        """
        vOpt = 0
        if(len(newState) != 0):
            vOpt = self.getQ(newState, self.actions[0])
            for nextAction in self.actions:
                if(self.getQ(newState,nextAction) > vOpt):
                    vOpt = self.getQ(newState, nextAction)
        target = reward + self.discount * vOpt
        inner = self.getStepSize() * (self.getQ(state,action) - target)
        for f,v in self.featureExtractor(state,action):
            self.weights[f] = self.weights[f] - inner * v

if __name__ == "__main__":
    print("hi")