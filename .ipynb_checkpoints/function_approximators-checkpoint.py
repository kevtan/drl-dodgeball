import itertools
import numpy as np


class LinearFunctionApproximator:
    
    def __init__(self, state_space_size, action_space_size, actions, discount=1, rate=0.1):
        self.weights = np.zeros(state_space_size + action_space_size)
        self.actions = actions
        self.discount = discount
        self.rate = rate
    
    def predict_q(self, state, action):
        if isinstance(action, float) or isinstance(action, int):
            action = np.array([action])
        state_action = np.concatenate((state, action))
        return np.dot(state_action, self.weights)
    
    def update_weights(self, old_state, action, reward, new_state):
        if isinstance(action, float) or isinstance(action, int):
            action = np.array([action])
        prediction = self.predict_q(old_state, action)
        V_opt_hat = max(self.predict_q(new_state, action) for action in self.actions)
        target = reward + self.discount * V_opt_hat
        self.weights -= self.rate * (prediction - target) * np.concatenate((old_state, action))

# class NeuralFunctionApproximator():
    
#     def __init__(self, state_space_size, action_space_size, actions, discount=1, rate=0.1):
#         super(NeuralFunctionApproximator, self).__init__()
#         self.fc1 = nn.Linear(state_space_size + action_space_size, 6)
#         self.fc2 = nn.Linear(6, action_space_size)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x
    


    def predict_q(self, state, action):
        if isinstance(action, float) or isinstance(action, int):
            action = np.array([action])
        state_action = np.concatenate((state, action))
        return np.dot(state_action, self.weights)
    
    def update_weights(self, old_state, action, reward, new_state):
        if isinstance(action, float) or isinstance(action, int):
            action = np.array([action])
        prediction = self.predict_q(old_state, action)
        V_opt_hat = max(self.predict_q(new_state, action) for action in self.actions)
        target = reward + self.discount * V_opt_hat
        self.weights -= self.rate * (prediction - target) * np.concatenate((old_state, action))


if __name__ == "__main__":
    actions = set(itertools.product((-1, 0, 1), (-1, 0, 1)))
    approximator = LinearFunctionApproximator(3, 2, actions)
    print(f"Before: {approximator.weights}")
    state = [1, -1, 1]
    action = [1, 0]
    print(f"State: {state}")
    print(f"Action: {action}")
    q = approximator.predict_q(state, action)
    print(f"Q-value: {q}")
    new_state = [0, 0, 0]
    print(f"New State: {new_state}")
    reward = 1
    print(f"Reward: {reward}")
    # breakpoint()
    approximator.update_weights(state, action, reward, new_state)
    print(f"After: {approximator.weights}")
