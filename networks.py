import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):

    def __init__(self, state_space_size, action_space_size):
        super(Linear, self).__init__()
        self.out = nn.Linear(state_space_size, action_space_size)
    
    def forward(self, data):
        return self.out(data)

class DNN(nn.Module):

    def __init__(self, state_space_size, action_space_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(state_space_size, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 10)
        self.out = nn.Linear(10, action_space_size)
    
    def forward(self, data):
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        data = F.relu(self.fc3(data))
        return self.out(data)

class Network1(nn.Module):

    def __init__(self, state_space_size, action_space_size):
        super(Network1, self).__init__()
        self.hidden = nn.Linear(state_space_size, 200)
        self.out = nn.Linear(200, action_space_size)
    
    def forward(self, data):
        data = F.relu(self.hidden(data))
        return self.out(data)

class AtariNetwork(nn.Module):
    """
    CNN network architecture found in this paper:
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def __init__(self, nOut):
        super(Network1, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(2592, 256)
        self.out = nn.Linear(256, nOut)
    
    def forward(self, data):
        data = F.relu(self.conv1(data))
        data = F.relu(self.conv2(data))
        data = data.view(-1)
        data = F.relu(self.fc1(data))
        return self.out(data)

class Network3(nn.Module):

    def __init__(self, state_space_size, action_space_size):
        super(Network3, self).__init__()
        self.fc1 = nn.Linear(state_space_size, 100)
        self.fc2 = nn.Linear(100, 75)
        self.fc3 = nn.Linear(75, 50)
        self.out = nn.Linear(50, action_space_size)
    
    def forward(self, data):
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        data = F.relu(self.fc3(data))
        return self.out(data)