import random

class ExperienceReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.experiences = []
    
    def __repr__(self):
        representation = ""
        representation += f"Capacity: {self.capacity}\n"
        representation += repr(self.experiences)
        return representation
    
    def add(self, experience):
        self.experiences.append(experience)
        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)
    
    def sample(self, size):
        if size > len(self.experiences):
            return self.experiences
        return random.sample(self.experiences, size)

if __name__ == "__main__":
    erm = ExperienceReplayMemory(10)
    for i in range(100):
        erm.add(i)
    sample = erm.sample(5)