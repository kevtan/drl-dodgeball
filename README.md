# DRL Dodgeball
DRL (Deep Reinforcement Learning) has led to the creation of agents that exhibit fascinatingly complex and intelligent behavior, especially in the area of computer games like [Dota 2](https://openai.com/blog/dota-2/) and [StarCraft II](https://arxiv.org/pdf/1708.04782.pdf). An exciting area of research for is to develop digital agents that eventually get deployed into physical robots, a task which OpenAI’s [Dactyl project](https://openai.com/blog/learning-dexterity/) demonstrates requires high-fidelity training environments. In this project, we designed a physically realistic simulation with high-dimensional sensory data sources, and trained an agent using a refined deep Q-network in it.

# Files

**train.py**
- Main file used for training, currently training agents with refined DQN
- Command: "py train.py"

 **train_DQN.py** 
- Training agents under the Unity environment with refined DQN
- Command: "py train_DQN.py"

 **train_DoubleDQN.py** 
- Training agents under the Unity environment with Double DQN
- Command: "py train_DoubleDQN.py"

 **train_vanilla.py**
- Training agents under the Unity environment with vanilla Q Learning
- Command : "py train_vanilla.py"

 **utils.py** 
- Modularized utility function which contains abstract functions necessary for training our agent

 **experience_replay.py** 
- Container for the storage and use of memories/experiences according to our prioritized replay buffer.

 **networkparams.pt**
- Configuration file for defining network parameters and topology

 **networks.py**
- Pytorch network definition following the parameters from networkparams.pt

 **PolicyGradients.ipynb** 
- Attempt at using Policy Gradients as an alternative learning mechanism to train intelligent agents on our environment

**QLearning.ipynb** 
- Generalized Q Learning to solve MDP problems

**QLearningAgentAndy.py**
- Implementation of generic Q learning on basic example environments from Unity

**\runs**
- TensorBoard based results of training runs

**\environments**
- Built Unity environments


## Future Work
- Accomodate continuous action spaces
- Acommodate multi-dimensional action spaces
- Accomodate visual observations (first-person and birds-eye)
- Experiment with more robust neural network architectures
- Experiment with different optimizers

![poster](assets/poster.jpg)
