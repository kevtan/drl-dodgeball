# Related Work

## Emergent Tool Use from Multi-Agent Interaction (OpenAI)

Explicitly setting reward functions for behaviours you want to see or you think would be helpful in accomplishing a task is difficult and not scalable. Rely on *autocurricula*, where improvements by one agent forces the others to improve. This is more resemblant of how natural selection works, and is a far more scalable solution. *Multi-agent interaction* is a key to self-supervised skill acquisition.

In the beginning preparation phase, all agents get 0 reward. After the preparation phases, at each timestep:
- Hiders get +1 reward if all hiders are hidden and -1 otherwise
- Seeks get the opposite reward of the hiders (zero-sum game)

Agents observer the following information:
- x and v of self
- x and v of other agents
- measurements from 30 lidar-like range sensors
- x, v, and size of boxes
- x and v of ramps


- Position, velocity, and size of objects
- Only entities in LOS and 135° visual cone can be percieved
- How much time is left in preparation phases
- What the teams are

Actions can perform the following actions:
- MOVE: set a force along their x and y axes + torque on their z axis
- GRAB: binary action to grab objects, which binds object to agent
- LOCK: binary action to fix the position of an object
Grabbing and locking only work on objects in front of the agent and within a small radius.

Agent policies consist of two neural networks:
1. Policy network: produce an action distribution
2. Critic network: predicts discounted future rewards

To encourage collaboration between hiders and seekers (within their own groups obviously), rewards are team-based not individual-based.

Simulating and training the agents was incredibly compute intensive and time intensive, and varied with respect to batch size. This is because hide-and-seek agents require an enormous amount of experience to progress through the 6 stages of strategies.
- 32k batch size required 98.8 hrs
- 64k batch size required 34.0 hrs
- 128k batch size required 19.8 hrs

## Intrinsic Motivation

A common paradigm for unsupervised exploration and skill acquisition. Basically you have the agent keep track of the states it has visited and make it want to explore new states; however, when the state space gets very large, it's very hard to find a good policy this way.

## Soft Actor Critic



## Proximal Policy Optimization Algorithms

Paper: https://arxiv.org/abs/1707.06347

This is a way to use data in the form for training episodes in order to update the parameters of the policy to make it produce better action suggestions in the future.

* PPO is the default reinforcement learning algorithm at OpenAI because it is super easy to use and performs comparably or better than state-of-the-art approaches.

> **Sample complexity** is the number of training samples needed by a machine learning algorithm in order to successfully learn a target function (a stochastic classifier or deterministic regressor).

* This paper seeks to improve the current state of affairs (Deep Q learning, vanilla policy gradient methods, and TRPO) by introducing an algorithm that attains the **data efficiency** and reliable performance of TRPO, while using only first-order optimization. We propose a novel objective with clipped probability ratios, which forms a pessimistic estimate (i.e., lower bound) of the performance of the policy. To optimize policies, we alternate between sampling data from the policy and performing several epochs of optimization on the sampled data.
* 

## Soft Actor Critic (SAC)

https://bair.berkeley.edu/blog/2018/12/14/sac/

## Generalized Advantage Estimation (GAE)

## Attention is All You Need (Ashish Vaswani et al)

## Evaluating Policies

Often times just looking at the reward over time is not a good metric for how your agents are performing. There are other ways of measuring how good your agent's policies are relative to their past policies or policies in some population:
1. ELO
2. TrueSkill

## Partially Observable MDPs



## Unity: A General Platform for Intelligent Agents

Paper: https://arxiv.org/pdf/1809.02627.pdf

* Simulating reinforcement learning agents in **physically realistic environments** (with 3D physics engines) is important so that the models trained in simulation can be transferred into real world settings. The efficacy of this approach has been shown by OpenAI who trained autonomous robots in simulation and was able to transfer their models to a real-world robot with great results.
* As learned models are able to perform increasingly complex tasks, the **complexity of the simulated environments themselves should increase** in order to continue to meaningfully challenge the algorithms and methods being explored. We identify the key dimensions of complexity we believe are essential for the continued development of AI systems: **sensory**, **physical**, **cognitive**, and **social**.
* In addition to complexity, there are additional practical constraints imposed by the simulator itself which must be taken into consideration when designing environments and experimental pipelines. Specifically, simulation environments 3 must be **flexibly controlled** by the researcher, and they must **run in a fast and distributed manner** in order to provide the iteration time required for experimental research. It is essential that an environment be able to provide greater than real-time simulation speed. It is possible to increase Unity ML-Agents simulations up to **one hundred times real-time** because Unity gives extensive control over rendering quality and the ability to turn off rendering completely when it is not needed (e.g. when your agent only has ray-cast based sensors).

>  **Curriculum learning** entails initially providing a simplified version of a task to an agent, and slowly increasing the task complexity as the agent’s performance increases. Implicitly requires the researcher to have the ability to modify the environment and build various iterations.
>
> **Domain Randomization** is another technique requiring flexible control of the simulator. The approach involves introducing enough variability into the simulation so that the models learned within the simulation can generalize to the real world. Requires the researcher to fiddle with lighting, textures, physics, and object placement within a scene.

* A Unity Project consists of a collection of Assets. These typically correspond to files within the Project. Scenes are a special type of Asset which define the environment or level of a Project. Scenes contain a definition of a hierarchical composition of **GameObjects**, which correspond to the actual objects, either physical or purely logical, within the environment. The behavior and function of each GameObject is determined by the **components** attached to it. There are a variety of built-in components provided with the Unity Editor, including **Cameras**, Meshes, Renderers, **RigidBodies**, and many others. It is also possible to define custom components using C# scripts or external plugins.
* The **ML-Agents toolkit** is an open source project which enables researchers and developers to create simulation environments using the Unity Editor and interact with them using a Python API. make it a potentially strong research platform. The toolkit provides a set of core and additional functionalities. The core functionality enables developers and researchers to define environments with the Unity Editor and associated C# scripts, and then expose these built environments to a straightforward Python API for interaction. Additional functionality includes a set of example environments and baseline algorithms.
* The toolkit is comprised of two parts:
  * Once the **ML-Agents SDK** is imported into the Unity project, Scenes within Unity can be made into learning environments. This happens through the use of three entities available in the SDK: Agent, Brain, and Academy. The Agents are responsible for collecting observations and taking actions. Each Brain is responsible for providing a policy for decision making for each of its associated agents. The Academy is responsible for global coordination of the environment simulation.
  * The provided Python package contains a class called **UnityEnvironment** that can be used to launch and interface with Unity executables (as well as the Editor) which contain the required components described above. There are 4 main functions (see paper). **BrainInfo** is a class which contains lists of observations, previous actions, rewards, and miscellaneous state variables. At each simulation step, BrainInfo contains information for all active agents within the scene which require decisions in order for the simulation to proceed.