import baseEngine
import sys
"""
CS 221: Dodgeball Deep Reinforcement Learning
Authors: Andy Khuu(andykhuu) and Kevin Tan(tankev)
This file serves as the uncluttered high level coding file which orchestrates the interactions the set up of the 
game as well as the initialization of the agents. 
"""

def main():
    #baseEngine.run_baseline_game()
    baseEngine.runManualGame()


if __name__ == '__main__':
    sys.exit(main())