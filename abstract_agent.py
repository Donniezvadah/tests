# bandits/abstract_agent.py
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

class AbstractAgent(metaclass=ABCMeta):
    """
    Abstract base class for reinforcement learning agents that interact with a multi-armed bandit environment.
    Defines the basic interface for agent actions, updates, and state initialization.
    """

    def init_actions(self, n_actions):
        """
        Initializes the agent's internal state (successes and failures counts for each action).

        Args:
            n_actions (int): The number of possible actions.
        """
        self._successes = np.zeros(n_actions)  # Number of successes for each action
        self._failures = np.zeros(n_actions)   # Number of failures for each action
        self._total_pulls = 0

    @abstractmethod
    def get_action(self):
        """
        Returns the agent's chosen action based on its current state and strategy.

        Returns:
            int: The index of the action to take.
        """
        pass

    def update(self, action, reward):
        """
        Updates the agent's internal state based on the observed reward after taking an action.

        Args:
            action (int): The index of the action that was taken.
            reward (int): The reward received (1 for success, 0 for failure).
        """
        self._total_pulls += 1
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1

    @property
    def name(self):
        """
        Returns the name of the agent.
        """
        return self.__class__.__name__