# bandits/bernoulli_bandit.py
import numpy as np
from .base_environment import BaseEnvironment

class BernoulliBandit(BaseEnvironment):
    """
    A Bernoulli bandit with K actions. Each action yields a reward of 1 with probability theta_k
    and 0 otherwise, where theta_k is unknown to the agent but fixed over time.
    """

    def __init__(self, n_actions=10, probs=None):
        """
        Initializes the Bernoulli bandit.

        Args:
            n_actions (int): The number of available actions (arms).
            probs (list or np.array): Optional array of probabilities for each action.
        """
        super().__init__()
        if probs is not None:
            # Convert probs to numpy array if it's a list
            probs = np.array(probs, dtype=float)
            if len(probs) != n_actions:
                raise ValueError(f"Number of probabilities ({len(probs)}) must match number of actions ({n_actions})")
            if not np.all((probs >= 0) & (probs <= 1)):
                raise ValueError("All probabilities must be between 0 and 1")
            self._probs = probs
        else:
            self._probs = np.random.random(n_actions)
            
        self._initial_probs = np.copy(self._probs)
        self.action_count = n_actions
        
    def pull(self, action):
        """
        Simulates pulling a lever (taking an action) and returns a reward.

        Args:
            action (int): The index of the action to take.

        Returns:
            float: 1.0 if a random number is less than the action's probability, 0.0 otherwise.
        """
        if not (0 <= action < self.action_count):
            raise ValueError(f"Action {action} is out of bounds. Must be between 0 and {self.action_count - 1}")

        return float(np.random.random() < self._probs[action])

    def optimal_reward(self):
        """
        Returns the expected reward of the optimal action. Used for regret calculation.

        Returns:
            float: The maximum probability among all actions.
        """
        return float(np.max(self._probs))

    def step(self):
        """
        Used in non-stationary versions of the bandit to change the probabilities.
        This implementation is stationary, so this method does nothing.
        """
        pass  # Stationary bandit, so no need to change probabilities

    def reset(self):
        """Resets the bandit to its initial state (initial probabilities)."""
        self._probs = np.copy(self._initial_probs)