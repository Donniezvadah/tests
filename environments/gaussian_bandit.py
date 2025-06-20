import numpy as np
from .base_environment import BaseEnvironment

def generate_configuration(n_actions=10):
    """
    Generate means and standard deviations for a Gaussian bandit environment.
    This function provides control over the environment parameters for experiments.
    
    Args:
        n_actions (int): Number of actions/arms in the bandit.
        
    Returns:
        tuple: (means, stds) where means and stds are numpy arrays of length n_actions
    """
    # Generate means with good separation for optimal regret
    # Use a logarithmic spacing to ensure good separation between arms
    means = np.logspace(0, 1, n_actions, base=2)
    
    # Use varying standard deviations to test Thompson sampling's ability to handle uncertainty
    # Higher variance for arms with higher means to increase the challenge
    stds = np.linspace(0.5, 2.0, n_actions)
    
    return means, stds

class GaussianBandit(BaseEnvironment):
    def __init__(self, n_actions=10, means=None, stds=None, seed=None):
        """
        Initialize a Gaussian bandit environment.
        
        Args:
            n_actions (int): Number of actions. If means/stds are provided, this should typically match their length 
                             or can be inferred. If means/stds are not provided, this defines the number of arms 
                             for which parameters might be set later (e.g., via set() or internal generation).
            means (list or np.array, optional): List of means for each arm.
            stds (list or np.array, optional): List of standard deviations for each arm.
            seed (int, optional): Seed for the random number generator.
        """
        super().__init__()
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

        if means is not None and stds is not None:
            means_arr = np.array(means, dtype=float)
            stds_arr = np.array(stds, dtype=float)

            if len(means_arr) != len(stds_arr):
                raise ValueError("Length of means and stds must match.")
            
            current_n_actions = len(means_arr)
            # n_actions from parameter is 10 by default. If it's passed as something else AND it mismatches, then error.
            if n_actions != 10 and n_actions != current_n_actions:
                 raise ValueError(
                     f"Explicitly passed n_actions ({n_actions}) "
                     f"does not match length of provided means/stds ({current_n_actions})."
                 )
            self.action_count = current_n_actions

            if not np.all(stds_arr > 0):
                raise ValueError("All standard deviations must be positive.")
            
            self.means = means_arr
            self.stds = stds_arr
            
            self._initial_means = np.copy(self.means)
            self._initial_stds = np.copy(self.stds)
            
            self._optimal_action = np.argmax(self.means)
            self._optimal_mean = self.means[self._optimal_action]
        elif n_actions is not None: # n_actions provided, but means/stds are None
            self.action_count = n_actions
            self.means = None 
            self.stds = None
            self._initial_means = None
            self._initial_stds = None
            self._optimal_action = None
            self._optimal_mean = None
        else: # Neither n_actions nor means/stds provided.
            raise ValueError("GaussianBandit requires n_actions, or both means and stds, for initialization.")
        
    def set(self, means, stds):
        """
        Set the environment parameters.
        
        Args:
            means (list or np.array): List of means for each arm.
            stds (list or np.array): List of standard deviations for each arm.
            
        Raises:
            ValueError: If the lengths don't match or stds are not positive.
        """
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        
        if len(means) != self.action_count:
            raise ValueError(f"Number of means ({len(means)}) must match number of actions ({self.action_count})")
        if len(stds) != self.action_count:
            raise ValueError(f"Number of standard deviations ({len(stds)}) must match number of actions ({self.action_count})")
        if not np.all(stds > 0):
            raise ValueError("All standard deviations must be positive")
            
        self.means = means
        self.stds = stds
        
        # Store initial state
        self._initial_means = np.copy(self.means)
        self._initial_stds = np.copy(self.stds)
        
        # Compute optimal action and value
        self._optimal_action = np.argmax(self.means)
        self._optimal_mean = self.means[self._optimal_action]
        
    def pull(self, action):
        """
        Pull the specified arm and return the reward.
        
        Args:
            action (int): The index of the action to take.
            
        Returns:
            float: The reward from the selected action.
            
        Raises:
            ValueError: If the action is out of bounds or environment not initialized.
        """
        if self.means is None or self.stds is None:
            raise ValueError("Environment not initialized. Call __init__ with means/stds or use set() method if means/stds were not provided during initialization.")
            
        if not (0 <= action < self.action_count):
            raise ValueError(f"Action {action} is out of bounds. Must be between 0 and {self.action_count - 1}")
        
        # Generate reward from normal distribution
        reward = self._rng.normal(self.means[action], self.stds[action])
        return float(reward)
        
    def optimal_reward(self):
        """
        Return the optimal reward (maximum mean).
        
        Returns:
            float: The maximum mean among all actions.
        """
        if self._optimal_mean is None:
            raise ValueError("Environment not initialized. Call set() first.")
        return float(self._optimal_mean)
        
    def reset(self):
        """
        Reset the environment to its initial state.
        """
        if self._initial_means is not None and self._initial_stds is not None:
            self.means = np.copy(self._initial_means)
            self.stds = np.copy(self._initial_stds)
            self._optimal_action = np.argmax(self.means)
            self._optimal_mean = self.means[self._optimal_action]
        
    def step(self):
        """
        Step the environment. In this implementation, it does nothing as it's a stationary environment.
        """
        pass
        
    def get_statistics(self):
        """
        Return statistics about the environment's performance.
        
        Returns:
            dict: Dictionary containing various statistics about the environment.
        """
        if self.means is None or self.stds is None:
            raise ValueError("Environment not initialized. Call set() first.")
        
        total_pulls = np.zeros(self.action_count)
        total_rewards = np.zeros(self.action_count)
        sum_squared_rewards = np.zeros(self.action_count)
        
        for i in range(self.action_count):
            total_pulls[i] = 1  # Assuming one pull per action
            total_rewards[i] = np.random.normal(self.means[i], self.stds[i])
            sum_squared_rewards[i] = total_rewards[i]**2
        
        empirical_means = total_rewards / (total_pulls + 1e-10)
        empirical_stds = np.sqrt(
            (sum_squared_rewards / (total_pulls + 1e-10)) - 
            (total_rewards / (total_pulls + 1e-10))**2
        )
        
        return {
            'total_pulls': total_pulls,
            'total_rewards': total_rewards,
            'sum_squared_rewards': sum_squared_rewards,
            'empirical_means': empirical_means,
            'empirical_stds': empirical_stds
        } 
        
        #this is a comment 