from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    def __init__(self):
        self.action_count = None
        
    @abstractmethod
    def pull(self, action):
        """Pull the specified arm and return the reward."""
        pass
        
    @abstractmethod
    def optimal_reward(self):
        """Return the optimal reward."""
        pass
        
    @abstractmethod
    def reset(self):
        """Reset the environment."""
        pass 