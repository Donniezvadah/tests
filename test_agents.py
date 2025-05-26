# bandits/tests/test_agents.py
import unittest
import numpy as np
from bernoulli_bandit import BernoulliBandit
from epsilon import EpsilonGreedyAgent
from ucb import UCBAgent
from thompson import ThompsonSamplingAgent


class TestAgents(unittest.TestCase):
    """
    Test class for checking the basic functionality of the bandit agents.
    """

    def setUp(self):
        """
        Set up a common bandit environment for the tests.
        """
        self.bandit = BernoulliBandit(n_actions=3)
        self.n_actions = self.bandit.action_count

    def test_epsilon_greedy_agent(self):
        """
        Test the EpsilonGreedyAgent's action selection and update methods.
        """
        agent = EpsilonGreedyAgent(epsilon=0.1)
        agent.init_actions(self.n_actions)
        action = agent.get_action()
        self.assertTrue(0 <= action < self.n_actions)
        reward = self.bandit.pull(action)
        agent.update(action, reward)
        self.assertIsNotNone(agent._successes) #Checks that init_actions() was called
        self.assertIsNotNone(agent._failures)

    def test_ucb_agent(self):
        """
        Test the UCBAgent's action selection and update methods.
        """
        agent = UCBAgent()
        agent.init_actions(self.n_actions)
        action = agent.get_action()
        self.assertTrue(0 <= action < self.n_actions)
        reward = self.bandit.pull(action)
        agent.update(action, reward)
        self.assertIsNotNone(agent._successes)  # Checks that init_actions() was called
        self.assertIsNotNone(agent._failures)

    def test_thompson_sampling_agent(self):
        """
        Test the ThompsonSamplingAgent's action selection and update methods.
        """
        agent = ThompsonSamplingAgent()
        agent.init_actions(self.n_actions)
        action = agent.get_action()
        self.assertTrue(0 <= action < self.n_actions)
        reward = self.bandit.pull(action)
        agent.update(action, reward)
        self.assertIsNotNone(agent._successes)  # Checks that init_actions() was called
        self.assertIsNotNone(agent._failures)

    def test_bandit_pull_raises_error_on_invalid_action(self):
        """Tests that the BernoulliBandit raises a ValueError when an invalid action is pulled."""
        bandit = BernoulliBandit(n_actions=3)
        with self.assertRaises(ValueError):
            bandit.pull(5)  # Action 5 is out of bounds (0, 1, 2)


if __name__ == '__main__':
    unittest.main()