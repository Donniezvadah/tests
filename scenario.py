import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import seaborn as sns
# from agents.llm_agent import LLMAgent  # Commented out LLM agent
from agents.epsilon import EpsilonGreedyAgent
from agents.gaussian_epsilon_greedy import GaussianEpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.gaussian_ucb import GaussianUCBAgent
from agents.thompson import ThompsonSamplingAgent
from agents.gaussian_thompson_sampling import GaussianThompsonSamplingAgent
from environments.bernoulli_bandit import BernoulliBandit
from environments.gaussian_bandit import GaussianBandit
import os

sns.set_theme(style="whitegrid")

class BanditScenario:
    """
    A class to manage different bandit scenarios and evaluate agent performance.
    """
    
    def __init__(self, n_arms: int = 2, n_trials: int = 100, n_episodes: int = 100):
        """
        Initialize the bandit scenario.
        
        Args:
            n_arms: Number of arms in the bandit
            n_trials: Number of trials per episode
            n_episodes: Number of episodes to run
        """
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.n_episodes = n_episodes
        
        # Initialize agents for Bernoulli bandit
        self.bernoulli_agents = {
            'Epsilon-Greedy': EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli'),
            'UCB': UCBAgent(),
            'Thompson Sampling': ThompsonSamplingAgent(environment_type='bernoulli'),
            # 'LLM': LLMAgent(model="gpt-4.1-nano"),  # Commented out LLM agent
        }
        
        # Initialize agents for Gaussian bandit
        self.gaussian_agents = {
            'Epsilon-Greedy': GaussianEpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1),
            'UCB': GaussianUCBAgent(n_arms=n_arms),
            'Thompson Sampling': GaussianThompsonSamplingAgent(n_arms=n_arms),
            # 'LLM': LLMAgent(model="gpt-4.1-nano"),  # Commented out LLM agent
        }
        
        # Set up plotting style
        # plt.style.use('seaborn')  # Removed due to error
        sns.set_palette("husl")
        
        self.plot_dir = 'scenarioplots'
        os.makedirs(self.plot_dir, exist_ok=True)
        
    def _sample_bernoulli_params(self, scenario: str) -> Tuple[float, float]:
        """
        Sample parameters for Bernoulli bandit based on scenario.
        
        Args:
            scenario: One of 'easy', 'medium', 'hard', 'uniform'
            
        Returns:
            Tuple of (p1, p2) probabilities
        """
        if scenario == 'uniform':
            p1 = np.random.uniform(0, 1)
            p2 = np.random.uniform(0, 1)
        elif scenario == 'easy':
            p1 = np.random.choice([0.1, 0.9])
            p2 = 1 - p1
        elif scenario == 'medium':
            p1 = np.random.choice([0.25, 0.75])
            p2 = 1 - p1
        elif scenario == 'hard':
            p1 = np.random.choice([0.4, 0.6])
            p2 = 1 - p1
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
            
        return p1, p2
        
    def _sample_gaussian_params(self, scenario: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Sample parameters for Gaussian bandit based on scenario.
        
        Args:
            scenario: One of 'easy', 'medium', 'hard', 'uniform'
            
        Returns:
            Tuple of ((mean1, std1), (mean2, std2))
        """
        if scenario == 'uniform':
            mean1 = np.random.uniform(-1, 1)
            mean2 = np.random.uniform(-1, 1)
            std1 = np.random.uniform(0.1, 0.5)
            std2 = np.random.uniform(0.1, 0.5)
        elif scenario == 'easy':
            mean1 = np.random.choice([-0.8, 0.8])
            mean2 = -mean1
            std1 = std2 = 0.1
        elif scenario == 'medium':
            mean1 = np.random.choice([-0.5, 0.5])
            mean2 = -mean1
            std1 = std2 = 0.2
        elif scenario == 'hard':
            mean1 = np.random.choice([-0.2, 0.2])
            mean2 = -mean1
            std1 = std2 = 0.3
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
            
        return (mean1, std1), (mean2, std2)
        
    def run_bernoulli_scenario(self, scenario: str):
        results = {agent_name: np.zeros((self.n_episodes, self.n_trials)) for agent_name in self.bernoulli_agents.keys()}
        optimal_per_episode = np.zeros(self.n_episodes)
        for episode in range(self.n_episodes):
            p1, p2 = self._sample_bernoulli_params(scenario)
            env = BernoulliBandit(n_actions=2, probs=[p1, p2])
            optimal_per_episode[episode] = max(p1, p2)
            for agent_name, agent in self.bernoulli_agents.items():
                agent.init_actions(self.n_arms)
                rewards = []
                for _ in range(self.n_trials):
                    action = agent.get_action()
                    reward = env.pull(action)
                    agent.update(action, reward)
                    rewards.append(reward)
                results[agent_name][episode, :] = rewards
        return results, optimal_per_episode
        
    def run_gaussian_scenario(self, scenario: str):
        results = {agent_name: np.zeros((self.n_episodes, self.n_trials)) for agent_name in self.gaussian_agents.keys()}
        optimal_per_episode = np.zeros(self.n_episodes)
        for episode in range(self.n_episodes):
            (mean1, std1), (mean2, std2) = self._sample_gaussian_params(scenario)
            env = GaussianBandit(n_actions=2)
            env.set([mean1, mean2], [std1, std2])
            optimal_per_episode[episode] = max(mean1, mean2)
            for agent_name, agent in self.gaussian_agents.items():
                agent.init_actions(self.n_arms)
                rewards = []
                for _ in range(self.n_trials):
                    action = agent.get_action()
                    reward = env.pull(action)
                    agent.update(action, reward)
                    rewards.append(reward)
                results[agent_name][episode, :] = rewards
        return results, optimal_per_episode
        
    def plot_cumulative_regret(self, results, optimal_per_episode, scenario, env_type):
        plt.figure(figsize=(7, 5))
        for agent_name, rewards in results.items():
            regret = np.cumsum(optimal_per_episode[:, None] - rewards, axis=1)
            mean_regret = regret.mean(axis=0)
            plt.plot(mean_regret, label=agent_name, linestyle='--')  # Use dashed lines
        plt.title(f'Cumulative Regret - {env_type} - {scenario.capitalize()}')
        plt.xlabel('Trial #')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'cumulative_regret_{env_type}_{scenario}.png'), dpi=200)
        plt.close()

    def plot_heatmap(self, regret_matrix: np.ndarray, row_labels: list, col_labels: list, title: str, filename: str):
        plt.figure(figsize=(6, 5))
        sns.heatmap(regret_matrix, annot=True, fmt='.2f', cmap='RdPu', xticklabels=col_labels, yticklabels=row_labels, cbar_kws={'label': 'Cumulative Regret'})
        plt.title(title)
        plt.xlabel('Agent')
        plt.ylabel('Training Condition')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=200)
        plt.close()

    def plot_easy_hard_subplots(self, easy_results, easy_optimal, hard_results, hard_optimal, env_type):
        plt.figure(figsize=(12, 5))
        scenarios = ['easy', 'hard']
        results_list = [easy_results, hard_results]
        optimal_list = [easy_optimal, hard_optimal]
        for i, (results, optimal, scenario) in enumerate(zip(results_list, optimal_list, scenarios)):
            plt.subplot(1, 2, i+1)
            for agent_name, rewards in results.items():
                regret = np.cumsum(optimal[:, None] - rewards, axis=1)
                mean_regret = regret.mean(axis=0)
                plt.plot(mean_regret, label=agent_name, linestyle='--')
            plt.title(f'{env_type} - {scenario.capitalize()}')
            plt.xlabel('Trial #')
            plt.ylabel('Cumulative Regret')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{env_type.lower()}_easy_hard_subplot.png'), dpi=200)
        plt.close()

def main():
    scenarios = ['easy', 'medium', 'hard', 'uniform']
    bandit_scenario = BanditScenario(n_trials=1000, n_episodes=200)
    regret_summary = {'Bernoulli': [], 'Gaussian': []}
    bernoulli_results_dict = {}
    bernoulli_optimal_dict = {}
    gaussian_results_dict = {}
    gaussian_optimal_dict = {}
    for scenario in scenarios:
        print(f"\nRunning {scenario} scenario...")
        # Bernoulli
        bernoulli_results, bernoulli_optimal = bandit_scenario.run_bernoulli_scenario(scenario)
        bandit_scenario.plot_cumulative_regret(bernoulli_results, bernoulli_optimal, scenario, 'Bernoulli')
        bernoulli_results_dict[scenario] = bernoulli_results
        bernoulli_optimal_dict[scenario] = bernoulli_optimal
        # Gaussian
        gaussian_results, gaussian_optimal = bandit_scenario.run_gaussian_scenario(scenario)
        bandit_scenario.plot_cumulative_regret(gaussian_results, gaussian_optimal, scenario, 'Gaussian')
        gaussian_results_dict[scenario] = gaussian_results
        gaussian_optimal_dict[scenario] = gaussian_optimal
        # For heatmap: store final cumulative regret for each agent (use per-episode optimal)
        regret_summary['Bernoulli'].append([
            np.mean(np.cumsum(bernoulli_optimal[:, None] - bernoulli_results[agent], axis=1)[:, -1])
            for agent in bandit_scenario.bernoulli_agents.keys()
        ])
        regret_summary['Gaussian'].append([
            np.mean(np.cumsum(gaussian_optimal[:, None] - gaussian_results[agent], axis=1)[:, -1])
            for agent in bandit_scenario.gaussian_agents.keys()
        ])
    # Plot heatmaps
    agents = list(bandit_scenario.bernoulli_agents.keys())
    bandit_scenario.plot_heatmap(np.array(regret_summary['Bernoulli']), scenarios, agents, 'Bernoulli Bandit: Final Cumulative Regret', 'bernoulli_regret_heatmap.png')
    bandit_scenario.plot_heatmap(np.array(regret_summary['Gaussian']), scenarios, agents, 'Gaussian Bandit: Final Cumulative Regret', 'gaussian_regret_heatmap.png')
    # Plot easy/hard subplots for both bandit types
    bandit_scenario.plot_easy_hard_subplots(
        bernoulli_results_dict['easy'], bernoulli_optimal_dict['easy'],
        bernoulli_results_dict['hard'], bernoulli_optimal_dict['hard'],
        'Bernoulli')
    bandit_scenario.plot_easy_hard_subplots(
        gaussian_results_dict['easy'], gaussian_optimal_dict['easy'],
        gaussian_results_dict['hard'], gaussian_optimal_dict['hard'],
        'Gaussian')
    print("Plots saved: cumulative regret curves, subplots, and heatmaps.")

if __name__ == "__main__":
    main() 