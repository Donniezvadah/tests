import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from agents.llm_agent import LLMAgent
from agents.epsilon import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson import ThompsonSamplingAgent
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
        
        # Initialize agents
        self.agents = {
            'LLM': LLMAgent(),
            'Epsilon-Greedy': EpsilonGreedyAgent(epsilon=0.1),
            'UCB': UCBAgent(),
            'Thompson Sampling': ThompsonSamplingAgent()
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
        results = {agent_name: np.zeros((self.n_episodes, self.n_trials)) for agent_name in self.agents.keys()}
        optimal_per_episode = np.zeros(self.n_episodes)
        for episode in range(self.n_episodes):
            p1, p2 = self._sample_bernoulli_params(scenario)
            env = BernoulliBandit(n_actions=2, probs=[p1, p2])
            optimal_per_episode[episode] = max(p1, p2)
            for agent_name, agent in self.agents.items():
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
        results = {agent_name: np.zeros((self.n_episodes, self.n_trials)) for agent_name in self.agents.keys()}
        optimal_per_episode = np.zeros(self.n_episodes)
        for episode in range(self.n_episodes):
            (mean1, std1), (mean2, std2) = self._sample_gaussian_params(scenario)
            env = GaussianBandit(n_actions=2)
            env.set([mean1, mean2], [std1, std2])
            optimal_per_episode[episode] = max(mean1, mean2)
            for agent_name, agent in self.agents.items():
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
            std_regret = regret.std(axis=0)
            plt.plot(mean_regret, label=agent_name)
            plt.fill_between(np.arange(self.n_trials), mean_regret-std_regret, mean_regret+std_regret, alpha=0.15)
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

def main():
    scenarios = ['easy', 'medium', 'hard', 'uniform']
    bandit_scenario = BanditScenario(n_trials=100, n_episodes=20)
    regret_summary = {'Bernoulli': [], 'Gaussian': []}
    for scenario in scenarios:
        print(f"\nRunning {scenario} scenario...")
        # Bernoulli
        bernoulli_results, bernoulli_optimal = bandit_scenario.run_bernoulli_scenario(scenario)
        bandit_scenario.plot_cumulative_regret(bernoulli_results, bernoulli_optimal, scenario, 'Bernoulli')
        # Gaussian
        gaussian_results, gaussian_optimal = bandit_scenario.run_gaussian_scenario(scenario)
        bandit_scenario.plot_cumulative_regret(gaussian_results, gaussian_optimal, scenario, 'Gaussian')
        # For heatmap: store final cumulative regret for each agent (use per-episode optimal)
        regret_summary['Bernoulli'].append([
            np.mean(np.cumsum(bernoulli_optimal[:, None] - bernoulli_results[agent], axis=1)[:, -1])
            for agent in bandit_scenario.agents.keys()
        ])
        regret_summary['Gaussian'].append([
            np.mean(np.cumsum(gaussian_optimal[:, None] - gaussian_results[agent], axis=1)[:, -1])
            for agent in bandit_scenario.agents.keys()
        ])
    # Plot heatmaps
    agents = list(bandit_scenario.agents.keys())
    bandit_scenario.plot_heatmap(np.array(regret_summary['Bernoulli']), scenarios, agents, 'Bernoulli Bandit: Final Cumulative Regret', 'bernoulli_regret_heatmap.png')
    bandit_scenario.plot_heatmap(np.array(regret_summary['Gaussian']), scenarios, agents, 'Gaussian Bandit: Final Cumulative Regret', 'gaussian_regret_heatmap.png')
    print("Plots saved: cumulative regret curves and heatmaps.")

if __name__ == "__main__":
    main() 