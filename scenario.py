import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from agents.llm_agent import LLMAgent
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
            'LLM': LLMAgent(model="gpt-4.1-nano"),
        }
        
        # Initialize agents for Gaussian bandit
        self.gaussian_agents = {
            'Epsilon-Greedy': GaussianEpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1),
            'UCB': GaussianUCBAgent(n_arms=n_arms),
            'Thompson Sampling': GaussianThompsonSamplingAgent(n_arms=n_arms),
            'LLM': LLMAgent(model="gpt-4.1-nano"),
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
            mean1 = np.random.choice([-0.2, 0.2]) #Hard = Close to 0
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
        
    def calculate_cumulative_regret(self, rewards, optimal_rewards):
        """Calculate cumulative regret ensuring it's non-decreasing.
        
        Args:
            rewards: Array of shape (n_episodes, n_trials) with rewards for each trial
            optimal_rewards: Array of shape (n_episodes,) with optimal reward for each episode
            
        Returns:
            Array of shape (n_episodes, n_trials) with cumulative regret
        """
        n_episodes, n_trials = rewards.shape
        cumulative_regret = np.zeros_like(rewards)
        
        for ep in range(n_episodes):
            optimal = optimal_rewards[ep]
            regret = np.maximum(0, optimal - rewards[ep])  # Ensure non-negative regret
            cumulative_regret[ep] = np.cumsum(regret)
            
            # Ensure regret is non-decreasing (handle any numerical issues)
            for t in range(1, n_trials):
                if cumulative_regret[ep, t] < cumulative_regret[ep, t-1]:
                    cumulative_regret[ep, t] = cumulative_regret[ep, t-1]
                    
        return cumulative_regret

    def plot_cumulative_regret(self, results, optimal_per_episode, scenario, env_type):
        sns.set_theme(style="whitegrid")
        color_palette = sns.color_palette("husl", len(results))
        plt.figure(figsize=(8, 6))
        for idx, (agent_name, rewards) in enumerate(results.items()):
            cumulative_regret = self.calculate_cumulative_regret(rewards, optimal_per_episode)
            mean_regret = cumulative_regret.mean(axis=0)
            plt.plot(mean_regret, label=agent_name, linestyle='-', linewidth=2, color=color_palette[idx])
        plt.title(f"Cumulative Regret\n{env_type} Bandit — {scenario.capitalize()} Scenario", fontsize=18, fontweight='bold', pad=18)
        plt.xlabel('Trial Number', fontsize=15, fontweight='bold', labelpad=10)
        plt.ylabel('Cumulative Regret', fontsize=15, fontweight='bold', labelpad=10)
        plt.legend(title="Agent", fontsize=12, title_fontsize=13, loc='upper left', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylim(bottom=0)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'cumulative_regret_{env_type}_{scenario}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap(self, regret_matrix: np.ndarray, row_labels: list, col_labels: list, title: str, filename: str):
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            regret_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=col_labels,
            yticklabels=row_labels,
            cbar_kws={'label': 'Final Cumulative Regret'},
            linewidths=0.5,
            linecolor='white',
            annot_kws={"fontsize":13, "fontweight":'bold'}
        )
        plt.title(title, fontsize=18, fontweight='bold', pad=18)
        plt.xlabel('Agent', fontsize=15, fontweight='bold', labelpad=10)
        plt.ylabel('Scenario', fontsize=15, fontweight='bold', labelpad=10)
        plt.xticks(fontsize=13, fontweight='bold', rotation=20)
        plt.yticks(fontsize=13, fontweight='bold', rotation=0)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_easy_hard_subplots(self, easy_results, easy_optimal, hard_results, hard_optimal, env_type):
        sns.set_theme(style="whitegrid")
        color_palette = sns.color_palette("husl", len(easy_results))
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, facecolor='white')
        scenarios = ['easy', 'hard']
        results_list = [easy_results, hard_results]
        optimal_list = [easy_optimal, hard_optimal]
        for i, (results, optimal, scenario) in enumerate(zip(results_list, optimal_list, scenarios)):
            ax = axes[i]
            for idx, (agent_name, rewards) in enumerate(results.items()):
                cumulative_regret = self.calculate_cumulative_regret(rewards, optimal)
                mean_regret = cumulative_regret.mean(axis=0)
                ax.plot(mean_regret, label=agent_name, linestyle='-', linewidth=2, color=color_palette[idx])
            ax.set_title(f"{env_type} Bandit\n{scenario.capitalize()} Scenario", fontsize=16, fontweight='bold', pad=14)
            ax.set_xlabel('Trial Number', fontsize=14, fontweight='bold', labelpad=8)
            if i == 0:
                ax.set_ylabel('Cumulative Regret', fontsize=14, fontweight='bold', labelpad=8)
            else:
                ax.set_ylabel('')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis='both', which='major', labelsize=12)
            sns.despine(ax=ax)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Agent", fontsize=13, title_fontsize=14, loc='upper center', ncol=len(easy_results), frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1.03))
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(self.plot_dir, f'{env_type.lower()}_easy_hard_subplot.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    scenarios = ['easy', 'medium', 'hard', 'uniform']
    bandit_scenario = BanditScenario(n_trials=25, n_episodes=10)
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
        # For heatmap: store final cumulative regret for each agent
        regret_summary['Bernoulli'].append([
            bandit_scenario.calculate_cumulative_regret(bernoulli_results[agent], bernoulli_optimal)[:, -1].mean()
            for agent in bandit_scenario.bernoulli_agents.keys()
        ])
        regret_summary['Gaussian'].append([
            bandit_scenario.calculate_cumulative_regret(gaussian_results[agent], gaussian_optimal)[:, -1].mean()
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