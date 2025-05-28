"""
Evaluation module for comparing agent behaviors using Wasserstein metric.

This module implements various methods to quantify and compare the behavior of different agents
using the Wasserstein metric (also known as Earth Mover's Distance). The Wasserstein metric
provides a way to measure the distance between probability distributions, which is particularly
useful for comparing agent behaviors and exploration patterns.

References:
- Wasserstein metric: https://en.wikipedia.org/wiki/Wasserstein_metric
- Optimal Transport: https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import List, Dict, Tuple, Union, Optional
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
from datetime import datetime
import importlib
import inspect
import sys

# Import all agents from the agents folder
from agents import (
    EpsilonGreedyAgent,
    UCBAgent,
    ThompsonSamplingAgent,
    KLUCBAgent,
    GaussianEpsilonGreedyAgent,
    GaussianUCBAgent,
    GaussianThompsonSamplingAgent,
    # LLMAgent  # Uncomment for later use
)

# Import all environments from the environments folder
from environments import (
    BernoulliBandit,
    GaussianBandit
)
from environments.gaussian_bandit import generate_configuration

class AgentBehaviorEvaluator:
    """
    A class for evaluating and comparing agent behaviors using Wasserstein metrics.
    
    This class provides methods to:
    1. Calculate Wasserstein distances between agent state distributions
    2. Compare exploration patterns
    3. Analyze sampling distributions
    4. Visualize behavioral differences
    5. Save results to CSV and plots to directory
    """
    
    def __init__(self, dimension: int = 2, output_dir: str = "Wasserstein_plots"):
        """
        Initialize the evaluator.
        
        Args:
            dimension (int): The dimensionality of the state space being analyzed.
                           Default is 2 for 2D environments.
            output_dir (str): Directory to save plots and results
        """
        self.dimension = dimension
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.comparison_results = []
        self.agent_statistics = []
        self.agent_states = {}  # NEW: Store states for each agent/environment
        
    def compute_wasserstein_distance(self, 
                                   distribution1: np.ndarray, 
                                   distribution2: np.ndarray,
                                   p: int = 1) -> float:
        """
        Compute the p-Wasserstein distance between two distributions.
        
        Args:
            distribution1 (np.ndarray): First distribution (samples or histogram)
            distribution2 (np.ndarray): Second distribution (samples or histogram)
            p (int): Order of the Wasserstein distance (1 or 2)
            
        Returns:
            float: The p-Wasserstein distance between the distributions
            
        Note:
            For p=1, this is also known as the Earth Mover's Distance
            For p=2, this is the quadratic Wasserstein distance
        """
        if p == 1:
            return wasserstein_distance(distribution1, distribution2)
        elif p == 2:
            # For p=2, we need to compute the squared distances
            return np.sqrt(wasserstein_distance(distribution1, distribution2) ** 2)
        else:
            raise ValueError("Only p=1 or p=2 are supported")
            
    def compare_agent_states(self,
                           agent1_states: List[np.ndarray],
                           agent2_states: List[np.ndarray],
                           agent1_name: str = "Agent1",
                           agent2_name: str = "Agent2",
                           metric: str = 'wasserstein') -> Dict[str, float]:
        """
        Compare the state distributions of two agents.
        
        Args:
            agent1_states (List[np.ndarray]): List of states visited by agent 1
            agent2_states (List[np.ndarray]): List of states visited by agent 2
            agent1_name (str): Name of the first agent
            agent2_name (str): Name of the second agent
            metric (str): The metric to use ('wasserstein' or 'euclidean')
            
        Returns:
            Dict[str, float]: Dictionary containing various comparison metrics
        """
        # Convert state lists to numpy arrays
        states1 = np.array(agent1_states)
        states2 = np.array(agent2_states)
        
        results = {
            'agent1_name': agent1_name,
            'agent2_name': agent2_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if metric == 'wasserstein':
            # Compute Wasserstein distance for each dimension
            for dim in range(self.dimension):
                dist = self.compute_wasserstein_distance(
                    states1[:, dim], 
                    states2[:, dim]
                )
                results[f'wasserstein_distance_dim_{dim}'] = float(dist)
                
            # Compute overall Wasserstein distance
            results['wasserstein_distance_overall'] = float(np.mean([
                results[f'wasserstein_distance_dim_{dim}'] 
                for dim in range(self.dimension)
            ]))
            
        # Store results for later CSV export
        self.comparison_results.append(results)
            
        return results
    
    def analyze_exploration_pattern(self,
                                  agent_states: List[np.ndarray],
                                  agent_name: str,
                                  grid_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Analyze the exploration pattern of an agent by creating a heatmap.
        
        Args:
            agent_states (List[np.ndarray]): List of states visited by the agent
            agent_name (str): Name of the agent
            grid_size (int): Size of the grid for creating the heatmap
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing the exploration heatmap
                                 and other metrics
        """
        states = np.array(agent_states)
        
        # Create a 2D histogram of visited states
        heatmap, x_edges, y_edges = np.histogram2d(
            states[:, 0], 
            states[:, 1],
            bins=grid_size
        )
        
        # Save heatmap plot
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap.T, origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Visit Count')
        plt.title(f'Exploration Pattern - {agent_name}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(os.path.join(self.output_dir, f'heatmap_{agent_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'heatmap': heatmap,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'coverage': np.count_nonzero(heatmap) / (grid_size * grid_size)
        }
    
    def visualize_comparison(self,
                           agent1_states: List[np.ndarray],
                           agent2_states: List[np.ndarray],
                           agent1_name: str = "Agent1",
                           agent2_name: str = "Agent2",
                           title: str = "Agent Behavior Comparison"):
        """
        Create a visualization comparing the behaviors of two agents.
        
        Args:
            agent1_states (List[np.ndarray]): States visited by agent 1
            agent2_states (List[np.ndarray]): States visited by agent 2
            agent1_name (str): Name of the first agent
            agent2_name (str): Name of the second agent
            title (str): Title for the visualization
        """
        states1 = np.array(agent1_states)
        states2 = np.array(agent2_states)
        
        plt.figure(figsize=(12, 5))
        
        # Plot agent 1's states
        plt.subplot(121)
        plt.scatter(states1[:, 0], states1[:, 1], alpha=0.5, label=agent1_name)
        plt.title(f'{agent1_name} Exploration')
        plt.legend()
        
        # Plot agent 2's states
        plt.subplot(122)
        plt.scatter(states2[:, 0], states2[:, 1], alpha=0.5, label=agent2_name)
        plt.title(f'{agent2_name} Exploration')
        plt.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save the comparison plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.output_dir, f'comparison_{agent1_name}_vs_{agent2_name}_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def compute_sampling_statistics(self,
                                  agent_states: List[np.ndarray],
                                  agent_name: str) -> Dict[str, float]:
        """
        Compute various statistics about the agent's sampling behavior.
        
        Args:
            agent_states (List[np.ndarray]): List of states visited by the agent
            agent_name (str): Name of the agent
            
        Returns:
            Dict[str, float]: Dictionary containing various sampling statistics
        """
        states = np.array(agent_states)
        
        stats = {
            'agent_name': agent_name,
            'mean_position_x': float(np.mean(states[:, 0])),
            'mean_position_y': float(np.mean(states[:, 1])),
            'std_position_x': float(np.std(states[:, 0])),
            'std_position_y': float(np.std(states[:, 1])),
            'coverage_area': float(self._compute_coverage_area(states)),
            'exploration_efficiency': float(self._compute_exploration_efficiency(states))
        }
        
        # Store results for later CSV export
        self.agent_statistics.append(stats)
        
        return stats
    
    def save_results_to_csv(self, filename: str = "wasserstein_metrics.csv"):
        """
        Save all collected results to a CSV file.
        
        Args:
            filename (str): Name of the CSV file to save results
        """
        if not self.comparison_results and not self.agent_statistics:
            print("No results to save!")
            return
            
        # Create comparison results DataFrame
        comparison_df = pd.DataFrame(self.comparison_results)
        
        # Create agent statistics DataFrame
        stats_df = pd.DataFrame(self.agent_statistics)
        
        # Save both DataFrames to separate sheets in an Excel file
        excel_path = os.path.join(self.output_dir, filename.replace('.csv', '.xlsx'))
        with pd.ExcelWriter(excel_path) as writer:
            comparison_df.to_excel(writer, sheet_name='Agent_Comparisons', index=False)
            stats_df.to_excel(writer, sheet_name='Agent_Statistics', index=False)
            
        print(f"Results saved to {excel_path}")
    
    def _compute_coverage_area(self, states: np.ndarray) -> float:
        """
        Compute the area covered by the agent's exploration.
        
        Args:
            states (np.ndarray): Array of states visited by the agent
            
        Returns:
            float: The area of the convex hull of visited states
        """
        from scipy.spatial import ConvexHull
        if len(states) < 3:
            return 0.0
        hull = ConvexHull(states)
        return hull.volume
    
    def _compute_exploration_efficiency(self, states: np.ndarray) -> float:
        """
        Compute the efficiency of the agent's exploration.
        
        Args:
            states (np.ndarray): Array of states visited by the agent
            
        Returns:
            float: A measure of exploration efficiency (0 to 1)
        """
        if len(states) < 2:
            return 0.0
            
        # Compute the ratio of unique states to total states
        unique_states = np.unique(states, axis=0)
        return len(unique_states) / len(states)

# Example usage
if __name__ == "__main__":
    # Define the number of arms, trials, and episodes
    n_arms = 2  # Increased from 2 to 10
    n_trials = 5000  # Increased from 25 to 5000
    n_episodes = 50  # Increased from 5 to 50

    # Initialize evaluator
    evaluator = AgentBehaviorEvaluator(dimension=2)

    # List of agents to evaluate
    agents = [
        EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli'),
        UCBAgent(),
        ThompsonSamplingAgent(environment_type='bernoulli'),
        KLUCBAgent(n_arms=n_arms),
        GaussianEpsilonGreedyAgent(n_arms=n_arms),
        GaussianUCBAgent(n_arms=n_arms),
        GaussianThompsonSamplingAgent(n_arms=n_arms),
        # LLMAgent()  # Uncomment for later use
    ]

    # List of environments to evaluate
    environments = [
        BernoulliBandit(n_actions=n_arms),
        GaussianBandit(n_actions=n_arms)
    ]

    # Initialize GaussianBandit with default means and standard deviations
    for env in environments:
        if isinstance(env, GaussianBandit):
            means, stds = generate_configuration(n_arms)
            env.set(means, stds)

    # Run each agent in each environment
    for env in environments:
        env_name = type(env).__name__
        optimal_reward = env.optimal_reward()
        
        for agent in agents:
            # Initialize tracking variables
            cumulative_rewards = []
            action_counts = np.zeros(n_arms)
            regrets = []
            rewards_over_time = []
            
            agent_states = []
            for episode in range(n_episodes):
                env.reset()
                agent.init_actions(n_arms)
                episode_rewards = []
                
                for trial in range(n_trials):
                    action = agent.get_action()
                    reward = env.pull(action)
                    agent.update(action, reward)
                    
                    # Track metrics
                    action_counts[action] += 1
                    episode_rewards.append(reward)
                    regret = optimal_reward - reward
                    regrets.append(regret)
                    
                    agent_states.append([action, reward])
                
                rewards_over_time.append(np.mean(episode_rewards))
                cumulative_rewards.append(np.sum(episode_rewards))
            
            # Compute additional statistics
            stats = {
                'agent_name': agent.name,
                'environment': env_name,
                'mean_reward': np.mean(rewards_over_time),
                'std_reward': np.std(rewards_over_time),
                'cumulative_reward': np.sum(cumulative_rewards),
                'average_regret': np.mean(regrets),
                'action_distribution': action_counts / np.sum(action_counts),
                'convergence_rate': np.mean(rewards_over_time[-10:]) / optimal_reward,  # Last 10 episodes
                'exploration_ratio': np.sum(action_counts < np.mean(action_counts)) / n_arms
            }
            
            evaluator.agent_statistics.append(stats)
            
            # Store agent states for later comparison
            key = f"{agent.name}_{env_name}"
            evaluator.agent_states[key] = agent_states
            
            # Generate and save exploration pattern plot
            evaluator.analyze_exploration_pattern(agent_states, key)
            
            # Plot performance metrics
            plt.figure(figsize=(15, 10))
            
            # Plot rewards over time
            plt.subplot(221)
            plt.plot(rewards_over_time)
            plt.title(f'Average Reward Over Time - {key}')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            
            # Plot cumulative regret
            plt.subplot(222)
            plt.plot(np.cumsum(regrets))
            plt.title(f'Cumulative Regret - {key}')
            plt.xlabel('Trial')
            plt.ylabel('Cumulative Regret')
            
            # Plot action distribution
            plt.subplot(223)
            plt.bar(range(n_arms), action_counts / np.sum(action_counts))
            plt.title(f'Action Selection Distribution - {key}')
            plt.xlabel('Action')
            plt.ylabel('Selection Frequency')
            
            # Plot convergence rate
            plt.subplot(224)
            plt.plot(np.array(rewards_over_time) / optimal_reward)
            plt.axhline(y=1.0, color='r', linestyle='--')
            plt.title(f'Convergence Rate - {key}')
            plt.xlabel('Episode')
            plt.ylabel('Performance Ratio')
            
            plt.tight_layout()
            plt.savefig(os.path.join(evaluator.output_dir, f'performance_{key}.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Compare all pairs of agents in each environment
    env_names = [type(env).__name__ for env in environments]
    for env_name in env_names:
        agent_keys = [f"{agent.name}_{env_name}" for agent in agents]
        for i, key1 in enumerate(agent_keys):
            for j, key2 in enumerate(agent_keys[i+1:], i+1):
                agent1_states = evaluator.agent_states[key1]
                agent2_states = evaluator.agent_states[key2]
                evaluator.compare_agent_states(agent1_states, agent2_states, key1, key2)
                # Generate and save comparison plot
                evaluator.visualize_comparison(agent1_states, agent2_states, key1, key2, title=f"{env_name} Comparison")

    # Save all results to Excel
    evaluator.save_results_to_csv() 