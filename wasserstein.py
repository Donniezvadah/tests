"""
Wasserstein Distance Analysis for Multi-Armed Bandit Agents

This script implements a methodology to compare the action selection strategies
of different bandit agents (including an LLM-based agent) using the 1-Wasserstein
distance. The analysis is performed on both Bernoulli and Gaussian bandit environments.

Methodology:
As outlined in the user's request, the comparison treats each agent's action
selection history as an empirical probability distribution. The distance between
these distributions is calculated over tumbling windows to observe how behavioral
similarities evolve over time.

The core of the methodology relies on a "fixed unit distance" cost metric between
any two distinct arms. This choice is particularly suited for categorical actions
in bandit problems where no inherent ordinal relationship exists. With this cost
function, the 1-Wasserstein distance is equivalent to half of the Total Variation
Distance (TVD) between the distributions.

The script is structured as follows:
1.  Configuration: Set up simulation parameters, agent configurations, and
    environment details from YAML files.
2.  Simulation: Run episodes for each agent in each environment, logging every
    action taken.
3.  Wasserstein Calculation: For each non-LLM agent, compute the Wasserstein
    distance between its action distribution and the LLM's distribution over
    time using a tumbling window approach.
4.  Visualization & Reporting: Generate plots showing the evolution of the
    Wasserstein distance and save summary statistics to a CSV file.

Reference: The methodology is inspired by the paper provided by the user:
https://arxiv.org/abs/2504.03743v1
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import os
import random
import sys

# --- Matplotlib Styling for Publication-Quality Plots ---
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 16,
    'pdf.fonttype': 42,  # Embed fonts in PDF for compatibility
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
})

# --- Import Agents and Environments ---
# --- Import Agents and Environments ---
# Add the script's own directory to the path to ensure that sibling packages
# ('agents', 'environments') can be found, as this script is at the project root.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# The parent directory is not needed for imports, but we define it to maintain
# compatibility with the original logic before correcting the config path below.
parent_dir = os.path.dirname(current_dir)

try:
    from agents import (
        EpsilonGreedyAgent,
        UCBAgent,
        ThompsonSamplingAgent,
        GaussianEpsilonGreedyAgent,
        GaussianUCBAgent,
        GaussianThompsonSamplingAgent,
        LLMAgent
    )
    from environments import (
        BernoulliBandit,
        GaussianBandit
    )
except ImportError as e:
    print(f"Fatal Error: Could not import required agent or environment modules: {e}")
    print("Please ensure this script is run from a location where 'agents' and 'environments' packages are accessible.")
    sys.exit(1)


def load_env_config(config_path: str) -> Dict:
    """Loads environment configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_clean_label(agent_name: str) -> str:
    """Standardizes agent labels for clean, publication-ready plots."""
    if 'LLM' in agent_name:
        return 'LLM'
    if 'EpsilonGreedy' in agent_name:
        return r'$\epsilon$-greedy'
    if 'ThompsonSampling' in agent_name:
        return 'TS'  # Shortened for clarity
    if 'UCB' in agent_name:
        return 'UCB'
    return agent_name

class WassersteinEvaluator:
    """
    Compares bandit agent action selection distributions using Wasserstein distance.
    """
    def __init__(self, output_dir: str = "wasserstein_analysis"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        self.all_distances = {}  # {env_label: {comparison_label: [distances]}}

        # Define styles for plotting for consistent and clear visuals
        self.agent_styles = {
            'LLM': {'color': 'red', 'linestyle': ':', 'linewidth': 2.5},
            r'$\epsilon$-greedy': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2},
            'UCB': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 2},
            'TS': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2},  # Key updated to match get_clean_label
        }
        self.default_style = {'color': '#7f7f7f', 'linestyle': '--', 'linewidth': 1.5}

    def run_simulation(self, agents: Dict[str, Any], environment: Any, n_trials: int, n_episodes: int) -> pd.DataFrame:
        """Runs the bandit simulation and logs actions and rewards for all agents."""
        log_data = []
        n_arms = environment.action_count

        for agent in agents.values():
            agent.init_actions(n_arms)

        for episode in range(n_episodes):
            environment.reset()
            for agent in agents.values():
                agent.init_actions(n_arms)

            print(f"  --- Episode {episode + 1}/{n_episodes} ---")
            for agent_name, agent in agents.items():
                print(f"    Simulating agent: {agent_name}...")
                for trial in range(n_trials):
                    action = agent.get_action()
                    reward = environment.pull(action)
                    agent.update(action, reward)
                    log_data.append({
                        'episode': episode,
                        'trial': trial,
                        'agent_name': get_clean_label(agent_name),
                        'action': action,
                        'reward': reward
                    })
        return pd.DataFrame(log_data)

    def compute_wasserstein_distance(self, df_log: pd.DataFrame, llm_agent_name: str, other_agent_name: str, n_arms: int, n_trials: int, window_size: int) -> List[float]:
        """
        Computes the 1-Wasserstein distance over tumbling windows of trials,
        aggregating actions across all episodes to form the distributions.
        """
        llm_df = df_log[df_log['agent_name'] == llm_agent_name]
        other_df = df_log[df_log['agent_name'] == other_agent_name]

        distances = []

        if window_size <= 0:
            print("      Error: Window size must be positive.")
            return []
        if n_trials < window_size:
            print(f"      Warning: Not enough trials for windowing. Trials: {n_trials}, Window: {window_size}")
            return []

        # Iterate through windows of trials
        for i in range(0, n_trials - window_size + 1, window_size):
            start_trial = i
            end_trial = i + window_size

            # Filter actions for the current trial window across all episodes
            window_llm_actions = llm_df[(llm_df['trial'] >= start_trial) & (llm_df['trial'] < end_trial)]['action'].values
            window_other_actions = other_df[(other_df['trial'] >= start_trial) & (other_df['trial'] < end_trial)]['action'].values

            if len(window_llm_actions) == 0 or len(window_other_actions) == 0:
                print(f"      Warning: No action data in window {i//window_size + 1} for {llm_agent_name} vs {other_agent_name}. Skipping.")
                continue

            # Create empirical probability distributions (histograms)
            hist_llm, _ = np.histogram(window_llm_actions, bins=np.arange(n_arms + 1), density=True)
            hist_other, _ = np.histogram(window_other_actions, bins=np.arange(n_arms + 1), density=True)

            # For a fixed unit cost d(i, j) = 1 if i != j, the 1-Wasserstein distance
            # is equivalent to half the Total Variation Distance (TVD).
            # TVD = 0.5 * sum(|p_i - q_i|)
            dist = 0.5 * np.sum(np.abs(hist_llm - hist_other))
            distances.append(dist)

        return distances

    def analyze_environment(self, env_name: str, env_config_path: str, agents_setup: Dict, n_trials: int, n_episodes: int, window_size: int):
        """Runs the full analysis pipeline for a single environment configuration."""
        print(f"\n--- Analyzing Environment: {env_name} ---")
        config = load_env_config(env_config_path)
        env_class_name = f"{env_name}Bandit"
        env_class = getattr(sys.modules[__name__], env_class_name)
        config.pop('name', None)  # Remove descriptive name key if present

        # Adapt config keys to match environment constructor arguments
        if env_name == 'Bernoulli':
            config['probs'] = config.pop('probabilities')
            config['n_actions'] = len(config['probs'])
        elif env_name == 'Gaussian':
            config['n_actions'] = len(config['means'])

        env = env_class(**config)
        n_arms = env.action_count
        
        agents = agents_setup[env_name](n_arms)

        # Run simulation
        df_log = self.run_simulation(agents, env, n_trials, n_episodes)

        # Wasserstein Analysis
        llm_clean_name = get_clean_label('LLMAgent')
        self.all_distances[env_name] = {}

        agent_clean_names = [get_clean_label(name) for name in agents.keys()]

        for other_agent_clean_name in agent_clean_names:
            if other_agent_clean_name == llm_clean_name:
                continue

            print(f"  Computing Wasserstein: {llm_clean_name} vs {other_agent_clean_name}")
            distances = self.compute_wasserstein_distance(df_log, llm_clean_name, other_agent_clean_name, n_arms, n_trials, window_size)
            
            if distances:
                comparison_label = f"{llm_clean_name} vs {other_agent_clean_name}"
                self.all_distances[env_name][comparison_label] = distances
                self.results.append({
                    'Environment': env_name,
                    'Comparison': comparison_label,
                    'Avg_Wasserstein_Dist': np.mean(distances),
                    'Std_Wasserstein_Dist': np.std(distances),
                    'Window_Size': window_size
                })
        
        self.plot_combined_distances(env_name, window_size)

    def plot_combined_distances(self, env_name: str, window_size: int, smoothing_window: int = 5):
        """Plots combined Wasserstein distances for all comparisons in an environment."""
        if env_name not in self.all_distances or not self.all_distances[env_name]:
            print(f"No data to plot for {env_name}.")
            return

        plt.figure(figsize=(12, 7))
        
        for comparison_label, distances in self.all_distances[env_name].items():
            other_agent_name = comparison_label.split(' vs. ')[1]
            style = self.agent_styles.get(other_agent_name, self.default_style)
            
            # Apply a rolling average to smooth the plot
            if len(distances) >= smoothing_window:
                smooth_distances = pd.Series(distances).rolling(window=smoothing_window, min_periods=1).mean()
            else:
                smooth_distances = distances

            plt.plot(smooth_distances, label=comparison_label, **style)
        
        plt.xlabel('Window', fontweight='bold')
        plt.ylabel('Wasserstein Distance', fontweight='bold')
        plt.legend(title='Agent Comparisons')
        plt.grid(True)
        plt.tight_layout()
        
        plot_filename = f"wasserstein_combined_{env_name}.pdf"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"  Combined plot saved to {plot_path}")

    def save_summary_results(self):
        """Saves summary statistics to a CSV and generates a final summary plot."""
        if not self.results:
            print("No results to save.")
            return

        df_results = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, "wasserstein_summary.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\nSummary results saved to {csv_path}")

        # Summary Bar Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_results, x='comparison', y='mean_dist', hue='environment', palette='viridis')
        plt.xlabel('Agent Comparison')
        plt.ylabel('Average Wasserstein Distance')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "wasserstein_summary_barplot.pdf")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Summary plot saved to {plot_path}")


if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # --- Main Configuration ---
    N_TRIALS = 50
    N_EPISODES = 25
    WINDOW_SIZE = 5
    OUTPUT_DIR = "wasserstein_analysis_output"

    # --- Agent Setup ---
    # Use lambdas to defer agent instantiation until n_arms is known from the env config
    agents_setup = {
        'Bernoulli': lambda n_arms: {
            'LLMAgent': LLMAgent(model="gpt-4.1-nano"),
            'EpsilonGreedyAgent': EpsilonGreedyAgent(epsilon=0.1),
            'UCBAgent': UCBAgent(),
            'ThompsonSamplingAgent': ThompsonSamplingAgent()
        },
        'Gaussian': lambda n_arms: {
            'LLMAgent': LLMAgent(model="gpt-4.1-nano"),
            'GaussianEpsilonGreedyAgent': GaussianEpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1),
            'GaussianUCBAgent': GaussianUCBAgent(n_arms=n_arms, c=2),
            'GaussianThompsonSamplingAgent': GaussianThompsonSamplingAgent(n_arms=n_arms)
        }
    }

    # --- Environment Configurations ---
    # Assumes config files are in a 'configurations/environment' directory inside the project root.
    config_dir = os.path.join(current_dir, 'configurations', 'environment')
    env_configs = {
        'Bernoulli': os.path.join(config_dir, 'bernoulli_env.yaml'),
        'Gaussian': os.path.join(config_dir, 'gaussian_env.yaml')
    }
    
    # Create dummy config files if they don't exist to ensure script can run
    for env_type, path in env_configs.items():
        if not os.path.exists(path):
            print(f"Configuration file not found: {path}. Creating a default.")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if env_type == 'Bernoulli':
                dummy_config = {'name': 'Default Bernoulli', 'probabilities': [0.1, 0.3, 0.5, 0.7, 0.9]}
            else:  # Gaussian
                dummy_config = {'name': 'Default Gaussian', 'means': [0.0, 0.5, 1.0, 1.5, 2.0], 'stds': [1, 1, 1, 1, 1]}
            with open(path, 'w') as f:
                yaml.dump(dummy_config, f)
            print(f"Created a default configuration file at: {path}")

    # --- Run Evaluation ---
    evaluator = WassersteinEvaluator(output_dir=OUTPUT_DIR)

    for env_name, config_path in env_configs.items():
        evaluator.analyze_environment(
            env_name=env_name,
            env_config_path=config_path,
            agents_setup=agents_setup,
            n_trials=N_TRIALS,
            n_episodes=N_EPISODES,
            window_size=WINDOW_SIZE
        )

    evaluator.save_summary_results()
    print("\n--- Evaluation Complete ---")