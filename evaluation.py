import numpy as np
from scipy.stats import wasserstein_distance, norm
from typing import List, Dict, Tuple, Union, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns # For potentially enhanced summary plots
import pandas as pd
import openpyxl # For saving to Excel
from scipy.optimize import linear_sum_assignment
import yaml
import datetime
import importlib
import inspect
import sys
import os
import random

# Set global matplotlib style for all plots (publication-grade, serif, colorblind-friendly)
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 15,
    'figure.titlesize': 16,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.edgecolor': 'gray',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'grid.color': 'gray',
    'grid.linestyle': ':',
    'grid.alpha': 0.3,
})

# Import all agents from the agents folder
from agents import (
    EpsilonGreedyAgent,
    UCBAgent,
    ThompsonSamplingAgent,
    KLUCBAgent,
    GaussianEpsilonGreedyAgent,
    GaussianUCBAgent,
    GaussianThompsonSamplingAgent,
    LLMAgent
)

# Import all environments from the environments folder
from environments import (
    BernoulliBandit,
    GaussianBandit
)
from environments.gaussian_bandit import generate_configuration


def load_env_config(config_path: str) -> Dict:
    """Loads environment configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _get_clean_label(agent_name):
    """
    Standardize agent labels for publication-ready plots.
    """
    # Order is important. Check for LLM first.
    if agent_name.startswith('LLM'):
        return 'LLM'
    # Then check for more complex names containing keywords.
    # These should catch variants like "EpsilonGreedy(...)" or "Epsilon-Greedy"
    if 'EpsilonGreedy' in agent_name or 'Epsilon-Greedy' in agent_name:
        return r'$\epsilon$-greedy'
    if 'ThompsonSampling' in agent_name or 'Thompson Sampling' in agent_name:
        return 'TS'
    # Simple direct match for UCB, assuming its name is consistently 'UCB'.
    if 'UCB' in agent_name:
        return 'UCB'
    # Fallback if no specific rule applies
    return agent_name

class AgentBehaviorEvaluator:
    """
    A class for evaluating and comparing agent behaviors using Wasserstein metrics.
    This is a minimal stub for test run.
    """
    def __init__(self, output_dir: str = "Wasserstein_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.agent_states = {} # This seems unused, consider removing if not needed elsewhere
        self.window_wass_results = []
        self.all_env_wasserstein_data = {} # To store {env_label: {comparison_label: {'distances': [], 'window_size': X, 'other_agent_raw_name': 'Y'}}}

        # Define colors and line styles for different agent types
        self.agent_styles = {
            'EpsilonGreedy': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2},
            'UCB': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 2},
            'ThompsonSampling': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2},
            'KL-UCB': {'color': '#d62728', 'linestyle': '-', 'linewidth': 2},
            'LLM': {'color': 'red', 'linestyle': ':', 'linewidth': 2.5},
            'LLMV2': {'color': 'red', 'linestyle': ':', 'linewidth': 2.5},
            'GaussianEpsilonGreedy': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2},
            'GaussianUCB': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 2},
            'GaussianThompsonSampling': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2},
        }
        self.default_style = {'color': '#7f7f7f', 'linestyle': '-', 'linewidth': 1.5} # Grey for unknown

    def compute_tumbling_window_wasserstein(self, llm_actions_rewards, other_actions_rewards, 
                                           llm_name_key, other_name_key, # These are the actual keys from dict
                                           n_arms, window_size, env_label):
        # llm_actions_rewards and other_actions_rewards are lists of [action, reward]
        llm_actions = np.array([item[0] for item in llm_actions_rewards])
        other_actions = np.array([item[0] for item in other_actions_rewards])

        min_len = min(len(llm_actions), len(other_actions))
        wasserstein_distances = []

        # Ensure window_size is not larger than the number of trials
        # and at least 1 to avoid issues with range()
        actual_window_size = min(window_size, min_len)
        if actual_window_size < 1 and min_len > 0: # if min_len is 0, actual_window_size will be 0
             actual_window_size = 1
        elif min_len == 0: # No actions to compare
            print(f"      Not enough data to compute Wasserstein for {_get_clean_label(llm_name_key)} vs {_get_clean_label(other_name_key)} in {env_label} (min_len: {min_len}, window_size: {window_size}).")
            return [], actual_window_size


        for i in range(0, min_len - actual_window_size + 1, actual_window_size): # Iterate with a step of actual_window_size
            window_llm = llm_actions[i : i + actual_window_size]
            window_other = other_actions[i : i + actual_window_size]
            
            # Create empirical distributions (histograms)
            hist_llm, _ = np.histogram(window_llm, bins=np.arange(n_arms + 1) - 0.5, density=True)
            hist_other, _ = np.histogram(window_other, bins=np.arange(n_arms + 1) - 0.5, density=True)
            
            # Correctly call wasserstein_distance for 1D distributions from histograms
            dist = wasserstein_distance(np.arange(n_arms), np.arange(n_arms), u_weights=hist_llm, v_weights=hist_other)
            wasserstein_distances.append(dist)
        
        if not wasserstein_distances: # Avoid plotting if no data
            print(f"      No Wasserstein distances computed for {_get_clean_label(llm_name_key)} vs {_get_clean_label(other_name_key)} in {env_label} (min_len: {min_len}, window_size: {actual_window_size}).")
            return [], actual_window_size

        # Store results
        self.window_wass_results.append({
            'Environment': env_label,
            'LLM_Agent': _get_clean_label(llm_name_key), # Use _get_clean_label for display
            'Other_Agent': _get_clean_label(other_name_key), # Use _get_clean_label for display
            'Window_Size': actual_window_size,
            'Avg_Wasserstein_Dist': np.mean(wasserstein_distances),
            'Std_Wasserstein_Dist': np.std(wasserstein_distances)
        })

        # Store for combined plot
        comparison_label = f"{_get_clean_label(llm_name_key)} vs {_get_clean_label(other_name_key)}"
        if env_label not in self.all_env_wasserstein_data:
            self.all_env_wasserstein_data[env_label] = {}
        self.all_env_wasserstein_data[env_label][comparison_label] = {
            'distances': wasserstein_distances,
            'window_size': actual_window_size,
            'other_agent_raw_name': other_name_key # Store the raw name of the other agent for styling
        }

        return wasserstein_distances, actual_window_size # Return distances for individual plotting if needed

    def _plot_and_save_individual_wasserstein(self, wasserstein_distances, llm_name_key, other_name_key, 
                                                env_label, actual_window_size, custom_plot_filename):
        if not wasserstein_distances:
            # print(f"      Skipping individual plot for {_get_clean_label(llm_name_key)} vs {_get_clean_label(other_name_key)} in {env_label} due to no data.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(wasserstein_distances, label=f"Window size: {actual_window_size}")
        # title = f'Sliding Window Wasserstein: {_get_clean_label(llm_name_key)} vs {_get_clean_label(other_name_key)} ({env_label})' # Title removed
        # plt.title(title, fontsize=14) 
        plt.xlabel('Window Number', fontsize=12)
        plt.ylabel('Wasserstein Distance', fontsize=12)
        plt.legend(prop={'weight':'normal'})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, custom_plot_filename)
        plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"      Individual plot saved to {plot_path}")

    def _get_style_for_agent(self, raw_agent_name: str) -> dict:
        """Gets the plot style for a given raw agent name (e.g., agent.name)."""
        # Normalize the raw agent name for keyword checking by removing hyphens and spaces.
        # Keywords ('EpsilonGreedy', 'UCB', etc.) are case-sensitive to match self.agent_styles keys.
        normalized_name = raw_agent_name.replace('-', '').replace(' ', '')
        
        key = None  # Initialize key to None

        # Handle LLM agents first due to their distinct naming
        if 'LLM' in raw_agent_name: # Check original raw_agent_name for 'LLM' substring
            if 'V2' in raw_agent_name.upper() and 'LLMV2' in self.agent_styles: # Check for V2 variant
                key = 'LLMV2'
            elif 'LLM' in self.agent_styles: # Fallback to base LLM style
                key = 'LLM'
        else:
            # Check for Gaussian variants
            # Use lowercased original name for 'gaussian' keyword to be flexible
            is_gaussian = 'gaussian' in raw_agent_name.lower() 

            if is_gaussian:
                if 'EpsilonGreedy' in normalized_name:
                    key = 'GaussianEpsilonGreedy'
                elif 'ThompsonSampling' in normalized_name:
                    key = 'GaussianThompsonSampling'
                # Note: KLUCBAgent Gaussian variant is not explicitly in styles;
                # 'UCB' check below would catch GaussianUCBAgent.
                elif 'UCB' in normalized_name: 
                    key = 'GaussianUCB'
            else: # Non-Gaussian (base) agents
                if 'EpsilonGreedy' in normalized_name:
                    key = 'EpsilonGreedy'
                elif 'ThompsonSampling' in normalized_name:
                    key = 'ThompsonSampling'
                elif 'KLUCB' in normalized_name: # Check for KL-UCB before general UCB
                    key = 'KL-UCB'
                elif 'UCB' in normalized_name:
                    key = 'UCB'

        # Return the specific style if found, otherwise the default
        if key and key in self.agent_styles:
            return self.agent_styles[key]
        
        # Uncomment for debugging if an agent isn't getting the expected style:
        # print(f"Warning: Using default style for agent: '{raw_agent_name}' (Normalized: '{normalized_name}', Derived Key: '{key}')")
        return self.default_style

    def plot_combined_wasserstein_for_env(self, env_label: str):
        if env_label not in self.all_env_wasserstein_data or not self.all_env_wasserstein_data[env_label]:
            print(f"No data available to generate combined plot for {env_label}.")
            return

        plt.figure(figsize=(12, 7))
        
        actual_window_size = 0 # To store the window size, assuming it's consistent for an env

        for comparison_label, data in self.all_env_wasserstein_data[env_label].items():
            distances = data['distances']
            current_ws = data['window_size']
            if actual_window_size == 0: actual_window_size = current_ws # set first time
            elif actual_window_size != current_ws : 
                print(f"Warning: Mismatch in window sizes for combined plot in {env_label}. Using {actual_window_size}")
            
            if distances:
                other_agent_raw_name = data['other_agent_raw_name']
                style = self._get_style_for_agent(other_agent_raw_name)
                plt.plot(distances, label=comparison_label, 
                         color=style['color'], 
                         linestyle=style['linestyle'], 
                         linewidth=style['linewidth'])
        
        plt.xlabel('Window Number', fontsize=14)
        plt.ylabel('Wasserstein Distance', fontsize=14)
        plt.legend(title='Agent Comparisons', fontsize='11', title_fontproperties={'weight':'normal', 'size':'13'}, prop={'weight':'normal'})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_filename = f"combined_wasserstein_dist_{env_label}_ws{actual_window_size}.pdf"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Combined plot for {env_label} saved to {plot_path}")

    def plot_wasserstein_heatmap_for_env(self, env_label: str):
        """Generates and saves a heatmap of Wasserstein distances for a given environment."""
        if env_label not in self.all_env_wasserstein_data or not self.all_env_wasserstein_data[env_label]:
            print(f"No data available to generate heatmap for {env_label}.")
            return

        heatmap_data = {}
        max_len = 0
        for comparison_label, data_dict in self.all_env_wasserstein_data[env_label].items():
            distances = data_dict['distances']
            heatmap_data[comparison_label] = distances
            if len(distances) > max_len:
                max_len = len(distances)
        
        # Pad shorter lists with NaN to ensure DataFrame compatibility
        for comparison_label in heatmap_data:
            if len(heatmap_data[comparison_label]) < max_len:
                padding = [float('nan')] * (max_len - len(heatmap_data[comparison_label]))
                heatmap_data[comparison_label].extend(padding)

        df = pd.DataFrame(heatmap_data).T # Transpose to have comparison labels as rows

        if df.empty:
            print(f"DataFrame for heatmap is empty for {env_label}.")
            return

        plt.figure(figsize=(12, max(6, len(df.index) * 0.5))) # Adjust height based on number of comparisons
        sns.heatmap(df, annot=False, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Wasserstein Distance'})
        # Using annot=False as it can get very cluttered. For detailed values, refer to Excel.
        
        plt.title(f'Wasserstein Distance Heatmap: {env_label} (Window Size: {self.all_env_wasserstein_data[env_label].get(next(iter(self.all_env_wasserstein_data[env_label])), {}).get("window_size", "N/A")})',
                  fontsize=16, fontdict={'weight': 'normal'})
        plt.xlabel('Window Number', fontsize=14, fontdict={'weight': 'normal'})
        plt.ylabel('Comparison Pair', fontsize=14, fontdict={'weight': 'normal'})
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12, rotation=0)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for title

        plot_filename = f"heatmap_wasserstein_dist_{env_label}_ws{self.all_env_wasserstein_data[env_label].get(next(iter(self.all_env_wasserstein_data[env_label])), {}).get('window_size', 'N_A')}.pdf"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  Heatmap for {env_label} saved to {plot_path}")

    def plot_summary_wasserstein_metrics(self):
        if not self.window_wass_results:
            print("No Wasserstein results to generate summary plot.")
            return

        df = pd.DataFrame(self.window_wass_results)
        if df.empty:
            print("Wasserstein results DataFrame is empty. Skipping summary plot.")
            return

        plt.figure(figsize=(14, 8))
        # Ensure 'Other_Agent' and 'Environment' are suitable for hue and x-axis
        # We want to plot 'Avg_Wasserstein_Dist'
        # Example: Grouped bar plot: LLM vs Agent X, LLM vs Agent Y on X-axis, grouped by Environment
        
        # Create a combined 'Comparison' column for better legend/grouping if needed
        df['Comparison'] = df['LLM_Agent'] + ' vs ' + df['Other_Agent']

        # Grouped bar plot
        # X-axis: Comparison Type (LLM vs EpsilonGreedy, LLM vs UCB, etc.)
        # Bars grouped by: Environment (Bernoulli, Gaussian)
        # Y-axis: Average Wasserstein Distance
        sns.barplot(data=df, x='Comparison', y='Avg_Wasserstein_Dist', hue='Environment', palette='viridis')
        
        # plt.title('Summary: Average Wasserstein Distances', fontsize=16) # Title removed
        plt.xlabel('Agent Comparison', fontsize=14)
        plt.ylabel('Average Wasserstein Distance', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(title='Environment Type', fontsize='11', title_fontproperties={'weight':'normal', 'size':'13'}, prop={'weight':'normal'})
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        plot_filename = "summary_avg_wasserstein_distances.pdf"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Summary metrics plot saved to {plot_path}")

    def plot_summary_wasserstein_violin(self):
        """Generates and saves a violin plot of Wasserstein distance distributions."""
        if not self.all_env_wasserstein_data:
            print("No Wasserstein data available to generate violin plot.")
            return

        plot_data = []
        for env_label, comparisons in self.all_env_wasserstein_data.items():
            for comparison_label, data in comparisons.items():
                for distance in data['distances']:
                    plot_data.append({
                        'Environment': env_label,
                        'Comparison': comparison_label,
                        'Wasserstein Distance': distance
                    })

        if not plot_data:
            print("No Wasserstein distances recorded. Skipping violin plot.")
            return

        df = pd.DataFrame(plot_data)

        plt.figure(figsize=(14, 8))
        sns.violinplot(data=df, x='Comparison', y='Wasserstein Distance', hue='Environment',
                       split=True, inner='quart', palette='viridis', cut=0)

        plt.xlabel('Agent Comparison', fontsize=14)
        plt.ylabel('Wasserstein Distance', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(title='Environment Type', fontsize='11', title_fontproperties={'weight':'normal', 'size':'13'}, prop={'weight':'normal'})
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()

        plot_filename = "summary_violin_wasserstein_distances.pdf"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Summary violin plot saved to {plot_path}")

    def save_results_to_csv(self, filename: str = "wasserstein_metrics_test.csv"):
        window_wass_df = pd.DataFrame(self.window_wass_results)
        # excel_path = os.path.join(self.output_dir, filename.replace('.csv', '.xlsx'))
        # For consistency with user's original naming, let's stick to .xlsx if that was intended, or .csv if specified.
        # The parameter is filename="wasserstein_metrics_test.csv", so let's make it an Excel file as per original code's behavior.
        excel_filename = filename
        if not excel_filename.endswith('.xlsx'):
            excel_filename = filename.split('.')[0] + '.xlsx'

        excel_path = os.path.join(self.output_dir, excel_filename)
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                if not window_wass_df.empty:
                    window_wass_df.to_excel(writer, sheet_name='SlidingWindow_Wass', index=False)
                else:
                    # Create an empty sheet if df is empty to avoid ExcelWriter error with no sheets
                    pd.DataFrame().to_excel(writer, sheet_name='SlidingWindow_Wass', index=False)
            print(f"Results saved to {excel_path}")
        except Exception as e:
            print(f"Error saving results to Excel: {e}. Ensure 'openpyxl' is installed.")
            # Fallback to CSV if Excel fails
            csv_path = os.path.join(self.output_dir, filename if filename.endswith('.csv') else filename.split('.')[0] + '.csv')
            if not window_wass_df.empty:
                window_wass_df.to_csv(csv_path, index=False)
                print(f"Results saved as CSV to {csv_path} due to Excel saving error.")
            else:
                print(f"No data to save to CSV.")

    def save_detailed_log_to_csv(self, detailed_log: List[Dict[str, Any]], filename: str):
        """Saves the detailed simulation log to a CSV file."""
        if not detailed_log:
            print("Warning: Detailed log is empty, skipping CSV save.")
            return
        df = pd.DataFrame(detailed_log)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Detailed log saved to {filepath}")

    def plot_true_distributions(self, environment: 'BaseBanditEnv', env_label: str):
        """Plots the true underlying reward distributions for the environment's arms."""
        fig, ax = plt.subplots(figsize=(10, 6))
        n_arms = environment.action_count

        if isinstance(environment, BernoulliBandit):
            probs = environment._probs
            ax.bar(np.arange(n_arms), probs, color=sns.color_palette('viridis', n_arms))
            ax.set_title(f'True Reward Probabilities for {env_label} Environment', fontsize=16)
            ax.set_xlabel('Action (Arm)', fontsize=12)
            ax.set_ylabel('Probability of Reward (p)', fontsize=12)
            ax.set_xticks(np.arange(n_arms))
            ax.set_ylim(0, 1)

        elif isinstance(environment, GaussianBandit):
            means = environment.means
            stds = environment.stds
            x = np.linspace(np.min(means) - 3 * np.max(stds), np.max(means) + 3 * np.max(stds), 1000)
            for i in range(n_arms):
                pdf = norm.pdf(x, means[i], stds[i])
                ax.plot(x, pdf, label=f'Arm {i} (μ={means[i]:.2f}, σ={stds[i]:.2f})')
            ax.set_title(f'True Reward Distributions for {env_label} Environment', fontsize=16)
            ax.set_xlabel('Reward Value', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.legend()

        else:
            print(f"Distribution plotting not implemented for environment type: {type(environment).__name__}")
            plt.close(fig)
            return

        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filename = f"true_distribution_{env_label}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        print(f"True distribution plot saved to {filepath}")

    def plot_action_distribution(self, action_histories: Dict[str, np.ndarray], env_name: str):
        """Plots the distribution of actions taken by each agent."""
        if not action_histories:
            print("Warning: Action histories are empty, skipping distribution plot.")
            return
        
        agent_names = list(action_histories.keys())
        n_agents = len(agent_names)
        if n_agents == 0:
            return

        # Determine the number of arms from the data
        all_actions = np.concatenate([h.flatten() for h in action_histories.values()])
        n_arms = np.max(all_actions) + 1

        fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5), sharey=True)
        if n_agents == 1:
            axes = [axes] # Make it iterable
        fig.suptitle(f'Action Distribution per Agent in {env_name} Environment', fontsize=20, y=1.02)

        for i, agent_name in enumerate(agent_names):
            ax = axes[i]
            history = action_histories[agent_name]
            actions = history.flatten()
            counts = np.bincount(actions, minlength=n_arms)
            
            sns.barplot(x=np.arange(n_arms), y=counts, ax=ax, palette='viridis', hue=np.arange(n_arms), legend=False)
            ax.set_title(agent_name, fontsize=16)
            ax.set_xlabel('Action (Arm)', fontsize=12)
            if i == 0:
                ax.set_ylabel('Frequency', fontsize=12)
            ax.set_xticks(np.arange(n_arms))
            ax.tick_params(axis='x', rotation=0)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"action_distribution_{env_name}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        print(f"Action distribution plot saved to {filepath}")

    def run_bandit_simulation(self, agents: Dict[str, 'BaseAgent'], environment: 'BaseBanditEnv', n_trials: int, n_episodes: int) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        """Runs a bandit simulation for the given agents and environment."""
        action_histories: Dict[str, np.ndarray] = {name: np.zeros((n_episodes, n_trials), dtype=int) for name in agents}
        detailed_log = []
        n_actions = environment.action_count

        # Initialize all agents with the number of actions from the environment
        for agent in agents.values():
            agent.init_actions(n_actions)

        for episode in range(n_episodes):
            # Reset agents and environment for each episode
            for agent in agents.values():
                agent.init_actions(environment.action_count)
            environment.reset()
            
            print(f"\n  --- Episode {episode + 1}/{n_episodes} ---")
            for agent_name, agent in agents.items():
                print(f"  Simulating agent: {agent_name}...")
                for trial in range(n_trials):
                    action = agent.get_action()
                    reward = environment.pull(action)
                    agent.update(action, reward)
                    action_histories[agent_name][episode, trial] = action

                    # Log detailed policy information
                    policy_info = None
                    if hasattr(agent, 'last_llm_response') and agent.last_llm_response is not None:
                        policy_info = agent.last_llm_response
                    elif hasattr(agent, 'q_values'):
                        policy_info = str(agent.q_values.tolist())
                    elif hasattr(agent, 'alpha') and hasattr(agent, 'beta'): # For Thompson Sampling
                        policy_info = f"alpha={str(agent.alpha.tolist())}, beta={str(agent.beta.tolist())}"

                    detailed_log.append({
                        'episode': episode,
                        'trial': trial,
                        'agent_name': agent_name,
                        'action': action,
                        'reward': reward,
                        'policy_info': policy_info
                    })
        
        return action_histories, detailed_log


if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # --- Evaluation Configuration ---
    n_trials = 20 # Reduced for faster testing
    n_episodes = 1   # For action distribution comparison, 1 episode suffices
    window_size = 10  # Tumbling window size for Wasserstein distance

    # --- Initialize Evaluator ---
    output_dir = "Wasserstein_plots_5arms_yaml"
    evaluator = AgentBehaviorEvaluator(output_dir=output_dir)

    # --- Environment and Agent Setup ---
    env_configs = {
        'Bernoulli': 'configurations/environment/bernoulli_env.yaml',
        'Gaussian': 'configurations/environment/gaussian_env.yaml'
    }

    for env_name, config_path in env_configs.items():
        config = load_env_config(config_path)
        env_class_name = f"{env_name}Bandit"
        env_class = getattr(sys.modules[__name__], env_class_name)

        # The 'name' key from YAML is for description, not a constructor argument.
        config.pop('name', None)

        # Adapt config keys to match environment constructor arguments
        if env_name == 'Bernoulli':
            if 'probabilities' in config:
                config['n_actions'] = len(config['probabilities'])
                config['probs'] = config.pop('probabilities')
            else:
                print(f"Warning: 'probabilities' key not found for {env_name}. Skipping.")
                continue
        elif env_name == 'Gaussian':
            if 'means' in config:
                config['n_actions'] = len(config['means'])
            else:
                print(f"Warning: 'means' key not found for {env_name}. Skipping.")
                continue
        
        env = env_class(**config)
        n_arms = env.action_count
        env_label = f"{env_class_name.replace('Bandit', '')}"
        print(f"\n--- Running Evaluation for {env_name} Environment ({env.action_count} arms) ---")

        # Plot true reward distributions
        evaluator.plot_true_distributions(env, env_label)

        # Agent setup
        if env_name == 'Bernoulli':
            agents = {
                'LLMAgent': LLMAgent(model="gpt-4.1-nano"),
                'EpsilonGreedyAgent': EpsilonGreedyAgent(epsilon=0.1),
                'UCBAgent': UCBAgent(),
                'ThompsonSamplingAgent': ThompsonSamplingAgent()
            }
        elif env_name == 'Gaussian':
            agents = {
                'LLMAgent': LLMAgent(model="gpt-4.1-nano"),
                'GaussianEpsilonGreedyAgent': GaussianEpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1),
                'GaussianUCBAgent': GaussianUCBAgent(n_arms=n_arms, c=2),
                'GaussianThompsonSamplingAgent': GaussianThompsonSamplingAgent(n_arms=n_arms)
            }

        # Run simulation
        action_histories, detailed_log = evaluator.run_bandit_simulation(agents, env, n_trials, n_episodes)
        
        # Save detailed log and plot action distributions
        evaluator.save_detailed_log_to_csv(detailed_log, f"detailed_log_{env_label}_{n_arms}arms.csv")
        evaluator.plot_action_distribution(action_histories, env_label)

        # Convert detailed log list to a DataFrame for analysis
        detailed_log = pd.DataFrame(detailed_log)

        # --- Wasserstein Distance Analysis ---
        llm_history = detailed_log[detailed_log['agent_name'] == 'LLMAgent'][['action', 'reward']].values.tolist()
        llm_agent_name = 'LLMAgent'

        for agent_name in agents:
            if agent_name == llm_agent_name:
                continue

            other_history = detailed_log[detailed_log['agent_name'] == agent_name][['action', 'reward']].values.tolist()
            
            print(f"  Computing Wasserstein: LLM vs {_get_clean_label(agent_name)} for {env_name}")
            
            distances, actual_window_size = evaluator.compute_tumbling_window_wasserstein(
                llm_history, other_history, llm_agent_name, agent_name, n_arms, window_size, env_label
            )
            
            # Plot and save individual Wasserstein distance plot
            plot_filename = f"LLM{_get_clean_label(agent_name).replace('$-', '').replace('$', '')}_{env_label.lower()[:4]}.pdf"
            evaluator._plot_and_save_individual_wasserstein(
                distances, llm_agent_name, agent_name, env_label, actual_window_size, plot_filename
            )
        
        evaluator.plot_combined_wasserstein_for_env(env_label)

    # --- Final Summary and Save --- 
    evaluator.save_results_to_csv(filename="wasserstein_metrics_5arms_yaml.csv")
    evaluator.plot_summary_wasserstein_metrics()
    evaluator.plot_summary_wasserstein_violin()
    print("\n--- Evaluation Complete ---")
