import numpy as np
from scipy.stats import wasserstein_distance
from typing import List, Dict, Tuple, Union, Optional
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


if __name__ == "__main__":
    # --- Evaluation Configuration ---
    n_trials = 2000 # To allow for 20 non-overlapping 100-step windows
    n_episodes = 25   # For action distribution comparison, 1 episode suffices
    window_size = 100 # As requested for Wasserstein windows
    output_dir = "Wasserstein_plots_5arms_yaml"
    os.makedirs(output_dir, exist_ok=True)
    evaluator = AgentBehaviorEvaluator(output_dir=output_dir)

    # Load environment configurations from YAML
    base_config_path = "configurations/environment"
    bernoulli_config_path = os.path.join(base_config_path, "bernoulli_env.yaml")
    gaussian_config_path = os.path.join(base_config_path, "gaussian_env.yaml")

    bernoulli_config = load_env_config(bernoulli_config_path)
    gaussian_config = load_env_config(gaussian_config_path)

    # Extract parameters
    bernoulli_probs = bernoulli_config['probabilities']
    bernoulli_seed = bernoulli_config.get('seed') # Allows seed to be optional in YAML
    n_arms_bernoulli = len(bernoulli_probs)

    gaussian_means = gaussian_config['means']
    gaussian_stds = gaussian_config['stds']
    gaussian_seed = gaussian_config.get('seed') # Allows seed to be optional in YAML
    n_arms_gaussian = len(gaussian_means)

    if n_arms_bernoulli != n_arms_gaussian:
        raise ValueError(
            f"Mismatch in number of arms: Bernoulli ({n_arms_bernoulli}) vs Gaussian ({n_arms_gaussian}). "
            "Please ensure YAML configurations define the same number of arms."
        )
    n_arms = n_arms_bernoulli # Use a single n_arms variable
    print(f"Running evaluation with {n_arms} arms, loaded from YAML configurations.")

    # Bernoulli agents
    bernoulli_agents = [
        LLMAgent(model="gpt-4.1-nano"),
        EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli'),
        UCBAgent(),
        ThompsonSamplingAgent(environment_type='bernoulli')
    ]
    # Gaussian agents - ensure they are initialized correctly for n_arms
    gaussian_agents = [
        LLMAgent(model="gpt-4.1-nano"),
        GaussianEpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1),
        GaussianUCBAgent(n_arms=n_arms),
        GaussianThompsonSamplingAgent(n_arms=n_arms)
    ]
    
    envs = [
        (BernoulliBandit(n_actions=n_arms, probs=bernoulli_probs, seed=bernoulli_seed), bernoulli_agents, 'Bernoulli'),
        (GaussianBandit(n_actions=n_arms, means=gaussian_means, stds=gaussian_stds, seed=gaussian_seed), gaussian_agents, 'Gaussian')
    ]

    for env, agents, env_label in envs:
        print(f"\n--- Running Evaluation for {env_label} Environment ({n_arms} arms) ---")
        # Collect action trajectories for all agents
        agent_states_dict = {}
        for agent in agents:
            print(f"  Simulating agent: {_get_clean_label(agent.name)}...")
            agent_states = []
            env.reset() # Reset environment for each agent
            agent.init_actions(n_arms) # Initialize/reset agent for n_arms
            for trial in range(n_trials):
                try:
                    action = agent.get_action()
                except Exception as e:
                    # print(f"    Error getting action for {agent.name}: {e}. Choosing random.")
                    action = np.random.choice(n_arms)
                reward = env.pull(action)
                try:
                    agent.update(action, reward)
                except Exception as e:
                    # print(f"    Error updating agent {agent.name}: {e}")
                    pass
                agent_states.append([action, reward])
            agent_states_dict[agent.name] = agent_states
        
        # Compute and plot Wasserstein for LLM vs each other agent
        llm_agent_key = None
        for key_in_dict in agent_states_dict.keys():
            if key_in_dict.startswith('LLM'):
                llm_agent_key = key_in_dict
                break
        
        if llm_agent_key is None:
            print(f"LLM agent data not found for {env_label}. Skipping Wasserstein plots for this environment.")
            continue # Skip to next environment if no LLM data
            
        for other_agent_name_key in agent_states_dict.keys():
            if other_agent_name_key == llm_agent_key: # Correctly skip LLM vs LLM
                continue

            # Generate custom filename
            other_agent_short_label = ""
            # Determine a short code for the filename based on other_agent_name_key (the raw name)
            if 'EpsilonGreedy' in other_agent_name_key:
                other_agent_short_label = "EG"
            elif 'UCB' in other_agent_name_key:
                other_agent_short_label = "UCB"
            elif 'ThompsonSampling' in other_agent_name_key:
                other_agent_short_label = "TS"
            else:
                # Skip if not one of the main agent types we want to compare against LLM
                continue 

            env_suffix = "bern" if "Bernoulli" in env_label else "gaus"
            custom_filename = f"LLM{other_agent_short_label}_{env_suffix}.pdf"
            
            # Display labels for console and plot titles
            print(f"  Computing Wasserstein: {_get_clean_label(llm_agent_key)} vs {_get_clean_label(other_agent_name_key)} for {env_label}")
            
            wasserstein_distances, actual_window_size = evaluator.compute_tumbling_window_wasserstein(
                llm_actions_rewards=agent_states_dict[llm_agent_key],
                other_actions_rewards=agent_states_dict[other_agent_name_key],
                llm_name_key=llm_agent_key, # Pass the actual key
                other_name_key=other_agent_name_key, # Pass the actual key
                n_arms=n_arms,
                window_size=window_size, # User has set this to 100
                env_label=env_label
            )
            
            # Now plot the individual comparison
            evaluator._plot_and_save_individual_wasserstein(
                wasserstein_distances=wasserstein_distances,
                llm_name_key=llm_agent_key,
                other_name_key=other_agent_name_key,
                env_label=env_label,
                actual_window_size=actual_window_size,
                custom_plot_filename=custom_filename
            )
        
        # After processing all agents for this environment, generate the combined plot for this environment
        evaluator.plot_combined_wasserstein_for_env(env_label)
        evaluator.plot_wasserstein_heatmap_for_env(env_label)
            
    # After all environments are processed, save results and generate summary plot
    evaluator.save_results_to_csv(filename="wasserstein_metrics_5arms_yaml.xlsx")
    evaluator.plot_summary_wasserstein_metrics()
    print("\n--- Evaluation Complete ---")

