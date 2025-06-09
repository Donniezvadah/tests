import numpy as np
from scipy.stats import wasserstein_distance
from typing import List, Dict, Tuple, Union, Optional
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
import yaml
from datetime import datetime
import importlib
import inspect
import sys

# Set global matplotlib style for all plots (publication-grade, serif, colorblind-friendly)
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
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
    if 'Gaussian' in agent_name and 'Epsilon' in agent_name:
        return r'$\epsilon$-greedy'
    if 'Gaussian' in agent_name and 'Thompson' in agent_name:
        return 'TS'
    if 'Gaussian' in agent_name and 'UCB' in agent_name:
        return 'UCB'
    if agent_name == 'Epsilon-Greedy':
        return r'$\epsilon$-greedy'
    if agent_name == 'Thompson Sampling':
        return 'TS'
    if agent_name == 'UCB':
        return 'UCB'
    if agent_name == 'LLM':
        return 'LLM'
    return agent_name

class AgentBehaviorEvaluator:
    """
    A class for evaluating and comparing agent behaviors using Wasserstein metrics.
    This is a minimal stub for test run.
    """
    def __init__(self, output_dir: str = "Wasserstein_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.agent_states = {}
        self.window_wass_results = []

    def compute_sliding_window_wasserstein(self, llm_actions_rewards, other_actions_rewards, 
                                           llm_name_key, other_name_key, # These are the actual keys from dict
                                           n_arms, window_size, env_label, 
                                           custom_plot_filename: Optional[str] = None):
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
            return

        if min_len - actual_window_size + 1 <= 0 and min_len > 0 : # Not enough data points for even one window
            print(f"      Not enough data points for a single window for {_get_clean_label(llm_name_key)} vs {_get_clean_label(other_name_key)} in {env_label} (min_len: {min_len}, window_size: {actual_window_size}).")
            return
            
        for i in range(min_len - actual_window_size + 1):
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
            return

        # Store results
        self.window_wass_results.append({
            'Environment': env_label,
            'LLM_Agent': _get_clean_label(llm_name_key), # Use _get_clean_label for display
            'Other_Agent': _get_clean_label(other_name_key), # Use _get_clean_label for display
            'Window_Size': actual_window_size,
            'Avg_Wasserstein_Dist': np.mean(wasserstein_distances),
            'Std_Wasserstein_Dist': np.std(wasserstein_distances)
        })

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(wasserstein_distances, label=f"Window size: {actual_window_size}")
        # Title uses cleaned labels (title generation removed as per request)
        # title = f'Sliding Window Wasserstein Distance: {_get_clean_label(llm_name_key)} vs {_get_clean_label(other_name_key)} ({env_label})'
        # plt.title(title, fontsize=14) 
        plt.xlabel('Window Start Index', fontsize=12, fontweight='bold')
        plt.ylabel('Wasserstein Distance', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Determine filename
        if custom_plot_filename:
            plot_filename = custom_plot_filename
        else:
            # Fallback to original naming convention if no custom name provided
            plot_filename = f"wasserstein_dist_{_get_clean_label(llm_name_key)}_vs_{_get_clean_label(other_name_key)}_{env_label}_ws{actual_window_size}.pdf"
        
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"      Plot saved to {plot_path}")

    def save_results_to_csv(self, filename: str = "wasserstein_metrics_test.csv"):
        window_wass_df = pd.DataFrame(self.window_wass_results)
        excel_path = os.path.join(self.output_dir, filename.replace('.csv', '.xlsx'))
        with pd.ExcelWriter(excel_path) as writer:
            if not window_wass_df.empty:
                window_wass_df.to_excel(writer, sheet_name='SlidingWindow_Wass', index=False)
        print(f"Results saved to {excel_path}")

if __name__ == "__main__":
    # --- Evaluation Configuration ---
    n_trials = 25  # Sufficient for multiple 100-step windows
    n_episodes = 25   # For action distribution comparison, 1 episode suffices
    window_size = 2 # As requested for Wasserstein windows
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
            
            evaluator.compute_sliding_window_wasserstein(
                llm_actions_rewards=agent_states_dict[llm_agent_key],
                other_actions_rewards=agent_states_dict[other_agent_name_key],
                llm_name_key=llm_agent_key, # Pass the actual key
                other_name_key=other_agent_name_key, # Pass the actual key
                n_arms=n_arms,
                window_size=window_size, # User has set this to 2
                env_label=env_label,
                custom_plot_filename=custom_filename
            )
            
    evaluator.save_results_to_csv(filename="wasserstein_metrics_5arms_yaml.xlsx")
    print("\n--- Evaluation Complete ---")
