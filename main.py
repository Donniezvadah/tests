# main.py
import numpy as np
import matplotlib.pyplot as plt
from environments.bernoulli_bandit import BernoulliBandit
from environments.gaussian_bandit import GaussianBandit, generate_configuration
from agents.llm_agent import LLMAgent
from agents.epsilon import EpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.thompson import ThompsonSamplingAgent
from utils.confidence import compute_confidence_interval
from plots.plot_utils import plot_regret_with_confidence
import os
import sys
from omegaconf import OmegaConf
import pandas as pd
from agents.gaussian_epsilon_greedy import GaussianEpsilonGreedyAgent
from agents.gaussian_ucb import GaussianUCBAgent
from agents.gaussian_thompson_sampling import GaussianThompsonSamplingAgent
from agents.ucb_kl import KLUCBAgent
from wasserstein import WassersteinEvaluator, get_clean_label


def load_config():
    """
    Load configuration from YAML files using OmegaConf.
    Returns a merged configuration dictionary.
    """
    try:
        # Load the main config file
        config = OmegaConf.load('configurations/config.yaml')
        # Load experiment config
        experiment_cfg = OmegaConf.load('configurations/experiment/experiment.yaml')
        # Load agent config (default: epsilon_greedy)
        agent_cfg = OmegaConf.load('configurations/agent/epsilon_greedy.yaml')
        # Load both environment configs
        bernoulli_env_cfg = OmegaConf.load('configurations/environment/bernoulli_env.yaml')
        gaussian_env_cfg = OmegaConf.load('configurations/environment/gaussian_env.yaml')
        # Merge all configs (default: Bernoulli)
        merged = OmegaConf.merge(config, experiment_cfg, agent_cfg, bernoulli_env_cfg)
        return OmegaConf.to_container(merged, resolve=True), bernoulli_env_cfg, gaussian_env_cfg
    except Exception as e:
        print(f"Error loading YAML configuration: {e}")
        sys.exit(1)

def run_simulation(env, agent, n_steps, n_trials, confidence_levels):
    """Run the bandit simulation with a single agent and log actions."""
    print(f"\nStarting simulation for {agent.name}...")
    regrets = np.zeros((n_trials, n_steps))
    cumulative_regrets = np.zeros((n_trials, n_steps))
    action_log = []
    
    for trial in range(n_trials):
        # print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Reset environment and agent
        env.reset()
        agent.init_actions(env.action_count)
        
        # Run simulation
        for step in range(n_steps):
            # Get action from agent
            action = agent.get_action()
            
            # Log action
            action_log.append({
                'episode': trial, 
                'trial': step, 
                'agent_name': get_clean_label(agent.name), 
                'action': action
            })

            # Get reward from environment
            reward = env.pull(action)
            
            # Update agent
            agent.update(action, reward)
            
            # Calculate regret
            optimal_reward = env.optimal_reward()
            regrets[trial, step] = optimal_reward - reward
            if step == 0:
                cumulative_regrets[trial, step] = regrets[trial, step]
            else:
                cumulative_regrets[trial, step] = cumulative_regrets[trial, step-1] + regrets[trial, step]
    
    # Calculate confidence intervals
    # print("Computing confidence intervals...")
    confidence_intervals = {}
    for level in confidence_levels:
        ci = compute_confidence_interval(cumulative_regrets, level)
        confidence_intervals.update(ci)
    
    return cumulative_regrets, confidence_intervals, action_log

def main():
    print("Starting main function...")
    try:
        # Load configuration
        print("Loading configuration...")
        config, bernoulli_env_cfg, gaussian_env_cfg = load_config()
        print("Configuration loaded successfully")

        # Override for a single run with T=25 and n_runs=5
        print("Overriding config for a single run: T=50, n_runs=5")
        config['experiment']['n_steps'] = 50    
        config['experiment']['n_runs'] = 25
        
        # Set random seeds
        print("Setting random seeds...")
        np.random.seed(config['seeds']['numpy'])
        print(f"Random seed set to: {config['seeds']['numpy']}")
        
        # --- Bernoulli Environment ---
        print("\n--- Testing Bernoulli Environment ---")
        probs = np.array([float(prob) for prob in bernoulli_env_cfg['probabilities']])
        env_bernoulli = BernoulliBandit(n_actions=len(probs), probs=probs)
        agents_bernoulli = [
            EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli'),
            UCBAgent(),
            KLUCBAgent(n_arms=len(probs)),
            ThompsonSamplingAgent(environment_type='bernoulli'),
            LLMAgent(model="gpt-4.1-nano"),
        ]
        
        all_regrets_bernoulli = {}
        all_intervals_bernoulli = {}
        all_actions_bernoulli = []

        for agent in agents_bernoulli:
            regrets, intervals, action_log = run_simulation(
                env_bernoulli, agent, config['experiment']['n_steps'],
                config['experiment']['n_runs'], config.get('confidence_levels', [0.95])
            )
            all_regrets_bernoulli[agent.name] = regrets
            all_intervals_bernoulli[agent.name] = intervals
            all_actions_bernoulli.extend(action_log)
        
        plot_regret_with_confidence(
            agents_bernoulli, all_regrets_bernoulli, all_intervals_bernoulli,
            config, "Bernoulli"
        )
        
        # --- Wasserstein Analysis for Bernoulli ---
        print("\n--- Wasserstein Analysis for Bernoulli ---")
        df_log_bernoulli = pd.DataFrame(all_actions_bernoulli)
        
        wasserstein_evaluator = WassersteinEvaluator(output_dir="wasserstein_analysis_output_main")
        llm_label = get_clean_label("LLMAgent")
        window_size = 5 # Reduced from 100 due to short run (T=25)
        n_trials = config['experiment']['n_steps']
        
        env_label_b = "Bernoulli"
        wasserstein_evaluator.all_distances[env_label_b] = {}
        
        other_agents_b = [agent for agent in agents_bernoulli if not isinstance(agent, LLMAgent)]

        for agent in other_agents_b:
            other_label = get_clean_label(agent.name)
            comparison_label = f"{llm_label} vs. {other_label}"
            
            distances = wasserstein_evaluator.compute_wasserstein_distance(
                df_log=df_log_bernoulli,
                llm_agent_name=llm_label,
                other_agent_name=other_label,
                n_arms=len(probs),
                n_trials=n_trials,
                window_size=window_size
            )
            
            if distances:
                wasserstein_evaluator.all_distances[env_label_b][comparison_label] = distances
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                wasserstein_evaluator.results.append({
                    'environment': env_label_b,
                    'comparison': comparison_label,
                    'mean_dist': mean_dist,
                    'std_dist': std_dist,
                    'window_size': window_size
                })

        wasserstein_evaluator.plot_combined_distances(env_label_b, window_size, config['experiment']['n_runs'])
        
        # --- Gaussian Environment ---
        print("\n--- Testing Gaussian Environment ---")
        means = np.array([float(mean) for mean in gaussian_env_cfg['means']])
        stds = np.array([float(std) for std in gaussian_env_cfg['stds']])
        env_gaussian = GaussianBandit(n_actions=len(means))
        env_gaussian.set(means, stds)
        
        agents_gaussian = [
            GaussianEpsilonGreedyAgent(n_arms=len(means), epsilon=0.1),
            GaussianUCBAgent(n_arms=len(means)),
            GaussianThompsonSamplingAgent(n_arms=len(means)),
            LLMAgent(model="gpt-4.1-nano"),
        ]
        
        all_regrets_gaussian = {}
        all_intervals_gaussian = {}
        all_actions_gaussian = []
        
        for agent in agents_gaussian:
            regrets, intervals, action_log = run_simulation(
                env_gaussian, agent, config['experiment']['n_steps'],
                config['experiment']['n_runs'], config.get('confidence_levels', [0.95])
            )
            all_regrets_gaussian[agent.name] = regrets
            all_intervals_gaussian[agent.name] = intervals
            all_actions_gaussian.extend(action_log)
        
        plot_regret_with_confidence(
            agents_gaussian, all_regrets_gaussian, all_intervals_gaussian,
            config, "Gaussian"
        )
        
        # --- Wasserstein Analysis for Gaussian ---
        print("\n--- Wasserstein Analysis for Gaussian ---")
        df_log_gaussian = pd.DataFrame(all_actions_gaussian)
        
        env_label_g = "Gaussian"
        wasserstein_evaluator.all_distances[env_label_g] = {}
        other_agents_g = [agent for agent in agents_gaussian if not isinstance(agent, LLMAgent)]

        for agent in other_agents_g:
            other_label = get_clean_label(agent.name)
            comparison_label = f"{llm_label} vs. {other_label}"
            
            distances = wasserstein_evaluator.compute_wasserstein_distance(
                df_log=df_log_gaussian,
                llm_agent_name=llm_label,
                other_agent_name=other_label,
                n_arms=len(means),
                n_trials=n_trials,
                window_size=window_size
            )
            
            if distances:
                wasserstein_evaluator.all_distances[env_label_g][comparison_label] = distances
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                wasserstein_evaluator.results.append({
                    'environment': env_label_g,
                    'comparison': comparison_label,
                    'mean_dist': mean_dist,
                    'std_dist': std_dist,
                    'window_size': window_size
                })
        
        wasserstein_evaluator.plot_combined_distances(env_label_g, window_size, config['experiment']['n_runs'])
        
        # --- Final Summary Report ---
        wasserstein_evaluator.save_summary_results()

        print("\nDone!")
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

# Add seeder for the bandits algorithms
#Now do a confidence interval 99% for the regret
# Create a folder called agents and save the agents in there and their strategies , and __init__.py file to import the agents add base agent  
# create a folder for the plots and save the plots in there
# folder for configuration and save the configuration in there 

