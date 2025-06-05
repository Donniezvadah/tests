import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from agents.llm_agent import LLMAgent
from agents.epsilon import EpsilonGreedyAgent
from agents.gaussian_epsilon_greedy import GaussianEpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.gaussian_ucb import GaussianUCBAgent
from agents.thompson import ThompsonSamplingAgent
from agents.gaussian_thompson_sampling import GaussianThompsonSamplingAgent
from environments.bernoulli_bandit import BernoulliBandit
from environments.gaussian_bandit import GaussianBandit

sns.set_theme(style="whitegrid")

RESULTS_DIR = "delays"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Delayed feedback simulation ---
def simulate_delayed(agent, env, T, delay):
    agent.init_actions(env.action_count)
    rewards = []
    regrets = []
    pending_feedback = []  # (arm, reward)
    optimal_mean = env.get_optimal_mean() if hasattr(env, 'get_optimal_mean') else None
    optimal_reward = 0
    for t in range(T):
        arm = agent.get_action() if hasattr(agent, 'get_action') else agent.select_arm()
        reward = env.pull(arm)
        pending_feedback.append((arm, reward))
        if optimal_mean is not None:
            optimal_reward += optimal_mean
        else:
            # fallback: try to use env._probs or env.means
            if hasattr(env, '_probs'):
                optimal_reward += np.max(env._probs)
            elif hasattr(env, 'means'):
                optimal_reward += np.max(env.means)
        # Provide feedback every 'delay' steps or at the last step
        if (t + 1) % delay == 0 or t == T - 1:
            for a, r in pending_feedback:
                agent.update(a, r)
            pending_feedback = []
        rewards.append(reward)
        regrets.append(optimal_reward - np.sum(rewards))
    return regrets[-1]  # Final cumulative regret


def plot_delayed_regrets(env_name, agents, delays, T, runs, n_arms):
    regrets = {name: [] for name in agents}
    stds = {name: [] for name in agents}  # For detailed tables
    for delay in delays:
        for name, agent_class in agents.items():
            run_regrets = []
            for _ in range(runs):
                if env_name == 'Bernoulli':
                    env = BernoulliBandit(n_actions=n_arms)
                else:
                    env = GaussianBandit(n_actions=n_arms)
                    from environments.gaussian_bandit import generate_configuration
                    means, stds_ = generate_configuration(n_arms)
                    env.set(means, stds_)
                # Re-instantiate agent for each run to avoid state carryover
                if name == 'Epsilon-Greedy':
                    if env_name == 'Bernoulli':
                        agent = EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli')
                    else:
                        agent = GaussianEpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1)
                elif name == 'UCB':
                    if env_name == 'Bernoulli':
                        agent = UCBAgent()
                    else:
                        agent = GaussianUCBAgent(n_arms=n_arms)
                elif name == 'Thompson Sampling':
                    if env_name == 'Bernoulli':
                        agent = ThompsonSamplingAgent(environment_type='bernoulli')
                    else:
                        agent = GaussianThompsonSamplingAgent(n_arms=n_arms)
                elif name == 'LLM':
                    agent = LLMAgent(model="gpt-4.1-nano")
                else:
                    continue
                regret = simulate_delayed(agent, env, T, delay)
                run_regrets.append(regret)
            regrets[name].append(np.mean(run_regrets))
            stds[name].append(np.std(run_regrets))
    # Plotting
    plt.figure(figsize=(9, 6))
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    for idx, (name, regret_list) in enumerate(regrets.items()):
        plt.plot(delays, regret_list, marker=markers[idx % len(markers)], label=name, linewidth=2.5, markersize=11)
    plt.xlabel('Delay $\\delta$', fontsize=17, fontweight='bold')
    plt.ylabel('$R(t)$', fontsize=17, fontweight='bold')
    # plt.title(f'Impact of Delayed Feedback on {env_name} Bandit', fontsize=20, fontweight='bold', pad=18)
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=14)
    plt.xticks(delays, fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout(rect=[0, 0.14, 1, 1])
    # caption = (f"Figure: Effect of feedback delay ($\\delta$) on cumulative regret for {env_name} bandit. "
    #            "Each curve is an agent. Markers show each delay. Lower regret is better. "
    #            f"Results are averaged over {runs} runs. Error bars show standard deviation.")
    # plt.figtext(0.5, 0.04, wrap=True, horizontalalignment='center', fontsize=13)
    plot_path = os.path.join(RESULTS_DIR, f"{env_name.lower()}_delays.pdf")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight', transparent=True)
    plt.close()

    # LaTeX Table (mean ± std)
    df_mean = pd.DataFrame(regrets, index=delays)
    df_std = pd.DataFrame(stds, index=delays)
    df_combined = df_mean.round(2).astype(str) + ' $\\pm$ ' + df_std.round(2).astype(str)
    df_combined.index.name = 'Delay $\\delta$'
    latex_table = df_combined.T.to_latex(
        caption=f"Cumulative regret (mean $\\pm$ std) for each agent and feedback delay ($\\delta$) in the {env_name} bandit.",
        label=f"tab:{env_name.lower()}_delays",
        column_format='l' + 'c' * len(delays),
        bold_rows=True,
        escape=False,
        multicolumn=True,
        multicolumn_format='c',
        position='htbp',
    )
    table_path = os.path.join(RESULTS_DIR, f"{env_name.lower()}_delays_table.tex")
    with open(table_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved plot to {plot_path} and LaTeX table to {table_path}")


def main():
    n_arms = 5
    T = 20
    runs = 20
    delays = [1, 2, 4, 5,10,20]
    # Use agent class references for clarity in plot_delayed_regrets
    bernoulli_agents = {
        'Epsilon-Greedy': EpsilonGreedyAgent,
        'UCB': UCBAgent,
        'Thompson Sampling': ThompsonSamplingAgent,
        'LLM': LLMAgent,
    }
    gaussian_agents = {
        'Epsilon-Greedy': GaussianEpsilonGreedyAgent,
        'UCB': GaussianUCBAgent,
        'Thompson Sampling': GaussianThompsonSamplingAgent,
        'LLM': LLMAgent,
    }
    plot_delayed_regrets('Bernoulli', bernoulli_agents, delays, T, runs, n_arms)
    plot_delayed_regrets('Gaussian', gaussian_agents, delays, T, runs, n_arms)


if __name__ == "__main__":
    main()
