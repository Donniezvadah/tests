import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib as mpl # Added
from agents.llm_agent import LLMAgent
from agents.epsilon import EpsilonGreedyAgent
from agents.gaussian_epsilon_greedy import GaussianEpsilonGreedyAgent
from agents.ucb import UCBAgent
from agents.gaussian_ucb import GaussianUCBAgent
from agents.thompson import ThompsonSamplingAgent
from agents.gaussian_thompson_sampling import GaussianThompsonSamplingAgent
from environments.bernoulli_bandit import BernoulliBandit
from environments.gaussian_bandit import GaussianBandit

# Set global matplotlib style for all plots
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
    'pdf.fonttype': 42, # For PDF export
    'ps.fonttype': 42,  # For PS export
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
sns.set_theme(style="whitegrid", palette="colorblind") # Updated

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
    # Helper function for legend labels
    def _get_clean_label(agent_name):
        # Standardize legend labels for publication
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

    # Color mapping - keys should match output of _get_clean_label
    color_map = {
        r'$\epsilon$-greedy': '#0173b2',  # Blue
        'UCB': '#de8f05',              # Orange
        'TS': '#029e73',               # Green
        'LLM': '#d55e00'               # Red
    }
    palette = sns.color_palette("colorblind") # Fallback palette for any agent not in color_map

    # Plotting
    plt.figure(figsize=(9, 6))
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']  # Distinct markers for each agent

    for idx, (name, regret_list) in enumerate(regrets.items()):
        clean_label = _get_clean_label(name)
        color = color_map.get(clean_label, palette[idx % len(palette)]) # Get specific color or fallback
        marker = markers[idx % len(markers)]
        plt.plot(delays, regret_list, label=clean_label, linewidth=1.5, color=color, marker=marker, markersize=7)

    plt.xlabel('Delay $\\delta$', fontweight="bold") # X-axis label style updated
    plt.ylabel('$R(t)$', fontweight="bold") # Y-axis label style updated

    plt.grid(True, axis='y', linestyle=':', alpha=0.3) # Updated grid

    plt.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=len(agents),
        frameon=False, handletextpad=0.7, columnspacing=1.2, borderaxespad=0.3
    ) # Updated legend

    plt.xticks(delays) # Fontsize managed by rcParams
    plt.yticks()       # Fontsize managed by rcParams

    plt.tight_layout(pad=0.2) # Updated tight_layout

    # Updated plot saving filename and parameters
    fname_base = os.path.join(RESULTS_DIR, f"{env_name.lower()}_delayed_cumulative_regret")
    pdf_plot_path = fname_base + ".pdf"
    plt.savefig(pdf_plot_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=600)
    plt.close()

    # LaTeX Table (mean ± std)
    df_mean = pd.DataFrame(regrets, index=delays)
    df_std = pd.DataFrame(stds, index=delays)
    df_combined = df_mean.round(2).astype(str) + ' $\\pm$ ' + df_std.round(2).astype(str)
    df_combined.index.name = 'Delay $\\delta$'
    latex_table = df_combined.T.to_latex(
        caption=f"Cumulative regret (mean $\\pm$ std) for each agent and feedback delay ($\delta$) in the {env_name} bandit.",
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
    print(f"Saved plot to {pdf_plot_path} and LaTeX table to {table_path}")


def main():
    n_arms = 5
    T = 20
    runs = 20
    delays = [1, 2,4, 5,10,20]
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
