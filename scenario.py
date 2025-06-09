import numpy as np
import random
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
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
            mean1 = np.random.choice([-0.2, 0.2])
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

    def _get_clean_label(self, agent_name):
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

    # Set global matplotlib style for all plots
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
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="colorblind")

    def plot_cumulative_regret(self, results, optimal_per_episode, scenario, env_type):
        import numpy as np
        import matplotlib.pyplot as plt
        color_cycle = sns.color_palette("colorblind")
        marker_cycle = ['o', 's', '^', 'D', 'P', 'X', '*']
        plt.figure(figsize=(6.5, 4))
        n_episodes, n_trials = next(iter(results.values())).shape
        for i, (agent_name, rewards) in enumerate(results.items()):
            cumulative_regret = self.calculate_cumulative_regret(rewards, optimal_per_episode)
            mean_regret = np.mean(cumulative_regret, axis=0)
            std_regret = np.std(cumulative_regret, axis=0)
            label = self._get_clean_label(agent_name)
            if agent_name == 'LLM':
                plt.plot(
                    np.arange(n_trials), mean_regret, label=label,
                    color='#FF0000',
                    linewidth=0.7
                )
            else:
                plt.plot(
                    np.arange(n_trials), mean_regret, label=label,
                    color=color_cycle[i % len(color_cycle)],
                    linewidth=0.7
                )
            # Error band: ±1 standard deviation
            if agent_name == 'LLM':
                plt.fill_between(
                    np.arange(n_trials),
                    mean_regret - std_regret,
                    mean_regret + std_regret,
                    color='#FF0000',
                    alpha=0.18, linewidth=0.7
                )
            else:
                plt.fill_between(
                    np.arange(n_trials),
                    mean_regret - std_regret,
                    mean_regret + std_regret,
                    color=color_cycle[i % len(color_cycle)],
                    alpha=0.18, linewidth=0.7
                )
        plt.xlabel("$t$", fontweight="bold")
        plt.ylabel("$R(t)$", fontweight="bold")
        plt.grid(True, axis='y', linestyle=':', alpha=0.3)
        plt.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=len(results),
            frameon=False, handletextpad=0.7, columnspacing=1.2, borderaxespad=0.3
        )
        plt.tight_layout(pad=0.2)
        fname_base = f"{self.plot_dir}/{env_type.lower()}_{scenario}_cumulative_regret"
        fname_base = fname_base.replace(' ', '_')
        plt.savefig(fname_base + ".pdf", bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_heatmap(self, regret_matrix, row_labels, col_labels, title, filename):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(7, 4.5))
        cmap = sns.color_palette("light:b", as_cmap=True)
        ax = sns.heatmap(
            regret_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            xticklabels=[self._get_clean_label(l) for l in col_labels],
            yticklabels=[self._get_clean_label(l) for l in row_labels],
            cbar_kws={'label': 'Final Cumulative Regret'},
            linewidths=0.4,
            linecolor='white',
            annot_kws={"fontsize":13, "fontweight":'bold'}
        )
        
        

        if ax.collections and hasattr(ax.collections[0], 'colorbar') and ax.collections[0].colorbar:
            ax.collections[0].colorbar.remove()
        sns.despine()
        plt.tight_layout(pad=0.1)
        fname = os.path.join(self.plot_dir, filename.replace('.png', '.pdf').replace(' ', '_'))
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close()


    def _plot_regret_box(self, agent_final, scenario, env_type):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from itertools import combinations
        from scipy.stats import ttest_ind
        color_cycle = sns.color_palette("colorblind")
        # Prepare data
        plt.figure(figsize=(8, 5))
        data = []
        for agent, vals in agent_final.items():
            for v in vals:
                data.append({'Agent': self._get_clean_label(agent), 'FinalRegret': v})
        df = pd.DataFrame(data)
        ax = sns.boxplot(x='Agent', y='FinalRegret', data=df, palette=color_cycle, linewidth=1.7, fliersize=0)
        sns.stripplot(x='Agent', y='FinalRegret', data=df, color='black', size=6, jitter=0.18, ax=ax, alpha=0.6)
        
        plt.ylabel('Final Cumulative Regret', fontsize=15, fontweight='bold', labelpad=8)
        plt.xticks(fontsize=13, fontweight='bold')
        plt.yticks(fontsize=13, fontweight='bold')
        ax.grid(False)
        plt.gcf().patch.set_facecolor('white')
        ax.set_facecolor('white')
        # Annotate means and stds above each box
        agent_names = list(df['Agent'].unique())
        for i, agent in enumerate(agent_names):
            vals = df[df['Agent'] == agent]['FinalRegret'].values
            mean = vals.mean()
            std = vals.std()
            ann_y = vals.max() + 0.05*(df['FinalRegret'].max()-df['FinalRegret'].min())
            ax.text(i, ann_y, f"μ={mean:.2f}\nσ={std:.2f}", ha='center', va='bottom', fontsize=11, fontweight='normal', color='black', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.13', alpha=0.85))
        # Significance stars for pairwise t-tests
        pairs = list(combinations(agent_names, 2))
        y_max = df['FinalRegret'].max()
        star_height = y_max + 0.12*(y_max-df['FinalRegret'].min())
        step = 0.07*(y_max-df['FinalRegret'].min())
        for idx, (a1, a2) in enumerate(pairs):
            x1, x2 = agent_names.index(a1), agent_names.index(a2)
            vals1, vals2 = df[df['Agent'] == a1]['FinalRegret'].values, df[df['Agent'] == a2]['FinalRegret'].values
            t, p = ttest_ind(vals1, vals2, equal_var=False)
            if p < 0.001:
                star = '***'
            elif p < 0.01:
                star = '**'
            elif p < 0.05:
                star = '*'
            else:
                star = ''
            if star:
                y = star_height + idx*step
                ax.plot([x1, x1, x2, x2], [y-0.01, y, y, y-0.01], lw=1.5, c='k')
                ax.text((x1+x2)/2, y+0.01, star, ha='center', va='bottom', color='red', fontsize=18, fontweight='bold')
        # Add legend/key for stars
        legend_text = "* p<0.05   ** p<0.01   *** p<0.001"
        plt.text(0.5, -0.19, legend_text, ha='center', va='center', fontsize=13, color='black', fontweight='bold', transform=ax.transAxes)
        plt.tight_layout(rect=[0,0.13,1,1])
        fname = f'{self.plot_dir}/{env_type.lower()}_{scenario}_regret_box.pdf'.replace(' ', '_')
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close()

    # === Statistical analysis and reporting methods ===
    @staticmethod
    def _latex_escape(s):
        return s.replace('_', '\_')

    def _get_final_cumregret(self, results, optimal):
        """Returns dict: agent_name -> [final cumulative regret per episode]"""
        agent_final = {}
        for agent_name, rewards in results.items():
            cumreg = self.calculate_cumulative_regret(rewards, optimal)
            agent_final[agent_name] = cumreg[:, -1]
        return agent_final

    @staticmethod
    def _pairwise_stats(agent_final):
        from scipy.stats import ttest_ind
        import numpy as np
        keys = list(agent_final.keys())
        stats = []
        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                if j <= i:
                    continue
                arr1, arr2 = agent_final[k1], agent_final[k2]
                t, p = ttest_ind(arr1, arr2, equal_var=False)
                stats.append((k1, k2, np.mean(arr1), np.std(arr1), np.mean(arr2), np.std(arr2), np.mean(arr1)-np.mean(arr2), t, p))
        return stats

    def _latex_table(self, stats, scenario, env_type):
        def pval_star(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return ''
        def bold_low(a, b):
            # Returns (a_str, b_str) with lower mean bolded
            if a < b:
                return f'\\textbf{{{a:.2f}}}', f'{b:.2f}'
            elif b < a:
                return f'{a:.2f}', f'\\textbf{{{b:.2f}}}'
            else:
                return f'{a:.2f}', f'{b:.2f}'
        header = r"""
\begin{table}[ht]
\centering
\caption{Detailed hypothesis testing for final cumulative regret differences between agents in the %s bandit (%s scenario). Means (μ) and standard deviations (σ) are shown for each agent. Significance stars: * p$<$0.05, ** p$<$0.01, *** p$<$0.001. The lower mean in each pair is bolded.}
\begin{tabular}{l l c c c c c}
\toprule
Agent 1 & Agent 2 & $\mu_1$ & $\sigma_1$ & $\mu_2$ & $\sigma_2$ & p-value \\
\midrule
""" % (env_type.capitalize(), scenario)
        body = ""
        for row in stats:
            k1, k2, m1, s1, m2, s2, diff, t, p = row
            m1_str, m2_str = bold_low(m1, m2)
            star = pval_star(p)
            body += f"{self._latex_escape(k1)} & {self._latex_escape(k2)} & {m1_str} & {s1:.2f} & {m2_str} & {s2:.2f} & {p:.3g}{star} \\\n"
        footer = r"\bottomrule\end{tabular}\vspace{0.5em}\newline\textbf{Key:} * p$<$0.05, ** p$<$0.01, *** p$<$0.001. Lower mean bolded.\end{table}"
        return header + body + footer

    def compare_agent_regrets_with_stats(self, easy_results, easy_optimal, hard_results, hard_optimal, env_type):
        import os
        os.makedirs(self.plot_dir, exist_ok=True)
        for scenario, results, optimal in [
            ("easy", easy_results, easy_optimal),
            ("hard", hard_results, hard_optimal)
        ]:
            agent_final = self._get_final_cumregret(results, optimal)
            stats = self._pairwise_stats(agent_final)
            self._plot_regret_box(agent_final, scenario, env_type)
            latex = self._latex_table(stats, scenario, env_type)
            with open(os.path.join(self.plot_dir, f"{env_type}_{scenario}_regret_stats.tex"), "w") as f:
                f.write(latex)

def main():
    scenarios = ['easy', 'medium', 'hard', 'uniform']
    bandit_scenario = BanditScenario(n_trials=50, n_episodes=20)
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

    print("Plots saved: cumulative regret curves, subplots, and heatmaps.")

    # Run hypothesis testing and visualization on easy/hard for both bandit types
    bandit_scenario.compare_agent_regrets_with_stats(
        bernoulli_results_dict['easy'], bernoulli_optimal_dict['easy'],
        bernoulli_results_dict['hard'], bernoulli_optimal_dict['hard'],
        'Bernoulli')
    bandit_scenario.compare_agent_regrets_with_stats(
        gaussian_results_dict['easy'], gaussian_optimal_dict['easy'],
        gaussian_results_dict['hard'], gaussian_optimal_dict['hard'],
        'Gaussian')
    print("Hypothesis testing and visualization done.")

if __name__ == "__main__":
    main() 