import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
import seaborn as sns

# Set global matplotlib style for publication quality plots
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

# Helper to get base agent name for color mapping
# (e.g., EpsilonGreedy from EpsilonGreedy(epsilon=0.1, bernoulli))
def get_base_agent_name(agent):
    return getattr(agent, '_name', str(agent).split('(')[0])

def plot_regret_with_confidence(agents, regret, confidence_intervals, config, env_name):
    """
    Plot regret with confidence intervals and save to plots directory.
    
    Args:
        agents: List of agents
        regret: Dictionary of regret data
        confidence_intervals: Dictionary of confidence intervals
        config: Configuration dictionary
        env_name: Name of the environment (for file naming)
    """
    try:
        plt.figure(figsize=(13, 8))
        
        # Helper function for legend labels
        def _get_clean_label(agent_name):
            # Standardize legend labels for publication
            # Gaussian variants use same label as base
            lower_name = agent_name.lower()
            if 'kl_ucb' in lower_name or 'kl-ucb' in lower_name or 'klucb' in lower_name:
                return 'KL_UCB'
            if 'Epsilon' in agent_name:
                return r'$\epsilon$-greedy'
            if 'Thompson' in agent_name:
                return 'TS'
            if 'UCB' in agent_name:
                return 'UCB'
            if agent_name == 'LLM' or agent_name == 'LLMV2':
                return 'LLM'
            return agent_name

        # Color mapping - keys should match output of _get_clean_label
        color_map = {
            r'$\epsilon$-greedy': '#0173b2',  # Blue
            'UCB': '#de8f05',              # Orange
            'TS': '#029e73',               # Green
            'LLM': '#d55e00',              # Red
            r'Gaussian $\epsilon$-greedy': '#0173b2',  # Same color as base
            'Gaussian UCB': '#de8f05',     # Same color as base
            'Gaussian TS': '#029e73'       # Same color as base
        }

        # Color mapping - keys should match output of _get_clean_label
        color_map = {
            r'$\epsilon$-greedy': '#0173b2',  # Blue
            'UCB': '#de8f05',              # Orange
            'TS': '#029e73',               # Green
            'LLM': '#FF0000',              # Pure Red
            'KL_UCB': '#000000'            # Pure Black
        }
        palette = sns.color_palette("colorblind")  # Fallback palette for any agent not in color_map

        # Define styles based on clean labels
        agent_styles = {}
        for agent in agents:
            base_name = get_base_agent_name(agent)
            clean_label = _get_clean_label(base_name)
            color = color_map.get(clean_label, palette[len(agent_styles) % len(palette)])
            # Always use solid line for LLM
            linestyle = '-'
            agent_styles[base_name] = {
                'color': color,
                'linestyle': linestyle,  # Solid line for all agents, including LLM
                'linewidth': 1.2,  # Thin lines
                'label': clean_label
            }
        
        # Fallback style for unknown agents
        default_style = {'color': '#7f7f7f', 'linestyle': '-', 'linewidth': 1.5, 'label': 'Unknown'}
        
        # Create a set to track which agent names we've already added to the legend
        legend_handles = {}
        
        for agent in agents:
            print(f"\nPlotting data for {agent.name}...")
            base_name = get_base_agent_name(agent)
            
            # Get style for this agent, or use default if not found
            style = agent_styles.get(base_name, default_style)
            

            
            avg_regret = np.mean(regret[agent.name], axis=0)
            print(f"Average regret shape: {avg_regret.shape}")
            
            # Plot the average regret curve
            print(f"Plotting average regret curve for {agent.name}")
            line, = plt.plot(avg_regret, 
                          label=style['label'],  # Use the standardized label
                          color=style['color'],
                          linestyle=style['linestyle'],
                          linewidth=style['linewidth'],
                          marker=None)
            
            # Store the first line of each base type for the legend
            if base_name not in legend_handles:
                legend_handles[base_name] = line
            
            # Plot confidence intervals with matching style but lighter color
            print(f"Plotting confidence intervals for {agent.name}")
            print(f"Available confidence levels: {confidence_intervals[agent.name].keys()}")
            for level, (lower, upper) in confidence_intervals[agent.name].items():
                print(f"Plotting {level} confidence interval")
                plt.fill_between(range(len(upper)), lower, upper, 
                               color=style['color'], 
                               alpha=0.13, 
                               linewidth=0)
                
        # Enhanced legend: below plot, bold, clear
        plt.xlabel('$t$', fontsize=16, fontweight='bold', labelpad=8)
        plt.ylabel('$R(t)$', fontsize=16, fontweight='bold', labelpad=8)
        # Remove plot title
        # plt.title(f'Average Cumulative Regret\n{env_name} Environment', fontsize=16, fontweight='bold', pad=14)
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')

        # Legend: below plot, centered, single row, no box, clean style
        plt.legend(
            loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=len(agents),
            frameon=False, handletextpad=0.7, columnspacing=1.2, borderaxespad=0.3
        )
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.88])

        # Add subtle company branding (e.g., logo) if available
        logo_path = config['paths'].get('company_logo', None)
        if logo_path and os.path.exists(logo_path):
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            arr_logo = plt.imread(logo_path)
            imagebox = OffsetImage(arr_logo, zoom=0.08, alpha=0.8)
            ab = AnnotationBbox(imagebox, (0.96, 0.94), frameon=False, xycoords='axes fraction', box_alignment=(1,1))
            plt.gca().add_artist(ab)

        # Create plots directory if it doesn't exist
        plots_dir = config['paths']['plots_dir']
        os.makedirs(plots_dir, exist_ok=True)

        # Save plot as PDF only, highest quality
        base_filename = f"regret_with_ci_{env_name.lower()}"
        print(f"Saving plot to {plots_dir}")
        plt.savefig(os.path.join(plots_dir, f"{base_filename}.pdf"), bbox_inches='tight', transparent=False, dpi=600)
        plt.close()
        print("Plotting completed successfully")
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__}")
        raise 