import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
import seaborn as sns

# Set global matplotlib style for company quality plots
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',  # Use a clean, professional sans-serif font
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'axes.labelweight': 'bold',
    'axes.edgecolor': '#333F4B',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.fontsize': 13,
    'legend.title_fontsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.dpi': 150,
    'savefig.dpi': 400,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.autolayout': True,
    'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
})
sns.set_theme(style="whitegrid", font_scale=1.2)

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
        plt.figure(figsize=(13, 8))  # Slightly wider for company slides
        
        # Define colors and line styles for different agent types
        agent_styles = {
            # Base agents
            'EpsilonGreedy': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2},
            'UCB': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 2},
            'ThompsonSampling': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2},
            'KL-UCB': {'color': '#d62728', 'linestyle': '-', 'linewidth': 2},
            # LLM agents (red dotted lines)
            'LLM': {'color': 'red', 'linestyle': ':', 'linewidth': 2.5},
            'LLMV2': {'color': 'red', 'linestyle': ':', 'linewidth': 2.5},
            # Gaussian variants (solid lines with same colors as base agents)
            'GaussianEpsilonGreedy': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2},
            'GaussianUCB': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 2},
            'GaussianThompsonSampling': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2},
        }
        
        # Fallback style for unknown agents
        default_style = {'color': '#7f7f7f', 'linestyle': '-', 'linewidth': 1.5}
        
        # Create a set to track which agent names we've already added to the legend
        legend_handles = {}
        
        for agent in agents:
            print(f"\nPlotting data for {agent.name}...")
            base_name = get_base_agent_name(agent)
            
            # Get style for this agent, or use default if not found
            style = agent_styles.get(base_name, default_style)
            
            # For LLM agents, use red dotted style
            if 'LLM' in agent.name or 'llm' in agent.name.lower():
                style = agent_styles.get('LLM', default_style)
                # Force red color and dotted line for all LLM agents
                style = {'color': 'red', 'linestyle': ':', 'linewidth': 2.5}
            
            avg_regret = np.mean(regret[agent.name], axis=0)
            print(f"Average regret shape: {avg_regret.shape}")
            
            # Plot the average regret curve
            print(f"Plotting average regret curve for {agent.name}")
            line, = plt.plot(avg_regret, 
                          label=agent.name,
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
                
        # Enhanced legend with frame, shadow, and company look
        plt.legend(frameon=True, fancybox=True, shadow=True, loc='upper left', fontsize=13, title="Agent", title_fontsize=14)
        
        plt.xlabel('Steps', fontsize=15, fontweight='bold', labelpad=8)
        plt.ylabel('Cumulative Regret', fontsize=15, fontweight='bold', labelpad=8)
        plt.title(f'Average Cumulative Regret\n{env_name} Environment', fontsize=18, fontweight='bold', pad=14)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
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
        
        # Save plots with environment name in filename
        base_filename = f"regret_with_ci_{env_name.lower()}"
        print(f"Saving plots to {plots_dir}")
        plt.savefig(os.path.join(plots_dir, f"{base_filename}.png"), dpi=400, bbox_inches='tight', transparent=False)
        plt.savefig(os.path.join(plots_dir, f"{base_filename}.pdf"), bbox_inches='tight', transparent=False)
        plt.close()
        print("Plotting completed successfully")
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__}")
        raise 