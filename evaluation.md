# Evaluation Framework for Multi-Armed Bandit Agents (`evaluation.py`)

## 1. Overview

The `evaluation.py` script is designed to assess and compare the behavior of different multi-armed bandit (MAB) agents, with a particular focus on understanding how a Large Language Model (LLM) agent performs and adapts relative to traditional bandit algorithms. It achieves this by simulating agent interactions within defined bandit environments (Bernoulli and Gaussian) and then analyzing their action selection patterns using the Wasserstein distance, specifically within a sliding window.

The primary goals are:
*   **Behavioral Comparison**: To quantitatively compare the action selection strategies of an LLM-based agent against well-established MAB algorithms like Epsilon-Greedy, UCB (Upper Confidence Bound), and Thompson Sampling.
*   **Adaptability Assessment**: To observe how the LLM agent's strategy evolves over time and how similar or different it is to agents explicitly designed for specific reward distributions (e.g., GaussianThompsonSamplingAgent for Gaussian bandits). This is particularly relevant given the LLM agent's generic initialization, which requires it to infer environmental characteristics.
*   **Metric-Driven Insights**: To use a robust metric like the Wasserstein distance to capture nuanced differences in action distributions that simpler metrics (like average reward or regret) might miss.

## 2. Setup and Configuration

### 2.1. Environment Configuration
The script loads environment parameters from YAML files located in the `configurations/environment/` directory (e.g., `bernoulli_env.yaml`, `gaussian_env.yaml`). This allows for easy modification of:
*   **Bernoulli Bandit**: Probabilities of success for each arm (`probabilities`), and an optional `seed`.
*   **Gaussian Bandit**: Mean (`means`) and standard deviation (`stds`) for the reward distribution of each arm, and an optional `seed`.

The `load_env_config` function handles reading these YAML files. The script ensures consistency by checking that the number of arms defined in both Bernoulli and Gaussian configurations is the same.

### 2.2. Agent Initialization
A predefined set of agents is initialized for both Bernoulli and Gaussian environments:

*   **Common Agent**:
    *   `LLMAgent(model="gpt-4.1-nano")`: This agent is initialized generically, without explicit information about the environment type (Bernoulli or Gaussian) at the point of instantiation within the evaluation script. Its ability to adapt is a key aspect being evaluated.

*   **Bernoulli-Specific Agents**:
    *   `EpsilonGreedyAgent(epsilon=0.1, environment_type='bernoulli')`
    *   `UCBAgent()` (implicitly defaults to or is suited for Bernoulli rewards)
    *   `ThompsonSamplingAgent(environment_type='bernoulli')`

*   **Gaussian-Specific Agents**:
    *   `GaussianEpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1)`
    *   `GaussianUCBAgent(n_arms=n_arms)`
    *   `GaussianThompsonSamplingAgent(n_arms=n_arms)`

Each agent is (re)initialized for the correct number of arms (`n_arms`) using `agent.init_actions(n_arms)` before each simulation run in an environment.

### 2.3. Global Settings
*   **Matplotlib Styling**: The script sets a global, publication-quality style for all Matplotlib plots, ensuring consistency and readability (serif font, colorblind-friendly, clear axes).
*   **Output Directory**: Plots and results are saved to a specified `output_dir` (e.g., "Wasserstein_plots_5arms_yaml").

## 3. Core Class: `AgentBehaviorEvaluator`

This class encapsulates the logic for computing and visualizing the Wasserstein distance between agent action trajectories.

### 3.1. `compute_sliding_window_wasserstein`

This is the central method for behavioral comparison.

*   **Purpose**: To compare the action selection distributions of the LLM agent against another agent over time. The sliding window approach helps to understand how the similarity/dissimilarity between the two agents' strategies evolves as they gather more experience.

*   **Inputs**:
    *   `llm_actions_rewards`: A list of `[action, reward]` pairs for the LLM agent.
    *   `other_actions_rewards`: A list of `[action, reward]` pairs for the comparison agent.
    *   `llm_name_key`, `other_name_key`: String keys identifying the agents.
    *   `n_arms`: The number of arms in the bandit environment.
    *   `window_size`: The number of trials to include in each window for comparison.
    *   `env_label`: A string label for the environment (e.g., 'Bernoulli', 'Gaussian').
    *   `custom_plot_filename`: Optional custom filename for the output plot.

*   **Methodology**:
    1.  **Extract Actions**: Only the actions (arm choices) are extracted from the input lists.
    2.  **Sliding Window**: The method iterates through the action sequences of both agents, creating overlapping windows of `window_size` actions.
        *   Let `A_llm = [a_1, a_2, ..., a_T]` be the sequence of actions for the LLM agent.
        *   Let `A_other = [b_1, b_2, ..., b_T]` be the sequence of actions for the other agent.
        *   For each window `i` starting at trial `i` and ending at `i + window_size - 1`:
            *   `window_llm = [a_i, ..., a_{i + window_size - 1}]`
            *   `window_other = [b_i, ..., b_{i + window_size - 1}]`
    3.  **Empirical Distributions**: Within each window, empirical probability distributions (histograms) of actions are created for both agents.
        *   `hist_llm`: Normalized counts of each arm chosen by the LLM agent in `window_llm`.
        *   `hist_other`: Normalized counts of each arm chosen by the other agent in `window_other`.
        These histograms represent the discrete probability distributions `P_llm(k)` and `P_other(k)` for choosing arm `k` within that specific window.
    4.  **Wasserstein Distance Calculation**: The 1D Wasserstein distance (also known as Earth Mover's Distance) is computed between these two empirical distributions for each window.

*   **Wasserstein Distance (W<sub>1</sub>)**:
    *   **Conceptual Explanation**: The Wasserstein distance measures the minimum "cost" to transform one probability distribution into another. For 1D distributions, it can be visualized as the minimum amount of "work" (mass times distance) required to move the "earth" (probability mass) of one distribution to match the shape of the other. A smaller distance implies the distributions are more similar.
    *   **Mathematical Formulation (for 1D discrete distributions)**:
        Given two discrete probability distributions `u` and `v` over `n_arms` (represented by `hist_llm` and `hist_other`), where `u_k` and `v_k` are the probabilities of selecting arm `k` (where `k` ranges from `0` to `n_arms - 1`), the 1st Wasserstein distance `W_1(u, v)` can be calculated as:
        `W_1(u, v) = sum_{k=0}^{n_arms-1} |CDF_u(k) - CDF_v(k)|`
        where `CDF_u(k)` is the cumulative distribution function of `u` up to arm `k` (i.e., `sum_{j=0}^{k} u_j`), and similarly for `CDF_v(k)`.
        The `scipy.stats.wasserstein_distance` function is used, which for 1D distributions, takes the arm indices (`np.arange(n_arms)`) as value points and the histogram probabilities (`u_weights=hist_llm`, `v_weights=hist_other`) as the weights for these points.
    *   **Why it's a suitable metric here**:
        *   **Captures Geometric Differences**: Unlike metrics like KL-divergence, Wasserstein distance considers the "distance" between the arm indices. If one agent pulls arm 1 often and another pulls arm 2 often, the Wasserstein distance will reflect that these arms are close. If the second agent pulled arm 5 instead, the distance would be larger. This is crucial for MABs where arm indices might have an implicit ordering or where "close" arms might have similar reward properties.
        *   **Metric Properties**: It's a true metric (satisfies non-negativity, identity of indiscernibles, symmetry, and triangle inequality).
        *   **Robustness**: It handles distributions with non-overlapping support well.

    5.  **Plotting**: A plot is generated showing the Wasserstein distance over the sequence of window start indices. This visualizes how the similarity between the LLM agent and the other agent evolves.
    6.  **Results Storage**: The average and standard deviation of the Wasserstein distances across all windows for a given pair of agents and environment are stored in `self.window_wass_results`.

### 3.2. `save_results_to_csv`
*   **Purpose**: To aggregate all computed sliding window Wasserstein metrics (average and standard deviation for each comparison) into a single Excel file.
*   **Methodology**: Converts the `self.window_wass_results` list of dictionaries into a Pandas DataFrame and saves it to an `.xlsx` file with a sheet named 'SlidingWindow_Wass'.

## 4. Main Execution Logic (`if __name__ == "__main__":`)

The script's main block orchestrates the entire evaluation process:

1.  **Configuration**:
    *   Sets `n_trials` (total steps per simulation), `n_episodes` (number of full simulations, though for action distribution comparison, 1 episode is often sufficient as done here for `agent_states_dict`), `window_size` for Wasserstein calculation, and the `output_dir`.
    *   Initializes `AgentBehaviorEvaluator`.
    *   Loads environment configurations from YAML files using `load_env_config`.
    *   Extracts `n_arms` and other parameters (probabilities, means, stds, seeds).

2.  **Agent and Environment Setup**:
    *   Defines lists of `bernoulli_agents` and `gaussian_agents`.
    *   Creates a list `envs` containing tuples of `(environment_instance, list_of_agents_for_that_env, environment_label)`.

3.  **Simulation Loop**:
    *   Iterates through each `(env, agents, env_label)` in `envs`.
    *   For each environment:
        *   Prints a header indicating the current environment.
        *   Initializes `agent_states_dict` to store action-reward trajectories for each agent in the current environment.
        *   Iterates through each `agent` in the `agents` list for the current environment:
            *   Resets the environment (`env.reset()`).
            *   Initializes/resets the agent for the current `n_arms` (`agent.init_actions(n_arms)`).
            *   Runs a simulation for `n_trials`:
                *   Agent selects an `action` (`agent.get_action()`).
                *   Environment provides a `reward` (`env.pull(action)`).
                *   Agent updates its internal state (`agent.update(action, reward)`).
                *   The `[action, reward]` pair is stored.
            *   The complete trajectory for the agent is stored in `agent_states_dict` with the agent's name as the key.

4.  **Wasserstein Distance Calculation and Plotting**:
    *   Identifies the LLM agent's data (`llm_agent_key`) in `agent_states_dict`.
    *   If LLM data is found, it iterates through all other agents' data in `agent_states_dict`.
    *   For each `other_agent_name_key` (excluding comparison of LLM with itself):
        *   Generates a `custom_filename` for the plot (e.g., `LLMEG_bern.pdf` for LLM vs EpsilonGreedy in Bernoulli).
        *   Calls `evaluator.compute_sliding_window_wasserstein` with the action trajectories of the LLM agent and the current `other_agent`, along with `n_arms`, `window_size`, `env_label`, and the `custom_filename`.

5.  **Save Aggregate Results**:
    *   After processing all environments and agent pairs, calls `evaluator.save_results_to_csv()` to save the summary statistics.

## 5. Rationale for Metrics and Approach

*   **Why Compare Action Distributions?**:
    Simply looking at cumulative regret or average reward might not fully capture *how* an agent achieves its performance. Two agents could have similar overall rewards but arrive at them through very different exploration/exploitation strategies. Analyzing action distributions provides a deeper insight into the agent's decision-making process and learning behavior. For an LLM agent, understanding if its action patterns resemble those of known optimal or heuristic strategies is crucial for validation and trust.

*   **Benefits of Wasserstein Distance**:
    As discussed in section 3.1, the Wasserstein distance is sensitive to the underlying geometry of the action space. This means it can distinguish between an agent that explores arms far from the optimal one versus an agent that explores arms close to the optimal one, even if their exploration rates are similar. This is more informative than metrics that treat all non-optimal arms equally.

*   **Significance of the Sliding Window Approach**:
    Bandit algorithms are dynamic; their strategies evolve as they learn. A single, aggregate comparison of action distributions over all trials might obscure important temporal dynamics. The sliding window:
    *   Reveals how the similarity between two agents' strategies changes over time (e.g., do they converge, diverge, or oscillate?).
    *   Can highlight different phases of learning (e.g., initial exploration vs. later exploitation).

*   **Understanding LLM Agent Adaptability**:
    The provided memory states: "This generic initialization means the LLMAgent must adapt to different reward distributions based on its internal prompting and logic, without being explicitly told the environment type at setup time..."
    This evaluation framework directly addresses this by:
    1.  Running the *same* `LLMAgent` initialization in both Bernoulli and Gaussian environments.
    2.  Comparing its behavior to agents *specifically designed* for those environments (e.g., `ThompsonSamplingAgent` for Bernoulli, `GaussianThompsonSamplingAgent` for Gaussian).
    The Wasserstein distance plots and summary statistics help quantify how well the LLM agent's emergent strategy aligns with these specialized agents in each respective environment. If the LLM's action distribution becomes similar to that of, say, `ThompsonSamplingAgent` in a Bernoulli environment, it suggests successful adaptation.

## 6. Outputs

The script produces two main types of outputs:

1.  **Sliding Window Wasserstein Distance Plots**:
    *   PDF files saved in the `output_dir`.
    *   Each plot shows the Wasserstein distance between the LLM agent and one other traditional agent, for a specific environment, calculated over a sliding window of trials.
    *   The x-axis represents the starting index of the window, and the y-axis is the Wasserstein distance.
    *   Filenames are structured to be informative (e.g., `LLMEG_bern.pdf`, `LLMTS_gaus.pdf`).

2.  **Aggregated Metrics Excel File**:
    *   An `.xlsx` file (e.g., `wasserstein_metrics_5arms_yaml.xlsx`) saved in the `output_dir`.
    *   Contains a sheet named `SlidingWindow_Wass`.
    *   This sheet tabulates the average and standard deviation of the Wasserstein distances for each LLM vs. Other Agent comparison, across all windows, for each environment. This provides a summary of overall behavioral similarity.

This detailed evaluation helps in understanding not just *if* an LLM agent can perform well in MAB tasks, but *how* its strategy forms and compares to established algorithms, shedding light on its learning and adaptation mechanisms.