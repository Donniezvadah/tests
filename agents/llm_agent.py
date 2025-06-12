import numpy as np
import random
import time
import os
import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from .base_agent import BaseAgent

class LLMAgent(BaseAgent):
    """
    An agent that uses OpenAI API to make decisions in the bandit environment.
    This agent maintains a history of actions and rewards to provide context to the LLM.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4.1-nano", 
                 temperature: float = 0.0, max_retries: int = 3, 
                 timeout: int = 30,
                 cache_path: str = "llm_cache.json"): 
        """
        Initialize the LLM agent with enhanced error handling.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from llm_api.txt.
            model: The model to use. Default is "gpt-4.1-nano".
            temperature: Controls randomness in the response (0-1).
            max_retries: Maximum number of retries for API calls.
            timeout: Timeout in seconds for API calls.
        """
        super().__init__("LLM")
        self.model = model
        self.temperature = max(0, min(1, temperature))  # Clamp between 0 and 1
        self.max_retries = max(1, max_retries)
        self.timeout = max(5, timeout)  # Minimum 5 second timeout
        self.cache_path = cache_path
        self._cache = self._load_cache()
        
        # Initialize state
        self.last_llm_response = None # To store the latest reasoning
        self._rewards = None
        self._counts = None
        self._action_history = []
        self._reward_history = []
        self._context_window = 100# Increased context window
        self._last_api_call = 0
        self._min_call_interval = 0.1  # 100ms between API calls to avoid rate limiting
        
        # Set up API client
        if api_key is None:
            try:
                with open('llm_api.txt', 'r') as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                raise ValueError("API key not provided and llm_api.txt not found")
        
        if not api_key:
            raise ValueError("API key is empty. Please provide a valid OpenAI API key.")
        
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        
        # Test the API connection
        self._test_connection()

    def update(self, action: int, reward: float):
        """
        Update the agent's internal state with the result of an action.
        
        Args:
            action (int): The action that was taken.
            reward (float): The reward that was received.
        """
        self._rewards[action] += reward
        self._counts[action] += 1
        self._action_history.append(action)
        self._reward_history.append(reward)

    def init_actions(self, n_actions):
        """
        Initialize the agent's internal state.
        
        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)
        self._rewards = np.zeros(n_actions)
        self._counts = np.zeros(n_actions)
        self._action_history = []
        self._reward_history = []

    def _throttle_api_calls(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
        self._last_api_call = time.time()
    
    def _test_connection(self):
        """Test the API connection with retries."""
        for attempt in range(self.max_retries):
            try:
                self._throttle_api_calls()
                self.client.models.list()
                return  # Success
            except (APIError, APITimeoutError, RateLimitError) as e:
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to connect to OpenAI API after {self.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _get_context_prompt(self) -> str:
        """Generate a detailed context prompt for the LLM based on recent history."""
        if not self._action_history:
            return "This is the first action. You should explore different actions to learn their reward probabilities."
        
        # Get recent history
        recent_actions = self._action_history[-self._context_window:]
        recent_rewards = self._reward_history[-self._context_window:]
        
        # Calculate statistics
        action_counts = np.bincount(recent_actions, minlength=len(self._rewards))
        action_rewards = np.zeros(len(self._rewards))
        action_variances = np.zeros(len(self._rewards))
        
        # Track rewards for each action to calculate variance
        action_reward_lists = [[] for _ in range(len(self._rewards))]
        for a, r in zip(recent_actions, recent_rewards):
            action_rewards[a] += r
            action_reward_lists[a].append(r)
        
        # Calculate mean and variance for each action
        for a in range(len(self._rewards)):
            if action_counts[a] > 0:
                action_rewards[a] /= action_counts[a]
                if len(action_reward_lists[a]) > 1:
                    action_variances[a] = np.var(action_reward_lists[a])
        
        # Generate context
        context = "You are playing a multi-armed bandit game with the following actions available:\n"
        context += f"Available actions: {list(range(len(self._rewards)))}\n\n"
        
        context += "Recent history (last 20 steps):\n"
        context += "Step\tAction\tReward\n"
        context += "----------------------\n"
        for i, (a, r) in enumerate(zip(recent_actions, recent_rewards)):
            context += f"{i+1:4d}\t{a:3d}\t{r:.2f}\n"
        
        context += "\nAction statistics:\n"
        context += "Action\tCount\tAvg Reward\tVariance\n"
        context += "----------------------------------\n"
        for a in range(len(self._rewards)):
            if action_counts[a] > 0:
                context += f"{a:3d}\t{action_counts[a]:3d}\t{action_rewards[a]:.3f}\t\t{action_variances[a]:.3f}\n"
            else:
                context += f"{a:3d}\t  0\t  N/A\t\t  N/A\n"
        
        # Add exploration guidance
        context += "\nGuidance:\n"
        context += "- Your goal is to maximize the total reward over time.\n"
        context += "- Balance exploration (trying actions with high uncertainty) with exploitation (choosing actions with high average rewards).\n"
        context += "- Consider both the mean reward and the variance when making decisions.\n"
        
        return context

    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with retries and error handling.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            The response text from the LLM.
            
        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self._throttle_api_calls()
                print(f"Calling OpenAI API (attempt {attempt + 1}/{self.max_retries}) with model: {self.model}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": """You are an expert at playing multi-armed bandit games. 
                        Your goal is to maximize the cumulative reward by balancing exploration and exploitation."""},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=100,
                    timeout=self.timeout
                )
                
                response_text = response.choices[0].message.content.strip()
                print(f"API Response: {response_text[:100]}...")  # Log first 100 chars
                
                return response_text
                
            except (APIError, APITimeoutError, RateLimitError) as e:
                last_error = e
                wait_time = (2 ** attempt) + (random.random() * 0.5)  # Add jitter
                print(f"API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                last_error = e
                print(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                break
        
        # If we get here, all retries failed
        raise RuntimeError(f"Failed to get response from LLM after {self.max_retries} attempts: {str(last_error)}")

    def _load_cache(self) -> Dict[str, str]:
        """Load the cache from a JSON file."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        """Save the current cache to a JSON file."""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self._cache, f, indent=4)
        except IOError as e:
            print(f"Warning: Could not save LLM cache to {self.cache_path}: {e}")

    def _get_prompt_hash(self, prompt: str) -> str:
        """Generate a unique hash for a given prompt."""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    
    def _parse_action_from_response(self, response_text: str) -> int:
        """
        Parse the action from the LLM response.
        
        Args:
            response_text: The raw response text from the LLM.
            
        Returns:
            The parsed action index.
            
        Raises:
            ValueError: If no valid action can be parsed.
        """
        # Try to find action in various formats
        patterns = [
            r'action\s*(\d+)',  # "action 1"
            r'choose\s*action\s*(\d+)',  # "choose action 1"
            r'\b(\d+)\b',  # Just a number
            r'"action"\s*:\s*(\d+)',  # JSON-like: "action": 1
            r'"action"\s*:\s*"(\d+)"',  # JSON-like: "action": "1"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    action = int(matches[0])
                    if 0 <= action < len(self._rewards):
                        print(f"Parsed action {action} from response")
                        return action
                except (ValueError, IndexError):
                    continue
        
        raise ValueError(f"Could not parse action from response: {response_text}")
    
    def get_action(self) -> int:
        """
        Get the next action from the LLM, using caching to avoid repeated API calls.
        
        Returns:
            The action to take.
        """
        if self._rewards is None or self._counts is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")

        # Untried actions first
        untried_actions = np.where(self._counts == 0)[0]
        if len(untried_actions) > 0:
            action = untried_actions[0]
            print(f"Exploring untried action: {action}")
            self.last_llm_response = "Exploratory action (untried)"
            return action
        
        # Generate prompt with detailed context
        context = self._get_context_prompt()
        prompt = f"""You are playing a multi-armed bandit game with {len(self._rewards)} actions.
Your goal is to maximize the cumulative reward over time.

{context}

Based on the above information, which action should you choose next?

Please respond with ONLY the action number (0-{len(self._rewards)-1}) and a VERY brief explanation (1-2 sentences).
For example:
1  # This action has the highest estimated reward based on current data

Your response must be in the format: 'N # explanation' where N is the action number (0-{len(self._rewards)-1})

Your response:"""

        # Check cache or call API
        prompt_hash = self._get_prompt_hash(prompt)
        if prompt_hash in self._cache:
            print(f"Cache hit for prompt. Using cached response.")
            response_text = self._cache[prompt_hash]
        else:
            print(f"Cache miss. Calling API.")
            response_text = self._call_llm_api(prompt)
            self._cache[prompt_hash] = response_text
            self._save_cache()

        self.last_llm_response = response_text # Save for logging
        
        # Parse action
        try:
            action = self._parse_action_from_response(response_text)
            print(f"Parsed action {action} from response")
        except ValueError as e:
            print(f"Error parsing action: {e}. Choosing a random action.")
            action = np.random.choice(len(self._rewards))
        
        print(f"LLM chose action {action}")
        return action
    
    def update(self, action, reward):
        """
        Update the agent's internal state based on the action taken and reward received.
        
        Args:
            action (int): The action that was taken.
            reward (float): The reward received.
        """
        self._rewards[action] += reward
        self._counts[action] += 1
        self._action_history.append(action)
        self._reward_history.append(reward)

    @property
    def name(self):
        """Returns the name of the agent."""
        return f"{self._name}({self.model})" 
    
    