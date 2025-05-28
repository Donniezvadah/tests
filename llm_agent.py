import numpy as np
import random
import time
import os
import re
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
                 timeout: int = 30, prompt_version: str = "full"):
        """
        Initialize the LLM agent with enhanced error handling.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from llm_api.txt.
            model: The model to use. Default is "gpt-4.1-nano".
            temperature: Controls randomness in the response (0-1).
            max_retries: Maximum number of retries for API calls.
            timeout: Timeout in seconds for API calls.
            prompt_version: Which prompt version to use ("full", "no_examples", "minimal", "guided")
        """
        super().__init__("LLM")
        self.model = model
        self.temperature = max(0, min(1, temperature))  # Clamp between 0 and 1
        self.max_retries = max(1, max_retries)
        self.timeout = max(5, timeout)  # Minimum 5 second timeout
        self.prompt_version = prompt_version
        
        # Initialize state
        self._rewards = None
        self._counts = None
        self._wins = None  # Track wins for each arm
        self._action_history = []
        self._reward_history = []
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

    def init_actions(self, n_actions):
        """
        Initialize the agent's internal state.
        
        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions)
        self._rewards = np.zeros(n_actions)
        self._counts = np.zeros(n_actions)
        self._wins = np.zeros(n_actions)  # Track wins for each arm
        self._action_history = []
        self._reward_history = []

    def _get_history_string(self) -> str:
        """Generate a string representation of the arm history."""
        history = []
        for arm in range(len(self._rewards)):
            pulls = int(self._counts[arm])
            wins = int(self._wins[arm])
            losses = pulls - wins
            win_rate = wins / pulls if pulls > 0 else 0
            history.append(f"Arm {arm}: {pulls} pulls, {wins} wins, {losses} losses (win rate: {win_rate:.2f})")
        return "\n".join(history)

    def _get_prompt(self) -> str:
        """Get the appropriate prompt based on the version."""
        prompt_file = os.path.join('prompts', f'prompt_{self.prompt_version}.txt')
        try:
            with open(prompt_file, 'r') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            raise ValueError(f"Prompt file {prompt_file} not found")
        
        return prompt_template.format(
            n_actions=len(self._rewards),
            history=self._get_history_string()
        )

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
                        {"role": "system", "content": "You are an expert at playing multi-armed bandit games."},
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
            r'(\d+)\s*#',  # "1 # explanation"
            r'arm\s*(\d+)',  # "arm 1"
            r'choose\s*arm\s*(\d+)',  # "choose arm 1"
            r'\b(\d+)\b',  # Just a number
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
        Choose an action using the LLM.
        
        Returns:
            int: The index of the chosen action.
            
        Raises:
            ValueError: If the agent is not initialized or if the LLM response is invalid.
            RuntimeError: If there are issues with the LLM API calls.
        """
        if self._rewards is None or self._counts is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")
        
        # If any action hasn't been tried yet, try it first
        if np.any(self._counts == 0):
            action = int(np.argmin(self._counts))
            print(f"Exploring untried action: {action}")
            return action
        
        # Get prompt and call LLM
        prompt = self._get_prompt()
        response_text = self._call_llm_api(prompt)
        
        # Parse the action from the response
        action = self._parse_action_from_response(response_text)
        print(f"LLM chose action {action}")
        
        # Validate the action is within bounds
        if not (0 <= action < len(self._rewards)):
            raise ValueError(f"LLM returned invalid action {action}, must be between 0 and {len(self._rewards)-1}")
            
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
        if reward > 0:  # Track wins
            self._wins[action] += 1
        self._action_history.append(action)
        self._reward_history.append(reward)

    @property
    def name(self):
        """Returns the name of the agent."""
        return f"{self._name}({self.model}-{self.prompt_version})" 