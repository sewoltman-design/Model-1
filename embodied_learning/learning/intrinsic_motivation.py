"""
Intrinsic motivation mechanisms for exploration and learning.

Implements:
1. Prediction-error novelty: Reward surprise/unpredictability
2. Goal-conditioned imagination: Reward reaching imagined goals
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IntrinsicMotivationConfig:
    """Configuration for intrinsic motivation."""
    prediction_error_weight: float = 0.1
    goal_imagination_weight: float = 0.05
    normalize_rewards: bool = True
    running_mean_window: int = 1000


class PredictionErrorNovelty:
    """
    Prediction-error based novelty reward.

    Rewards the agent for visiting states that are surprising to the
    world model, encouraging exploration of novel states.
    """

    def __init__(
        self,
        weight: float = 0.1,
        normalize: bool = True,
        running_mean_window: int = 1000
    ):
        self.weight = weight
        self.normalize = normalize
        self.running_mean_window = running_mean_window

        # Running statistics for normalization
        self.error_history = []
        self.error_mean = 0.0
        self.error_std = 1.0

    def compute_reward(
        self,
        prediction_errors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute novelty reward from prediction errors.

        Args:
            prediction_errors: Prediction error magnitudes [batch]

        Returns:
            Intrinsic novelty rewards [batch]
        """
        errors = prediction_errors.detach().cpu().numpy()

        # Update running statistics
        self.error_history.extend(errors.tolist())
        if len(self.error_history) > self.running_mean_window:
            self.error_history = self.error_history[-self.running_mean_window:]

        self.error_mean = np.mean(self.error_history)
        self.error_std = np.std(self.error_history) + 1e-8

        # Normalize if requested
        if self.normalize:
            normalized_errors = (errors - self.error_mean) / self.error_std
            rewards = normalized_errors
        else:
            rewards = errors

        # Scale by weight
        rewards = rewards * self.weight

        return torch.tensor(rewards, device=prediction_errors.device, dtype=prediction_errors.dtype)

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about prediction errors."""
        return {
            'prediction_error_mean': self.error_mean,
            'prediction_error_std': self.error_std,
            'history_length': len(self.error_history),
        }


class GoalImaginationReward:
    """
    Goal-conditioned imagination reward.

    Rewards progress towards imagined goal states, encouraging
    directed exploration and planning.
    """

    def __init__(
        self,
        weight: float = 0.05,
        goal_dim: int = 256,
        horizon: int = 10
    ):
        self.weight = weight
        self.goal_dim = goal_dim
        self.horizon = horizon

        # Goal sampling
        self.goal_buffer = []
        self.max_buffer_size = 10000

    def sample_goals(
        self,
        current_latent: torch.Tensor,
        num_goals: int = 1
    ) -> torch.Tensor:
        """
        Sample goal states for imagination.

        Args:
            current_latent: Current latent state [batch, latent_dim]
            num_goals: Number of goals to sample

        Returns:
            Goal latent states [batch, num_goals, latent_dim]
        """
        batch_size = current_latent.shape[0]
        device = current_latent.device

        if len(self.goal_buffer) < num_goals:
            # Sample random goals if buffer not full
            goals = torch.randn(
                batch_size, num_goals, self.goal_dim,
                device=device
            )
        else:
            # Sample from buffer
            indices = np.random.choice(len(self.goal_buffer), size=num_goals, replace=False)
            goals = [self.goal_buffer[i] for i in indices]
            goals = torch.stack(goals, dim=0).unsqueeze(0)  # [1, num_goals, dim]
            goals = goals.repeat(batch_size, 1, 1).to(device)

        return goals

    def add_to_buffer(self, latent_states: torch.Tensor):
        """
        Add latent states to goal buffer.

        Args:
            latent_states: Latent states to add [batch, latent_dim]
        """
        states = latent_states.detach().cpu()
        for state in states:
            self.goal_buffer.append(state)

        # Maintain buffer size
        if len(self.goal_buffer) > self.max_buffer_size:
            self.goal_buffer = self.goal_buffer[-self.max_buffer_size:]

    def compute_reward(
        self,
        current_latent: torch.Tensor,
        next_latent: torch.Tensor,
        goals: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute goal-reaching reward.

        Rewards the agent for moving closer to sampled goal states.

        Args:
            current_latent: Current state [batch, latent_dim]
            next_latent: Next state [batch, latent_dim]
            goals: Goal states [batch, num_goals, latent_dim]

        Returns:
            Goal-reaching rewards [batch]
        """
        # Compute distance to goals
        # current_dist: [batch, num_goals]
        current_dist = torch.norm(
            current_latent.unsqueeze(1) - goals,
            dim=2
        )

        # next_dist: [batch, num_goals]
        next_dist = torch.norm(
            next_latent.unsqueeze(1) - goals,
            dim=2
        )

        # Reward for getting closer to any goal
        progress = current_dist - next_dist  # Positive if getting closer
        max_progress = progress.max(dim=1)[0]  # Best goal progress

        # Scale by weight
        rewards = max_progress * self.weight

        return rewards

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about goal buffer."""
        return {
            'goal_buffer_size': len(self.goal_buffer),
            'goal_buffer_capacity': self.max_buffer_size,
        }


class IntrinsicMotivation:
    """
    Combined intrinsic motivation system.

    Combines prediction-error novelty and goal-conditioned imagination
    to provide shaped intrinsic rewards.
    """

    def __init__(self, config: Optional[IntrinsicMotivationConfig] = None):
        self.config = config or IntrinsicMotivationConfig()

        # Initialize components
        self.prediction_error = PredictionErrorNovelty(
            weight=self.config.prediction_error_weight,
            normalize=self.config.normalize_rewards,
            running_mean_window=self.config.running_mean_window
        )

        self.goal_imagination = GoalImaginationReward(
            weight=self.config.goal_imagination_weight
        )

    def compute_intrinsic_reward(
        self,
        prediction_errors: torch.Tensor,
        current_latent: torch.Tensor,
        next_latent: torch.Tensor,
        goals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined intrinsic reward.

        Args:
            prediction_errors: World model prediction errors [batch]
            current_latent: Current latent state [batch, latent_dim]
            next_latent: Next latent state [batch, latent_dim]
            goals: Optional goal states [batch, num_goals, latent_dim]

        Returns:
            Tuple of (intrinsic_rewards, info_dict)
        """
        # Prediction error novelty
        novelty_reward = self.prediction_error.compute_reward(prediction_errors)

        # Goal imagination reward
        if goals is None:
            goals = self.goal_imagination.sample_goals(current_latent)

        goal_reward = self.goal_imagination.compute_reward(
            current_latent, next_latent, goals
        )

        # Combine rewards
        intrinsic_reward = novelty_reward + goal_reward

        # Add states to goal buffer
        self.goal_imagination.add_to_buffer(next_latent)

        info = {
            'novelty_reward': novelty_reward,
            'goal_reward': goal_reward,
            'intrinsic_reward': intrinsic_reward,
        }

        return intrinsic_reward, info

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics from all intrinsic motivation components."""
        stats = {}
        stats.update(self.prediction_error.get_statistics())
        stats.update(self.goal_imagination.get_statistics())
        return stats
