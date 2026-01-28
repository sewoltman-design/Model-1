"""
Policy network for reinforcement learning.

The policy maps observations (or latent states) to actions,
optimized via RL from task rewards and intrinsic motivation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

from embodied_learning.models.networks import MLPNetwork


@dataclass
class PolicyConfig:
    """Configuration for policy."""
    latent_dim: int = 256
    action_dim: int = 4
    hidden_dims: Tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5


class Policy(nn.Module):
    """
    Actor-critic policy for RL.

    Outputs both actions (actor) and value estimates (critic).
    Uses a stochastic policy with Gaussian actions.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        super().__init__()
        self.config = config or PolicyConfig()

        # Actor network: latent -> action distribution
        self.actor_mean = MLPNetwork(
            input_dim=self.config.latent_dim,
            output_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            activation='relu'
        )

        # Action log std (learnable parameter)
        self.actor_logstd = nn.Parameter(
            torch.zeros(self.config.action_dim)
        )

        # Critic network: latent -> value
        self.critic = MLPNetwork(
            input_dim=self.config.latent_dim,
            output_dim=1,
            hidden_dims=self.config.hidden_dims,
            activation='relu'
        )

    def forward(
        self,
        latent: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy.

        Args:
            latent: Latent state [batch, latent_dim]
            deterministic: If True, return mean action

        Returns:
            Tuple of (action, log_prob)
        """
        # Actor
        action_mean = self.actor_mean(latent)
        action_mean = torch.tanh(action_mean)  # Bounded actions

        if deterministic:
            return action_mean, torch.zeros_like(action_mean[:, 0])

        # Sample action
        action_std = torch.exp(self.actor_logstd)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()
        action = torch.clamp(action, -1, 1)  # Clip to action space

        # Log probability
        log_prob = action_dist.log_prob(action).sum(dim=1)

        return action, log_prob

    def evaluate_actions(
        self,
        latent: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO-style updates.

        Args:
            latent: Latent state [batch, latent_dim]
            actions: Actions taken [batch, action_dim]

        Returns:
            Tuple of (log_prob, value, entropy)
        """
        # Actor
        action_mean = self.actor_mean(latent)
        action_mean = torch.tanh(action_mean)

        action_std = torch.exp(self.actor_logstd)
        action_dist = torch.distributions.Normal(action_mean, action_std)

        # Log probability of taken actions
        log_prob = action_dist.log_prob(actions).sum(dim=1)

        # Entropy for exploration
        entropy = action_dist.entropy().sum(dim=1)

        # Value estimate
        value = self.critic(latent).squeeze(-1)

        return log_prob, value, entropy

    def get_value(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate.

        Args:
            latent: Latent state [batch, latent_dim]

        Returns:
            Value estimate [batch]
        """
        return self.critic(latent).squeeze(-1)

    def act(
        self,
        latent: torch.Tensor,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Sample action (for environment interaction).

        Args:
            latent: Latent state tensor
            deterministic: If True, return mean action

        Returns:
            Action as numpy array
        """
        with torch.no_grad():
            action, _ = self.forward(latent, deterministic)
            return action.cpu().numpy()

    def compute_policy_loss(
        self,
        latent: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        clip_epsilon: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO policy loss.

        Args:
            latent: Latent states [batch, latent_dim]
            actions: Actions taken [batch, action_dim]
            advantages: Advantage estimates [batch]
            old_log_probs: Old policy log probs [batch]
            returns: Return targets [batch]
            clip_epsilon: PPO clipping parameter

        Returns:
            Dictionary of losses
        """
        # Evaluate actions under current policy
        log_probs, values, entropy = self.evaluate_actions(latent, actions)

        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus for exploration
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss +
            self.config.entropy_coef * entropy_loss
        )

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'entropy': entropy.mean(),
        }


class GoalConditionedPolicy(Policy):
    """
    Goal-conditioned policy for imagination-based planning.

    Extends the base policy to condition on goal states for
    hierarchical/goal-based control.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        super().__init__(config)

        # Goal-conditioned actor (takes latent + goal)
        self.goal_actor_mean = MLPNetwork(
            input_dim=self.config.latent_dim * 2,  # state + goal
            output_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            activation='relu'
        )

        # Goal-conditioned critic
        self.goal_critic = MLPNetwork(
            input_dim=self.config.latent_dim * 2,
            output_dim=1,
            hidden_dims=self.config.hidden_dims,
            activation='relu'
        )

    def forward_goal_conditioned(
        self,
        latent: torch.Tensor,
        goal: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with goal conditioning.

        Args:
            latent: Current latent state [batch, latent_dim]
            goal: Goal latent state [batch, latent_dim]
            deterministic: If True, return mean action

        Returns:
            Tuple of (action, log_prob)
        """
        # Concatenate state and goal
        state_goal = torch.cat([latent, goal], dim=1)

        # Actor
        action_mean = self.goal_actor_mean(state_goal)
        action_mean = torch.tanh(action_mean)

        if deterministic:
            return action_mean, torch.zeros_like(action_mean[:, 0])

        # Sample action
        action_std = torch.exp(self.actor_logstd)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()
        action = torch.clamp(action, -1, 1)

        log_prob = action_dist.log_prob(action).sum(dim=1)

        return action, log_prob

    def get_value_goal_conditioned(
        self,
        latent: torch.Tensor,
        goal: torch.Tensor
    ) -> torch.Tensor:
        """Get value estimate for goal-conditioned policy."""
        state_goal = torch.cat([latent, goal], dim=1)
        return self.goal_critic(state_goal).squeeze(-1)
