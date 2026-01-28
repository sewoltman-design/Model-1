"""
World model for self-supervised learning from interaction.

The world model learns to predict future observations and rewards,
providing a predictive representation of the environment dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from embodied_learning.models.networks import Encoder, Decoder, RecurrentStateModel


@dataclass
class WorldModelConfig:
    """Configuration for world model."""
    latent_dim: int = 256
    hidden_dim: int = 512
    action_dim: int = 4
    proprio_dim: int = 10
    vision_shape: Tuple[int, int, int] = (64, 64, 3)
    touch_dim: int = 8
    learning_rate: float = 3e-4
    kl_weight: float = 0.1
    reconstruction_weight: float = 1.0
    reward_weight: float = 1.0


class WorldModel(nn.Module):
    """
    Predictive world model trained self-supervised.

    The model learns to:
    1. Encode observations into latent representations
    2. Predict future latent states given actions
    3. Decode latent states back to observations
    4. Predict rewards

    This provides intrinsic objectives for exploration and planning.
    """

    def __init__(self, config: Optional[WorldModelConfig] = None):
        super().__init__()
        self.config = config or WorldModelConfig()

        # Encoder: obs -> latent
        self.encoder = Encoder(
            proprio_dim=self.config.proprio_dim,
            vision_shape=self.config.vision_shape,
            touch_dim=self.config.touch_dim,
            latent_dim=self.config.latent_dim
        )

        # Recurrent state space model: (latent, action) -> next_latent
        self.dynamics = RecurrentStateModel(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        )

        # Decoder: latent -> obs
        self.decoder = Decoder(
            latent_dim=self.config.latent_dim,
            proprio_dim=self.config.proprio_dim,
            vision_shape=self.config.vision_shape,
            touch_dim=self.config.touch_dim
        )

        # Variational components (for stochastic latent)
        self.latent_mean = nn.Linear(self.config.latent_dim, self.config.latent_dim)
        self.latent_logstd = nn.Linear(self.config.latent_dim, self.config.latent_dim)

    def encode(
        self,
        proprio: torch.Tensor,
        vision: torch.Tensor,
        touch: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observations to latent state.

        Args:
            proprio: Proprioceptive input [batch, proprio_dim]
            vision: Visual input [batch, H, W, C]
            touch: Touch input [batch, touch_dim]
            deterministic: If True, return mean without sampling

        Returns:
            Tuple of (latent, mean, logstd)
        """
        # Encode to deterministic features
        encoded = self.encoder(proprio, vision, touch)

        # Variational parameters
        mean = self.latent_mean(encoded)
        logstd = self.latent_logstd(encoded)
        logstd = torch.clamp(logstd, -10, 2)  # Constrain std

        # Sample latent (or use mean if deterministic)
        if deterministic:
            latent = mean
        else:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            latent = mean + eps * std

        return latent, mean, logstd

    def decode(
        self,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent state to observations.

        Args:
            latent: Latent state [batch, latent_dim]

        Returns:
            Tuple of (proprio, vision, touch) predictions
        """
        return self.decoder(latent)

    def predict_next(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next latent state and reward.

        Args:
            latent: Current latent state [batch, latent_dim]
            action: Action taken [batch, action_dim]
            hidden: RNN hidden state

        Returns:
            Tuple of (next_latent, reward_pred, next_hidden)
        """
        return self.dynamics(latent, action, hidden)

    def imagine_trajectory(
        self,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
        horizon: int
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory by rolling out the world model.

        Args:
            initial_latent: Starting latent state [batch, latent_dim]
            actions: Sequence of actions [batch, horizon, action_dim]
            horizon: Number of steps to imagine

        Returns:
            Dictionary containing imagined trajectory
        """
        batch_size = initial_latent.shape[0]
        device = initial_latent.device

        # Storage for trajectory
        latents = [initial_latent]
        rewards = []
        hidden = self.dynamics.init_hidden(batch_size, device)

        # Roll out dynamics
        current_latent = initial_latent
        for t in range(horizon):
            action_t = actions[:, t]
            next_latent, reward_pred, hidden = self.predict_next(
                current_latent, action_t, hidden
            )
            latents.append(next_latent)
            rewards.append(reward_pred)
            current_latent = next_latent

        # Stack results
        latents = torch.stack(latents, dim=1)  # [batch, horizon+1, latent_dim]
        rewards = torch.stack(rewards, dim=1)  # [batch, horizon, 1]

        return {
            'latents': latents,
            'rewards': rewards,
        }

    def compute_loss(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute world model training loss.

        Args:
            observations: Current observations dict
            actions: Actions taken [batch, action_dim]
            rewards: Actual rewards [batch, 1]
            next_observations: Next observations dict

        Returns:
            Dictionary of losses
        """
        # Encode current and next observations
        latent, mean, logstd = self.encode(
            observations['proprioception'],
            observations['exteroception'],
            observations['touch']
        )

        next_latent_target, next_mean, next_logstd = self.encode(
            next_observations['proprioception'],
            next_observations['exteroception'],
            next_observations['touch']
        )

        # Predict next state and reward
        next_latent_pred, reward_pred, _ = self.predict_next(latent, actions)

        # Reconstruction loss (decode current state)
        proprio_pred, vision_pred, touch_pred = self.decode(latent)

        proprio_loss = F.mse_loss(
            proprio_pred,
            observations['proprioception']
        )

        # Vision loss (MSE on normalized images)
        vision_target = observations['exteroception'].float() / 255.0
        vision_loss = F.mse_loss(vision_pred, vision_target)

        touch_loss = F.mse_loss(touch_pred, observations['touch'])

        reconstruction_loss = proprio_loss + vision_loss + touch_loss

        # Dynamics loss (predict next latent)
        dynamics_loss = F.mse_loss(next_latent_pred, next_latent_target.detach())

        # Reward prediction loss
        reward_loss = F.mse_loss(reward_pred, rewards)

        # KL divergence for regularization
        kl_loss = self._compute_kl_divergence(mean, logstd)

        # Total loss
        total_loss = (
            self.config.reconstruction_weight * reconstruction_loss +
            dynamics_loss +
            self.config.reward_weight * reward_loss +
            self.config.kl_weight * kl_loss
        )

        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'dynamics_loss': dynamics_loss,
            'reward_loss': reward_loss,
            'kl_loss': kl_loss,
            'proprio_loss': proprio_loss,
            'vision_loss': vision_loss,
            'touch_loss': touch_loss,
        }

    def _compute_kl_divergence(
        self,
        mean: torch.Tensor,
        logstd: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between latent distribution and standard normal.

        Args:
            mean: Latent mean
            logstd: Latent log standard deviation

        Returns:
            KL divergence
        """
        std = torch.exp(logstd)
        kl = -0.5 * torch.sum(1 + 2*logstd - mean.pow(2) - std.pow(2), dim=1)
        return kl.mean()

    def get_prediction_error(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute prediction error for intrinsic motivation.

        This measures how surprising the transition was, which can be
        used as a novelty bonus for exploration.

        Args:
            observations: Current observations
            actions: Actions taken
            next_observations: Next observations

        Returns:
            Prediction error (surprise) for each transition
        """
        with torch.no_grad():
            # Encode states
            latent, _, _ = self.encode(
                observations['proprioception'],
                observations['exteroception'],
                observations['touch'],
                deterministic=True
            )

            next_latent_target, _, _ = self.encode(
                next_observations['proprioception'],
                next_observations['exteroception'],
                next_observations['touch'],
                deterministic=True
            )

            # Predict next state
            next_latent_pred, _, _ = self.predict_next(latent, actions)

            # Compute prediction error
            prediction_error = torch.norm(
                next_latent_pred - next_latent_target,
                dim=1
            )

        return prediction_error
