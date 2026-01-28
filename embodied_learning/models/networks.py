"""
Neural network building blocks for embodied learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Encoder(nn.Module):
    """
    Observation encoder that processes multimodal sensory input.

    Encodes proprioception, vision, and touch into a latent representation.
    """

    def __init__(
        self,
        proprio_dim: int = 10,
        vision_shape: Tuple[int, int, int] = (64, 64, 3),
        touch_dim: int = 8,
        latent_dim: int = 256
    ):
        super().__init__()

        self.proprio_dim = proprio_dim
        self.vision_shape = vision_shape
        self.touch_dim = touch_dim
        self.latent_dim = latent_dim

        # Proprioception encoder
        self.proprio_net = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Vision encoder (CNN)
        self.vision_net = nn.Sequential(
            nn.Conv2d(vision_shape[2], 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate vision feature size
        with torch.no_grad():
            dummy_vision = torch.zeros(1, *vision_shape).permute(0, 3, 1, 2)
            vision_features = self.vision_net(dummy_vision)
            self.vision_feature_dim = vision_features.shape[1]

        # Touch encoder
        self.touch_net = nn.Sequential(
            nn.Linear(touch_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusion layer
        fusion_input_dim = 128 + self.vision_feature_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(
        self,
        proprio: torch.Tensor,
        vision: torch.Tensor,
        touch: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode multimodal observations.

        Args:
            proprio: [batch, proprio_dim]
            vision: [batch, H, W, C]
            touch: [batch, touch_dim]

        Returns:
            Latent encoding: [batch, latent_dim]
        """
        # Encode each modality
        proprio_feat = self.proprio_net(proprio)

        # Vision: permute to [batch, C, H, W]
        vision = vision.permute(0, 3, 1, 2)
        vision_feat = self.vision_net(vision)

        touch_feat = self.touch_net(touch)

        # Fuse all modalities
        combined = torch.cat([proprio_feat, vision_feat, touch_feat], dim=1)
        latent = self.fusion(combined)

        return latent


class Decoder(nn.Module):
    """
    Observation decoder for reconstructing observations from latent state.

    Used by the world model to predict future observations.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        proprio_dim: int = 10,
        vision_shape: Tuple[int, int, int] = (64, 64, 3),
        touch_dim: int = 8
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        self.vision_shape = vision_shape
        self.touch_dim = touch_dim

        # Proprioception decoder
        self.proprio_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, proprio_dim)
        )

        # Vision decoder (transposed CNN)
        self.vision_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.vision_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, vision_shape[2], 4, stride=2),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Touch decoder
        self.touch_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, touch_dim),
            nn.Sigmoid()
        )

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent state to observations.

        Args:
            latent: [batch, latent_dim]

        Returns:
            Tuple of (proprio, vision, touch) predictions
        """
        # Decode proprio
        proprio_pred = self.proprio_decoder(latent)

        # Decode vision
        vision_feat = self.vision_fc(latent)
        vision_feat = vision_feat.view(-1, 256, 4, 4)
        vision_pred = self.vision_decoder(vision_feat)
        vision_pred = vision_pred.permute(0, 2, 3, 1)  # Back to [B, H, W, C]

        # Decode touch
        touch_pred = self.touch_decoder(latent)

        return proprio_pred, vision_pred, touch_pred


class RecurrentStateModel(nn.Module):
    """
    Recurrent state space model for world dynamics.

    Predicts next latent state given current state and action.
    Uses GRU for temporal modeling.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 4,
        hidden_dim: int = 512
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # State-action fusion
        self.input_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Recurrent dynamics
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Reward predictor
        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward.

        Args:
            state: Current latent state [batch, latent_dim]
            action: Action taken [batch, action_dim]
            hidden: GRU hidden state [batch, hidden_dim]

        Returns:
            Tuple of (next_state, predicted_reward, next_hidden)
        """
        # Combine state and action
        state_action = torch.cat([state, action], dim=1)
        x = self.input_net(state_action)

        # Recurrent update
        if hidden is None:
            hidden = torch.zeros(
                state.shape[0], self.hidden_dim,
                device=state.device, dtype=state.dtype
            )

        next_hidden = self.gru(x, hidden)

        # Predict next state
        next_state = self.output_net(next_hidden)

        # Predict reward
        reward_pred = self.reward_net(next_hidden)

        return next_state, reward_pred, next_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class MLPNetwork(nn.Module):
    """General purpose MLP network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = 'relu'
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
