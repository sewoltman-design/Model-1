"""
Joint training algorithm for embodied learning.

Combines:
1. World model training (self-supervised)
2. Policy optimization (RL from task rewards)
3. Intrinsic motivation (shaped objectives)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from embodied_learning.environment.physics_env import PhysicsEnvironment
from embodied_learning.models.world_model import WorldModel, WorldModelConfig
from embodied_learning.models.policy import Policy, PolicyConfig
from embodied_learning.learning.intrinsic_motivation import (
    IntrinsicMotivation,
    IntrinsicMotivationConfig
)


@dataclass
class AgentConfig:
    """Configuration for embodied learning agent."""
    # Model configs
    world_model: WorldModelConfig = None
    policy: PolicyConfig = None
    intrinsic_motivation: IntrinsicMotivationConfig = None

    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    ppo_clip: float = 0.2

    # Intrinsic motivation
    extrinsic_reward_weight: float = 1.0
    intrinsic_reward_weight: float = 0.1

    # Update frequencies
    policy_update_freq: int = 2048  # Steps between policy updates
    world_model_update_freq: int = 1  # Update every step

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.world_model is None:
            self.world_model = WorldModelConfig()
        if self.policy is None:
            self.policy = PolicyConfig()
        if self.intrinsic_motivation is None:
            self.intrinsic_motivation = IntrinsicMotivationConfig()


class ReplayBuffer:
    """Experience replay buffer for training."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Dict):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Dict:
        """Sample batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Collate batch
        collated = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], dict):
                collated[key] = {}
                for subkey in batch[0][key].keys():
                    collated[key][subkey] = torch.stack([
                        torch.tensor(b[key][subkey]) for b in batch
                    ])
            else:
                collated[key] = torch.stack([
                    torch.tensor(b[key]) for b in batch
                ])

        return collated

    def __len__(self):
        return len(self.buffer)


class EmbodiedLearningAgent:
    """
    Embodied learning agent with joint policy and world model training.

    This is the main training algorithm that combines:
    - Self-supervised world model learning
    - RL policy optimization
    - Intrinsic motivation for exploration
    """

    def __init__(
        self,
        env: PhysicsEnvironment,
        config: Optional[AgentConfig] = None
    ):
        self.env = env
        self.config = config or AgentConfig()

        # Device
        self.device = torch.device(self.config.device)

        # Initialize models
        self.world_model = WorldModel(self.config.world_model).to(self.device)
        self.policy = Policy(self.config.policy).to(self.device)

        # Intrinsic motivation
        self.intrinsic_motivation = IntrinsicMotivation(
            self.config.intrinsic_motivation
        )

        # Optimizers
        self.world_model_optimizer = optim.Adam(
            self.world_model.parameters(),
            lr=self.config.learning_rate
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        # Training state
        self.total_steps = 0
        self.episode_count = 0

        # Rollout buffer for policy updates
        self.rollout_buffer = []

    def train(self, num_steps: int) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            num_steps: Number of environment steps to train for

        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'world_model_loss': [],
            'policy_loss': [],
            'intrinsic_rewards': [],
        }

        obs, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0

        for step in range(num_steps):
            # Select action
            action, log_prob, value = self._select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store experience
            experience = {
                'observation': obs,
                'action': action,
                'reward': reward,
                'next_observation': next_obs,
                'done': done,
                'log_prob': log_prob,
                'value': value,
            }

            self.replay_buffer.push(experience)
            self.rollout_buffer.append(experience)

            # Update world model
            if len(self.replay_buffer) >= self.config.batch_size:
                if step % self.config.world_model_update_freq == 0:
                    wm_loss = self._update_world_model()
                    metrics['world_model_loss'].append(wm_loss)

            # Update policy
            if len(self.rollout_buffer) >= self.config.policy_update_freq:
                policy_loss = self._update_policy()
                metrics['policy_loss'].append(policy_loss)
                self.rollout_buffer = []

            # Update state
            obs = next_obs
            episode_return += reward
            episode_length += 1
            self.total_steps += 1

            # Episode end
            if done:
                metrics['episode_returns'].append(episode_return)
                metrics['episode_lengths'].append(episode_length)

                obs, _ = self.env.reset()
                episode_return = 0.0
                episode_length = 0
                self.episode_count += 1

        return metrics

    def _select_action(
        self,
        obs: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action using current policy.

        Args:
            obs: Observation dictionary

        Returns:
            Tuple of (action, log_prob, value)
        """
        # Convert observation to tensors
        obs_tensors = self._obs_to_tensors(obs)

        with torch.no_grad():
            # Encode observation to latent
            latent, _, _ = self.world_model.encode(
                obs_tensors['proprioception'],
                obs_tensors['exteroception'],
                obs_tensors['touch'],
                deterministic=True
            )

            # Get action and value from policy
            action, log_prob = self.policy(latent, deterministic=False)
            value = self.policy.get_value(latent)

        return (
            action.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
            value.cpu().numpy()[0]
        )

    def _update_world_model(self) -> float:
        """Update world model using replay buffer."""
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Move to device
        obs = self._batch_to_device(batch['observation'])
        actions = batch['action'].to(self.device)
        rewards = batch['reward'].unsqueeze(1).to(self.device)
        next_obs = self._batch_to_device(batch['next_observation'])

        # Compute world model loss
        losses = self.world_model.compute_loss(obs, actions, rewards, next_obs)

        # Backward pass
        self.world_model_optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(),
            max_norm=1.0
        )
        self.world_model_optimizer.step()

        return losses['total_loss'].item()

    def _update_policy(self) -> float:
        """Update policy using PPO."""
        # Compute advantages and returns
        advantages, returns = self._compute_gae()

        # Convert rollout to tensors
        obs_list = [exp['observation'] for exp in self.rollout_buffer]
        actions = torch.tensor(
            np.stack([exp['action'] for exp in self.rollout_buffer]),
            device=self.device
        )
        old_log_probs = torch.tensor(
            np.stack([exp['log_prob'] for exp in self.rollout_buffer]),
            device=self.device
        )

        # Encode all observations
        latents = []
        for obs in obs_list:
            obs_tensors = self._obs_to_tensors(obs)
            with torch.no_grad():
                latent, _, _ = self.world_model.encode(
                    obs_tensors['proprioception'],
                    obs_tensors['exteroception'],
                    obs_tensors['touch'],
                    deterministic=True
                )
            latents.append(latent)

        latents = torch.cat(latents, dim=0)

        # PPO epochs
        total_loss = 0.0
        for _ in range(self.config.ppo_epochs):
            # Compute policy loss
            losses = self.policy.compute_policy_loss(
                latents,
                actions,
                advantages,
                old_log_probs,
                returns,
                clip_epsilon=self.config.ppo_clip
            )

            # Backward pass
            self.policy_optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm=self.config.policy.max_grad_norm
            )
            self.policy_optimizer.step()

            total_loss += losses['total_loss'].item()

        return total_loss / self.config.ppo_epochs

    def _compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Returns:
            Tuple of (advantages, returns)
        """
        # Extract data from rollout
        rewards = []
        values = []
        dones = []

        for exp in self.rollout_buffer:
            # Compute intrinsic reward
            obs = exp['observation']
            next_obs = exp['next_observation']

            obs_tensors = self._obs_to_tensors(obs)
            next_obs_tensors = self._obs_to_tensors(next_obs)
            action = torch.tensor(exp['action'], device=self.device).unsqueeze(0)

            with torch.no_grad():
                # Get prediction error
                prediction_error = self.world_model.get_prediction_error(
                    obs_tensors, action, next_obs_tensors
                )

                # Get latent states
                latent, _, _ = self.world_model.encode(
                    obs_tensors['proprioception'],
                    obs_tensors['exteroception'],
                    obs_tensors['touch'],
                    deterministic=True
                )

                next_latent, _, _ = self.world_model.encode(
                    next_obs_tensors['proprioception'],
                    next_obs_tensors['exteroception'],
                    next_obs_tensors['touch'],
                    deterministic=True
                )

                # Compute intrinsic reward
                intrinsic_reward, _ = self.intrinsic_motivation.compute_intrinsic_reward(
                    prediction_error, latent, next_latent
                )

            # Combined reward
            extrinsic = exp['reward'] * self.config.extrinsic_reward_weight
            intrinsic = intrinsic_reward.cpu().numpy()[0] * self.config.intrinsic_reward_weight
            total_reward = extrinsic + intrinsic

            rewards.append(total_reward)
            values.append(exp['value'])
            dones.append(exp['done'])

        # Compute GAE
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = advantages + torch.tensor(values, device=self.device, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _obs_to_tensors(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation dict to tensors."""
        return {
            'proprioception': torch.tensor(
                obs['proprioception'], device=self.device, dtype=torch.float32
            ).unsqueeze(0),
            'exteroception': torch.tensor(
                obs['exteroception'], device=self.device, dtype=torch.float32
            ).unsqueeze(0),
            'touch': torch.tensor(
                obs['touch'], device=self.device, dtype=torch.float32
            ).unsqueeze(0),
        }

    def _batch_to_device(self, batch_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert batched observation to device."""
        # batch_obs is a stacked tensor from replay buffer
        # Need to handle the dict structure
        if isinstance(batch_obs, dict):
            return {k: v.to(self.device) for k, v in batch_obs.items()}
        else:
            # If it's already been processed, just move to device
            return batch_obs.to(self.device)

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'world_model': self.world_model.state_dict(),
            'policy': self.policy.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
