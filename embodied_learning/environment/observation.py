"""
Observation processing and partial observability utilities.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""
    history_length: int = 4  # Number of frames to stack
    normalize: bool = True
    proprioception_dim: int = 10
    vision_shape: tuple = (64, 64, 3)
    touch_dim: int = 8


class PartialObservation:
    """
    Processes and manages partial observations with history.

    Maintains a history of observations and provides utilities for
    observation normalization and preprocessing.
    """

    def __init__(self, config: Optional[ObservationConfig] = None):
        self.config = config or ObservationConfig()
        self.history: List[Dict] = []

        # Running statistics for normalization
        self.proprio_mean = np.zeros(self.config.proprioception_dim)
        self.proprio_std = np.ones(self.config.proprioception_dim)
        self.update_count = 0

    def process(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Process raw observation into normalized form.

        Args:
            obs: Raw observation dictionary

        Returns:
            Processed observation with normalization and history
        """
        processed = {}

        # Process proprioception
        proprio = obs['proprioception']
        if self.config.normalize:
            proprio = self._normalize_proprioception(proprio)
        processed['proprioception'] = proprio

        # Process exteroception (vision)
        vision = obs['exteroception']
        processed['exteroception'] = self._process_vision(vision)

        # Process touch
        processed['touch'] = obs['touch']

        # Update history
        self.history.append(processed)
        if len(self.history) > self.config.history_length:
            self.history.pop(0)

        return processed

    def get_stacked_observation(self) -> Dict[str, np.ndarray]:
        """
        Get observation with temporal history stacked.

        Returns:
            Dictionary with stacked observations across time
        """
        if len(self.history) == 0:
            return {}

        # Pad history if needed
        while len(self.history) < self.config.history_length:
            self.history.insert(0, self.history[0])

        stacked = {}

        # Stack proprioception
        stacked['proprioception'] = np.concatenate([
            h['proprioception'] for h in self.history
        ])

        # Stack vision along channel dimension
        stacked['exteroception'] = np.concatenate([
            h['exteroception'] for h in self.history
        ], axis=-1)

        # Stack touch
        stacked['touch'] = np.concatenate([
            h['touch'] for h in self.history
        ])

        return stacked

    def _normalize_proprioception(self, proprio: np.ndarray) -> np.ndarray:
        """Normalize proprioception using running statistics."""
        # Update running statistics
        self.update_count += 1
        alpha = 1.0 / self.update_count
        self.proprio_mean = (1 - alpha) * self.proprio_mean + alpha * proprio
        self.proprio_std = np.sqrt(
            (1 - alpha) * self.proprio_std**2 + alpha * (proprio - self.proprio_mean)**2 + 1e-8
        )

        # Normalize
        normalized = (proprio - self.proprio_mean) / (self.proprio_std + 1e-8)
        return np.clip(normalized, -10, 10)

    def _process_vision(self, vision: np.ndarray) -> np.ndarray:
        """Process visual observation."""
        # Normalize to [0, 1]
        return vision.astype(np.float32) / 255.0

    def reset(self):
        """Reset observation history."""
        self.history = []

    def get_feature_dim(self) -> Dict[str, int]:
        """Get dimensionality of processed features."""
        return {
            'proprioception': self.config.proprioception_dim * self.config.history_length,
            'exteroception': self.config.vision_shape + (self.config.history_length,),
            'touch': self.config.touch_dim * self.config.history_length,
        }
