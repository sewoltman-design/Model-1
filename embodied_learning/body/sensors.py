"""
Sensory systems for proprioception and exteroception.
"""

import numpy as np
from typing import Dict, Optional


class ProprioceptiveSensor:
    """
    Proprioceptive sensor for internal state.

    Senses the body's own state: position, velocity, joint angles,
    muscle activation, etc. This is analogous to biological proprioception.
    """

    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level
        self.previous_readings = None

    def sense(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray,
        actuator_states: np.ndarray
    ) -> np.ndarray:
        """
        Generate proprioceptive observation.

        Args:
            position: 3D position
            velocity: 3D velocity
            orientation: Quaternion orientation
            angular_velocity: Angular velocity
            actuator_states: Current actuator states

        Returns:
            Proprioceptive feature vector
        """
        # Concatenate all proprioceptive information
        proprio = np.concatenate([
            position,
            velocity,
            orientation,
            angular_velocity,
            actuator_states[:4] if len(actuator_states) >= 4 else np.zeros(4),
        ])

        # Add sensor noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, proprio.shape)
            proprio = proprio + noise

        self.previous_readings = proprio
        return proprio.astype(np.float32)

    def reset(self):
        """Reset sensor state."""
        self.previous_readings = None


class ExteroceptiveSensor:
    """
    Exteroceptive sensor for external world.

    Senses external stimuli: vision, touch, proximity, etc.
    This is analogous to biological exteroception.
    """

    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level

    def sense(self, env_observation: Dict) -> Dict:
        """
        Process exteroceptive observations.

        Args:
            env_observation: Raw observation from environment

        Returns:
            Processed exteroceptive features
        """
        extero = {}

        # Vision (with noise)
        if 'exteroception' in env_observation:
            vision = env_observation['exteroception']
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level * 10, vision.shape)
                vision = np.clip(vision + noise, 0, 255).astype(np.uint8)
            extero['vision'] = vision

        # Touch/proximity sensors
        if 'touch' in env_observation:
            touch = env_observation['touch']
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, touch.shape)
                touch = np.clip(touch + noise, 0, 1)
            extero['touch'] = touch.astype(np.float32)

        return extero

    def reset(self):
        """Reset sensor state."""
        pass
