"""
Actuated body with sensorimotor capabilities.

This module defines the agent's body, including actuators and sensors
for interaction with the environment.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from embodied_learning.body.sensors import ProprioceptiveSensor, ExteroceptiveSensor


@dataclass
class BodyConfig:
    """Configuration for the actuated body."""
    num_actuators: int = 4
    actuator_strength: float = 10.0
    mass: float = 1.0
    height: float = 1.0
    sensor_noise: float = 0.01


class ActuatedBody:
    """
    Actuated body with proprioceptive and exteroceptive sensors.

    The body represents the agent's physical embodiment in the world,
    including:
    - Actuators: Motors/muscles that apply forces
    - Proprioception: Internal state sensing (position, velocity, etc.)
    - Exteroception: External sensing (vision, touch, etc.)
    """

    def __init__(self, config: Optional[BodyConfig] = None):
        self.config = config or BodyConfig()

        # Physical properties
        self.mass = self.config.mass
        self.height = self.config.height

        # Actuators
        self.num_actuators = self.config.num_actuators
        self.actuator_commands = np.zeros(self.num_actuators)
        self.actuator_states = np.zeros(self.num_actuators)

        # Sensors
        self.proprioceptive_sensor = ProprioceptiveSensor(self.config.sensor_noise)
        self.exteroceptive_sensor = ExteroceptiveSensor(self.config.sensor_noise)

        # Body state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([1, 0, 0, 0])  # Quaternion
        self.angular_velocity = np.zeros(3)

    def set_commands(self, commands: np.ndarray):
        """
        Set actuator commands.

        Args:
            commands: Array of actuator commands in [-1, 1]
        """
        self.actuator_commands = np.clip(commands, -1, 1)

    def update_state(self, env_state: Dict):
        """
        Update body state from environment.

        Args:
            env_state: Current environment state
        """
        if 'agent' in env_state:
            agent = env_state['agent']
            self.position = agent['position']
            self.velocity = agent['velocity']
            self.orientation = agent['orientation']
            self.angular_velocity = agent['angular_velocity']

    def get_proprioception(self) -> np.ndarray:
        """
        Get proprioceptive observation (internal state).

        Returns:
            Proprioceptive features: [position, velocity, orientation, etc.]
        """
        return self.proprioceptive_sensor.sense(
            position=self.position,
            velocity=self.velocity,
            orientation=self.orientation,
            angular_velocity=self.angular_velocity,
            actuator_states=self.actuator_states
        )

    def get_exteroception(self, env_observation: Dict) -> Dict:
        """
        Get exteroceptive observation (external sensing).

        Args:
            env_observation: Raw observation from environment

        Returns:
            Processed exteroceptive features
        """
        return self.exteroceptive_sensor.sense(env_observation)

    def apply_actuators(self) -> np.ndarray:
        """
        Convert actuator commands to forces/torques.

        Returns:
            Action array to apply to environment
        """
        # Update actuator states (with delay/dynamics)
        alpha = 0.3  # Actuator response rate
        self.actuator_states = (
            (1 - alpha) * self.actuator_states +
            alpha * self.actuator_commands
        )

        # Scale by actuator strength
        actions = self.actuator_states * self.config.actuator_strength

        return actions

    def get_state_dict(self) -> Dict:
        """Get complete body state as dictionary."""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'actuator_commands': self.actuator_commands.copy(),
            'actuator_states': self.actuator_states.copy(),
        }

    def reset(self):
        """Reset body to initial state."""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([1, 0, 0, 0])
        self.angular_velocity = np.zeros(3)
        self.actuator_commands = np.zeros(self.num_actuators)
        self.actuator_states = np.zeros(self.num_actuators)
        self.proprioceptive_sensor.reset()
        self.exteroceptive_sensor.reset()
