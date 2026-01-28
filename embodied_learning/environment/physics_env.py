"""
Physics-based environment with partial observability.

This module implements a physics simulation environment where agents
interact with a world through limited sensory observations.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Configuration for the physics environment."""
    dt: float = 0.02  # Simulation timestep
    gravity: float = -9.81  # Gravity constant
    friction: float = 0.5  # Surface friction
    max_episode_steps: int = 1000
    observation_noise: float = 0.01  # Sensor noise level
    partial_observability: bool = True
    vision_range: float = 10.0  # Visual range for partial observability
    vision_angle: float = 120.0  # Field of view in degrees


class PhysicsEnvironment(gym.Env):
    """
    Physics-based environment with partial observability.

    The environment simulates a physical world where agents can interact
    through limited sensory information. Observations are noisy and partial,
    requiring agents to build internal world models.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        super().__init__()
        self.config = config or EnvironmentConfig()

        # Environment state
        self.time = 0.0
        self.step_count = 0
        self.done = False

        # World state (positions, velocities, objects)
        self.world_state = {}
        self.objects = []
        self.agent_state = None

        # Action and observation spaces
        # Actions: [forward, lateral, rotation, jump/interact]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observations: proprioception + exteroception (vision, touch, etc.)
        self.observation_space = gym.spaces.Dict({
            'proprioception': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
            ),
            'exteroception': gym.spaces.Box(
                low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
            ),
            'touch': gym.spaces.Box(
                low=0, high=1, shape=(8,), dtype=np.float32
            )
        })

        self._initialize_world()

    def _initialize_world(self):
        """Initialize the physical world state."""
        # Ground plane
        self.world_state['ground'] = {
            'position': np.array([0, 0, 0]),
            'normal': np.array([0, 0, 1])
        }

        # Scattered objects in the environment
        np.random.seed(None)  # Different each reset
        num_objects = np.random.randint(5, 15)

        self.objects = []
        for _ in range(num_objects):
            obj = {
                'position': np.random.uniform(-20, 20, size=3),
                'velocity': np.zeros(3),
                'type': np.random.choice(['box', 'sphere', 'cylinder']),
                'size': np.random.uniform(0.5, 2.0),
                'color': np.random.uniform(0, 1, size=3)
            }
            obj['position'][2] = abs(obj['position'][2])  # Keep above ground
            self.objects.append(obj)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.time = 0.0
        self.step_count = 0
        self.done = False

        # Reset agent state
        self.agent_state = {
            'position': np.array([0.0, 0.0, 1.0]),
            'velocity': np.zeros(3),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # Quaternion
            'angular_velocity': np.zeros(3),
        }

        self._initialize_world()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one timestep of the environment.

        Args:
            action: Control commands [forward, lateral, rotation, jump]

        Returns:
            observation: Partial observation of the world state
            reward: Task reward signal
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information dictionary
        """
        action = np.clip(action, -1.0, 1.0)

        # Apply physics update
        self._apply_action(action)
        self._physics_step()

        # Update time
        self.time += self.config.dt
        self.step_count += 1

        # Get observation with partial observability
        observation = self._get_observation()

        # Compute reward
        reward = self._compute_reward(action)

        # Check termination
        terminated = self._check_termination()
        truncated = self.step_count >= self.config.max_episode_steps

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray):
        """Apply agent actions to update forces/torques."""
        # Extract action components
        forward = action[0]
        lateral = action[1]
        rotation = action[2]
        jump = action[3]

        # Apply forces in agent's local frame
        orientation_vec = self._quaternion_to_direction(self.agent_state['orientation'])
        right_vec = np.cross(orientation_vec, np.array([0, 0, 1]))

        force = forward * orientation_vec + lateral * right_vec
        force = force * 10.0  # Force magnitude

        # Apply to velocity (simplified dynamics)
        self.agent_state['velocity'][:2] += force[:2] * self.config.dt

        # Apply rotation
        self.agent_state['angular_velocity'][2] = rotation * 3.0

        # Jump (vertical impulse)
        if jump > 0.5 and self.agent_state['position'][2] <= 1.0:
            self.agent_state['velocity'][2] = 5.0

    def _physics_step(self):
        """Update physics simulation."""
        # Update agent position and orientation
        self.agent_state['position'] += self.agent_state['velocity'] * self.config.dt

        # Apply gravity
        self.agent_state['velocity'][2] += self.config.gravity * self.config.dt

        # Apply friction
        friction_coeff = self.config.friction
        if self.agent_state['position'][2] <= 1.0:  # On ground
            self.agent_state['velocity'][:2] *= (1.0 - friction_coeff * self.config.dt)

        # Ground collision
        if self.agent_state['position'][2] < 1.0:
            self.agent_state['position'][2] = 1.0
            self.agent_state['velocity'][2] = max(0, self.agent_state['velocity'][2])

        # Update orientation from angular velocity
        angle = np.linalg.norm(self.agent_state['angular_velocity']) * self.config.dt
        if angle > 0:
            axis = self.agent_state['angular_velocity'] / np.linalg.norm(self.agent_state['angular_velocity'])
            rotation_quat = self._axis_angle_to_quaternion(axis, angle)
            self.agent_state['orientation'] = self._quaternion_multiply(
                rotation_quat, self.agent_state['orientation']
            )

        # Update object positions (simple dynamics)
        for obj in self.objects:
            obj['velocity'][2] += self.config.gravity * self.config.dt
            obj['position'] += obj['velocity'] * self.config.dt

            # Ground collision for objects
            if obj['position'][2] < obj['size'] / 2:
                obj['position'][2] = obj['size'] / 2
                obj['velocity'][2] = 0
                obj['velocity'][:2] *= 0.9  # Friction

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Generate partial observation with noise.

        Returns observations combining:
        - Proprioception: internal state (pose, velocity, etc.)
        - Exteroception: external sensing (vision, touch)
        """
        # Proprioception: agent's internal state
        proprio = np.concatenate([
            self.agent_state['position'],
            self.agent_state['velocity'],
            self.agent_state['orientation'],
        ])

        # Add proprioceptive noise
        if self.config.observation_noise > 0:
            proprio += np.random.normal(0, self.config.observation_noise, size=proprio.shape)

        # Exteroception: visual observation (simplified)
        vision = self._render_vision()

        # Touch sensors (8 directions around the agent)
        touch = self._compute_touch_sensors()

        return {
            'proprioception': proprio.astype(np.float32),
            'exteroception': vision,
            'touch': touch
        }

    def _render_vision(self) -> np.ndarray:
        """
        Render visual observation with partial observability.

        Returns a top-down view of visible objects within the agent's
        field of view and vision range.
        """
        img = np.ones((64, 64, 3), dtype=np.uint8) * 135  # Gray background

        agent_pos = self.agent_state['position']
        agent_dir = self._quaternion_to_direction(self.agent_state['orientation'])

        # Render visible objects
        for obj in self.objects:
            # Check if object is in vision range
            diff = obj['position'] - agent_pos
            dist = np.linalg.norm(diff[:2])

            if dist > self.config.vision_range:
                continue

            # Check if in field of view
            if dist > 0:
                obj_dir = diff[:2] / dist
                angle = np.arccos(np.clip(np.dot(agent_dir[:2], obj_dir), -1, 1))
                if angle > np.radians(self.config.vision_angle / 2):
                    continue

            # Project to image coordinates
            rel_pos = diff[:2]
            img_x = int(32 + rel_pos[0] * 3)
            img_y = int(32 - rel_pos[1] * 3)

            if 0 <= img_x < 64 and 0 <= img_y < 64:
                color = (obj['color'] * 255).astype(np.uint8)
                size = max(1, int(obj['size'] * 2))
                img[max(0, img_y-size):min(64, img_y+size),
                    max(0, img_x-size):min(64, img_x+size)] = color

        # Add observation noise
        if self.config.observation_noise > 0:
            noise = np.random.normal(0, self.config.observation_noise * 10, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img

    def _compute_touch_sensors(self) -> np.ndarray:
        """Compute touch/proximity sensors in 8 directions."""
        touch = np.zeros(8)
        agent_pos = self.agent_state['position']

        # Check proximity to objects in 8 directions
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle), 0])

            # Find closest object in this direction
            min_dist = float('inf')
            for obj in self.objects:
                diff = obj['position'] - agent_pos
                proj = np.dot(diff, direction)
                if proj > 0:  # Object in front
                    dist = np.linalg.norm(diff)
                    if dist < min_dist:
                        min_dist = dist

            # Convert distance to touch signal (0=far, 1=touching)
            touch[i] = np.exp(-min_dist / 2.0)

        return touch.astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute task reward.

        This is a simple exploration reward. In practice, this would be
        task-specific (e.g., reaching a goal, collecting objects, etc.)
        """
        # Reward for exploration (visiting new areas)
        exploration_reward = 0.1

        # Small penalty for excessive actions (energy cost)
        action_cost = -0.01 * np.sum(action**2)

        # Reward for staying alive
        survival_reward = 0.01

        return exploration_reward + action_cost + survival_reward

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if agent falls off the world
        if self.agent_state['position'][2] < -10:
            return True

        # Terminate if agent goes too far
        if np.linalg.norm(self.agent_state['position'][:2]) > 100:
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            'time': self.time,
            'step': self.step_count,
            'agent_position': self.agent_state['position'].copy(),
            'agent_velocity': self.agent_state['velocity'].copy(),
            'num_objects': len(self.objects),
        }

    # Quaternion utilities
    def _quaternion_to_direction(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to forward direction vector."""
        # Rotate [1, 0, 0] by quaternion
        x = 1 - 2*(q[2]**2 + q[3]**2)
        y = 2*(q[1]*q[2] + q[0]*q[3])
        z = 2*(q[1]*q[3] - q[0]*q[2])
        return np.array([x, y, z])

    def _axis_angle_to_quaternion(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to quaternion."""
        half_angle = angle / 2
        return np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle)
        ])

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        """Render the environment (optional, for visualization)."""
        return self._render_vision()

    def get_full_state(self) -> Dict:
        """
        Get complete environment state (for evaluation only).

        This provides access to the full state, bypassing partial observability.
        Used for ground-truth evaluation and analysis.
        """
        return {
            'agent': self.agent_state.copy(),
            'objects': [obj.copy() for obj in self.objects],
            'world': self.world_state.copy(),
            'time': self.time,
        }
