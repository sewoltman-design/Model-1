"""
Embodied Learning Framework

A reproducible framework for embodied learning combining:
- Physics-based environments with partial observability
- Actuated bodies with proprioception and exteroception
- Joint policy and world model training
- Intrinsic motivation through prediction error and goal imagination
"""

__version__ = "0.1.0"

from embodied_learning.environment.physics_env import PhysicsEnvironment
from embodied_learning.body.actuated_body import ActuatedBody
from embodied_learning.models.policy import Policy
from embodied_learning.models.world_model import WorldModel
from embodied_learning.algorithm.joint_training import EmbodiedLearningAgent

__all__ = [
    "PhysicsEnvironment",
    "ActuatedBody",
    "Policy",
    "WorldModel",
    "EmbodiedLearningAgent",
]
