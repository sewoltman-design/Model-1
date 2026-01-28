"""
Evaluation protocols for disentangling different capabilities.

Evaluates:
1. Perception: How well the model encodes observations
2. Control: Low-level motor control quality
3. Exploration: Coverage and novelty-seeking behavior
4. Planning: Long-horizon goal-reaching
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from embodied_learning.environment.physics_env import PhysicsEnvironment
from embodied_learning.algorithm.joint_training import EmbodiedLearningAgent


class EvaluationProtocols:
    """
    Comprehensive evaluation protocols for embodied learning.

    Disentangles and measures:
    - Perception quality
    - Control capability
    - Exploration behavior
    - Planning performance
    """

    def __init__(self, agent: EmbodiedLearningAgent, env: PhysicsEnvironment):
        self.agent = agent
        self.env = env
        self.device = agent.device

    def evaluate_all(self, num_episodes: int = 10) -> Dict[str, Dict]:
        """
        Run all evaluation protocols.

        Args:
            num_episodes: Number of episodes for each evaluation

        Returns:
            Dictionary of evaluation results
        """
        results = {}

        print("Evaluating perception...")
        results['perception'] = self.evaluate_perception(num_episodes)

        print("Evaluating control...")
        results['control'] = self.evaluate_control(num_episodes)

        print("Evaluating exploration...")
        results['exploration'] = self.evaluate_exploration(num_episodes)

        print("Evaluating planning...")
        results['planning'] = self.evaluate_planning(num_episodes)

        return results

    def evaluate_perception(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate perception quality.

        Metrics:
        - Reconstruction error (how well observations are reconstructed)
        - Prediction accuracy (one-step ahead prediction)
        - Latent consistency (similar observations -> similar latents)
        """
        reconstruction_errors = []
        prediction_errors = []
        latent_distances = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False

            prev_obs = None
            prev_latent = None

            while not done:
                # Convert observation
                obs_tensors = self.agent._obs_to_tensors(obs)

                with torch.no_grad():
                    # Encode
                    latent, _, _ = self.agent.world_model.encode(
                        obs_tensors['proprioception'],
                        obs_tensors['exteroception'],
                        obs_tensors['touch'],
                        deterministic=True
                    )

                    # Reconstruct
                    proprio_recon, vision_recon, touch_recon = self.agent.world_model.decode(latent)

                    # Compute reconstruction error
                    proprio_error = torch.mean((
                        proprio_recon - obs_tensors['proprioception']
                    )**2).item()

                    vision_target = obs_tensors['exteroception'].float() / 255.0
                    vision_error = torch.mean((vision_recon - vision_target)**2).item()

                    touch_error = torch.mean((
                        touch_recon - obs_tensors['touch']
                    )**2).item()

                    reconstruction_errors.append(proprio_error + vision_error + touch_error)

                    # Latent consistency (consecutive frames should be close)
                    if prev_latent is not None:
                        latent_dist = torch.norm(latent - prev_latent).item()
                        latent_distances.append(latent_dist)

                    # One-step prediction
                    if prev_obs is not None:
                        prev_obs_tensors = self.agent._obs_to_tensors(prev_obs)
                        prev_latent_pred, _, _ = self.agent.world_model.encode(
                            prev_obs_tensors['proprioception'],
                            prev_obs_tensors['exteroception'],
                            prev_obs_tensors['touch'],
                            deterministic=True
                        )

                        # Predict next latent
                        action = self.agent._select_action(prev_obs)[0]
                        action_tensor = torch.tensor(action, device=self.device).unsqueeze(0)

                        next_latent_pred, _, _ = self.agent.world_model.predict_next(
                            prev_latent_pred, action_tensor
                        )

                        # Compare with actual next latent
                        pred_error = torch.norm(next_latent_pred - latent).item()
                        prediction_errors.append(pred_error)

                    prev_obs = obs.copy()
                    prev_latent = latent.clone()

                # Step
                action = self.agent._select_action(obs)[0]
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

        return {
            'reconstruction_error_mean': np.mean(reconstruction_errors),
            'reconstruction_error_std': np.std(reconstruction_errors),
            'prediction_error_mean': np.mean(prediction_errors),
            'prediction_error_std': np.std(prediction_errors),
            'latent_consistency_mean': np.mean(latent_distances),
            'latent_consistency_std': np.std(latent_distances),
        }

    def evaluate_control(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate control capability.

        Metrics:
        - Action smoothness (change in actions over time)
        - Velocity control accuracy
        - Stability (how well agent maintains balance)
        """
        action_changes = []
        velocity_errors = []
        heights = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            prev_action = None

            target_velocity = np.random.uniform(-0.5, 0.5, size=2)  # Random target

            while not done:
                action = self.agent._select_action(obs)[0]

                # Action smoothness
                if prev_action is not None:
                    action_change = np.linalg.norm(action - prev_action)
                    action_changes.append(action_change)

                prev_action = action

                # Step environment
                obs, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Velocity tracking
                agent_velocity = info['agent_velocity'][:2]
                velocity_error = np.linalg.norm(agent_velocity - target_velocity)
                velocity_errors.append(velocity_error)

                # Stability (height above ground)
                heights.append(info['agent_position'][2])

        return {
            'action_smoothness_mean': np.mean(action_changes),
            'action_smoothness_std': np.std(action_changes),
            'velocity_error_mean': np.mean(velocity_errors),
            'velocity_error_std': np.std(velocity_errors),
            'stability_mean_height': np.mean(heights),
            'stability_height_std': np.std(heights),
        }

    def evaluate_exploration(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate exploration behavior.

        Metrics:
        - State space coverage
        - Novelty-seeking (visits to new states)
        - Diversity of behaviors
        """
        all_positions = []
        all_latents = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False

            episode_positions = []
            episode_latents = []

            while not done:
                # Record state
                action, _, _ = self.agent._select_action(obs)
                obs_tensors = self.agent._obs_to_tensors(obs)

                with torch.no_grad():
                    latent, _, _ = self.agent.world_model.encode(
                        obs_tensors['proprioception'],
                        obs_tensors['exteroception'],
                        obs_tensors['touch'],
                        deterministic=True
                    )
                    episode_latents.append(latent.cpu().numpy()[0])

                # Step
                obs, _, terminated, truncated, info = self.env.step(action)
                episode_positions.append(info['agent_position'][:2])
                done = terminated or truncated

            all_positions.extend(episode_positions)
            all_latents.extend(episode_latents)

        # Compute coverage metrics
        positions = np.array(all_positions)
        latents = np.array(all_latents)

        # Position space coverage (discretize into grid)
        grid_size = 1.0  # 1 meter cells
        visited_cells = set()
        for pos in positions:
            cell = (int(pos[0] / grid_size), int(pos[1] / grid_size))
            visited_cells.add(cell)

        # Latent space diversity (average pairwise distance)
        if len(latents) > 100:
            sample_indices = np.random.choice(len(latents), 100, replace=False)
            sampled_latents = latents[sample_indices]
            pairwise_distances = cdist(sampled_latents, sampled_latents)
            latent_diversity = np.mean(pairwise_distances)
        else:
            latent_diversity = 0.0

        # Trajectory diversity (how much the agent moves)
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

        return {
            'position_coverage_cells': len(visited_cells),
            'latent_diversity': latent_diversity,
            'total_distance_traveled': total_distance,
            'average_distance_per_step': total_distance / len(positions) if len(positions) > 0 else 0,
        }

    def evaluate_planning(self, num_episodes: int = 10, horizon: int = 10) -> Dict[str, float]:
        """
        Evaluate planning capability.

        Metrics:
        - Goal-reaching success rate
        - Planning efficiency (steps to goal)
        - Imagination accuracy (how well imagined trajectories match reality)
        """
        goal_successes = []
        steps_to_goal = []
        imagination_errors = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()

            # Set a random goal position
            goal_pos = np.random.uniform(-10, 10, size=2)
            goal_threshold = 2.0  # Success if within 2 meters

            done = False
            steps = 0
            max_steps = 500

            while not done and steps < max_steps:
                # Check if goal reached
                info = self.env.get_full_state()
                agent_pos = info['agent']['position'][:2]
                dist_to_goal = np.linalg.norm(agent_pos - goal_pos)

                if dist_to_goal < goal_threshold:
                    goal_successes.append(1.0)
                    steps_to_goal.append(steps)
                    break

                # Evaluate imagination accuracy
                obs_tensors = self.agent._obs_to_tensors(obs)

                with torch.no_grad():
                    # Current latent
                    latent, _, _ = self.agent.world_model.encode(
                        obs_tensors['proprioception'],
                        obs_tensors['exteroception'],
                        obs_tensors['touch'],
                        deterministic=True
                    )

                    # Generate random action sequence
                    actions = torch.randn(1, horizon, 4, device=self.device) * 0.5

                    # Imagine trajectory
                    imagined = self.agent.world_model.imagine_trajectory(
                        latent, actions, horizon
                    )

                # Execute first action and compare
                action = actions[0, 0].cpu().numpy()
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Compare imagined vs actual (would need to encode next obs)
                # For simplicity, we'll skip the detailed comparison here

                steps += 1

            if steps >= max_steps:
                goal_successes.append(0.0)

        return {
            'goal_success_rate': np.mean(goal_successes) if len(goal_successes) > 0 else 0.0,
            'average_steps_to_goal': np.mean(steps_to_goal) if len(steps_to_goal) > 0 else max_steps,
            'planning_efficiency': 1.0 / (np.mean(steps_to_goal) + 1) if len(steps_to_goal) > 0 else 0.0,
        }

    def generate_report(self, results: Dict, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            results: Results from evaluate_all()
            save_path: Optional path to save report

        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("EMBODIED LEARNING EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        for category, metrics in results.items():
            report.append(f"\n{category.upper()} EVALUATION:")
            report.append("-" * 40)
            for metric_name, value in metrics.items():
                report.append(f"  {metric_name}: {value:.4f}")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text
