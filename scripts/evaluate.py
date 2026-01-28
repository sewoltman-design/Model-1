"""
Evaluation script for trained embodied learning agents.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/default_config.yaml
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embodied_learning.environment.physics_env import PhysicsEnvironment, EnvironmentConfig
from embodied_learning.algorithm.joint_training import EmbodiedLearningAgent, AgentConfig
from embodied_learning.models.world_model import WorldModelConfig
from embodied_learning.models.policy import PolicyConfig
from embodied_learning.learning.intrinsic_motivation import IntrinsicMotivationConfig
from embodied_learning.evaluation.protocols import EvaluationProtocols


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_configs_from_yaml(config: dict):
    """Create configuration objects from YAML config."""
    env_config = EnvironmentConfig(**config['environment'])
    wm_config = WorldModelConfig(**config['world_model'])
    policy_config = PolicyConfig(**config['policy'])
    im_config = IntrinsicMotivationConfig(**config['intrinsic_motivation'])

    agent_config = AgentConfig(
        world_model=wm_config,
        policy=policy_config,
        intrinsic_motivation=im_config,
        **config['training']
    )

    return env_config, agent_config


def visualize_episode(agent, env, deterministic=True):
    """Run and visualize one episode."""
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    positions = []

    print("\nRunning episode...")

    while not done and steps < 1000:
        # Get action
        obs_tensors = agent._obs_to_tensors(obs)

        with torch.no_grad():
            latent, _, _ = agent.world_model.encode(
                obs_tensors['proprioception'],
                obs_tensors['exteroception'],
                obs_tensors['touch'],
                deterministic=True
            )

            action, _ = agent.policy(latent, deterministic=deterministic)
            action = action.cpu().numpy()[0]

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1
        positions.append(info['agent_position'][:2].copy())

        if steps % 100 == 0:
            print(f"  Step {steps}, Reward: {total_reward:.2f}, Position: {info['agent_position'][:2]}")

    print(f"\nEpisode finished:")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Distance traveled: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f}m")

    return total_reward, steps, positions


def main():
    parser = argparse.ArgumentParser(description='Evaluate embodied learning agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize episodes')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy')
    parser.add_argument('--output', type=str, default='evaluation_results.txt',
                        help='Output file for results')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Create environment and agent
    env_config, agent_config = create_configs_from_yaml(config)
    env = PhysicsEnvironment(env_config)
    agent = EmbodiedLearningAgent(env, agent_config)

    # Load checkpoint
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Total training steps: {agent.total_steps}")
    print(f"Total episodes: {agent.episode_count}")

    # Run comprehensive evaluation
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 60)

    evaluator = EvaluationProtocols(agent, env)
    results = evaluator.evaluate_all(num_episodes=args.num_episodes)

    report = evaluator.generate_report(results)
    print("\n" + report)

    # Save results
    with open(args.output, 'w') as f:
        f.write(report)
    print(f"\nResults saved to {args.output}")

    # Visualize episodes if requested
    if args.visualize:
        print("\n" + "=" * 60)
        print("EPISODE VISUALIZATION")
        print("=" * 60)

        episode_returns = []
        episode_lengths = []

        for i in range(min(5, args.num_episodes)):
            print(f"\n--- Episode {i+1} ---")
            ret, length, positions = visualize_episode(
                agent, env, deterministic=args.deterministic
            )
            episode_returns.append(ret)
            episode_lengths.append(length)

        print("\n" + "-" * 60)
        print("EPISODE STATISTICS")
        print("-" * 60)
        print(f"Average return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
        print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")


if __name__ == '__main__':
    main()
