"""
Training script for embodied learning framework.

Usage:
    python scripts/train.py --config configs/default_config.yaml
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add parent directory to path
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


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_configs_from_yaml(config: dict):
    """Create configuration objects from YAML config."""
    # Environment config
    env_config = EnvironmentConfig(**config['environment'])

    # World model config
    wm_config = WorldModelConfig(**config['world_model'])

    # Policy config
    policy_config = PolicyConfig(**config['policy'])

    # Intrinsic motivation config
    im_config = IntrinsicMotivationConfig(**config['intrinsic_motivation'])

    # Agent config
    agent_config = AgentConfig(
        world_model=wm_config,
        policy=policy_config,
        intrinsic_motivation=im_config,
        **config['training']
    )

    return env_config, agent_config


def main():
    parser = argparse.ArgumentParser(description='Train embodied learning agent')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Set seed for reproducibility
    set_seed(config['seed'])
    print(f"Set random seed to {config['seed']}")

    # Create directories
    log_dir = Path(config['logging']['log_dir'])
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create environment and agent
    env_config, agent_config = create_configs_from_yaml(config)
    env = PhysicsEnvironment(env_config)
    agent = EmbodiedLearningAgent(env, agent_config)

    print("\nEnvironment and agent created successfully")
    print(f"Device: {agent.device}")
    print(f"World model parameters: {sum(p.numel() for p in agent.world_model.parameters()):,}")
    print(f"Policy parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

    # Load checkpoint if resuming
    if args.resume:
        agent.load(args.resume)
        print(f"\nResumed from checkpoint: {args.resume}")
        print(f"Total steps: {agent.total_steps}")
        print(f"Episodes: {agent.episode_count}")

    # Evaluation only mode
    if args.eval_only:
        print("\n" + "=" * 60)
        print("RUNNING EVALUATION")
        print("=" * 60)

        evaluator = EvaluationProtocols(agent, env)
        results = evaluator.evaluate_all(num_episodes=10)

        report = evaluator.generate_report(results)
        print("\n" + report)

        # Save report
        report_path = log_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")

        return

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    num_steps = config['training']['num_steps']
    eval_interval = config['training']['eval_interval']
    save_interval = config['training']['save_interval']
    log_interval = config['logging']['log_interval']

    best_return = -float('inf')
    episode_returns = []

    try:
        for step in range(agent.total_steps, num_steps):
            # Training step
            if step % log_interval == 0:
                print(f"\nStep {step}/{num_steps}")

            # Train for log_interval steps
            steps_to_train = min(log_interval, num_steps - step)
            metrics = agent.train(steps_to_train)

            # Log metrics
            if len(metrics['episode_returns']) > 0:
                avg_return = np.mean(metrics['episode_returns'])
                episode_returns.extend(metrics['episode_returns'])

                print(f"  Episodes: {agent.episode_count}")
                print(f"  Avg Return (last episodes): {avg_return:.2f}")
                if len(metrics['world_model_loss']) > 0:
                    print(f"  World Model Loss: {np.mean(metrics['world_model_loss']):.4f}")
                if len(metrics['policy_loss']) > 0:
                    print(f"  Policy Loss: {np.mean(metrics['policy_loss']):.4f}")

                # Track best model
                if avg_return > best_return:
                    best_return = avg_return
                    best_path = checkpoint_dir / 'best_model.pt'
                    agent.save(str(best_path))
                    print(f"  New best model saved (return: {best_return:.2f})")

            # Save checkpoint
            if (step + 1) % save_interval == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_step_{step+1}.pt'
                agent.save(str(checkpoint_path))
                print(f"\nCheckpoint saved: {checkpoint_path}")

            # Run evaluation
            if (step + 1) % eval_interval == 0:
                print("\n" + "-" * 60)
                print("RUNNING EVALUATION")
                print("-" * 60)

                evaluator = EvaluationProtocols(agent, env)
                results = evaluator.evaluate_all(num_episodes=5)

                report = evaluator.generate_report(results)
                print("\n" + report)

                # Save evaluation results
                eval_path = log_dir / f'eval_step_{step+1}.txt'
                with open(eval_path, 'w') as f:
                    f.write(report)

                print("-" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Final save
    final_path = checkpoint_dir / 'final_model.pt'
    agent.save(str(final_path))
    print(f"\nFinal model saved: {final_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    evaluator = EvaluationProtocols(agent, env)
    results = evaluator.evaluate_all(num_episodes=10)

    report = evaluator.generate_report(results)
    print("\n" + report)

    final_report_path = log_dir / 'final_evaluation_report.txt'
    with open(final_report_path, 'w') as f:
        f.write(report)

    print(f"\nTraining complete!")
    print(f"Best return: {best_return:.2f}")
    print(f"Total episodes: {agent.episode_count}")
    print(f"Final report saved to {final_report_path}")


if __name__ == '__main__':
    main()
