"""
Demo script showcasing the embodied learning framework.

This script demonstrates:
1. Creating an environment
2. Initializing an agent
3. Training the agent
4. Evaluating different capabilities
5. Visualizing results
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embodied_learning.environment.physics_env import PhysicsEnvironment, EnvironmentConfig
from embodied_learning.algorithm.joint_training import EmbodiedLearningAgent, AgentConfig
from embodied_learning.models.world_model import WorldModelConfig
from embodied_learning.models.policy import PolicyConfig
from embodied_learning.learning.intrinsic_motivation import IntrinsicMotivationConfig
from embodied_learning.evaluation.protocols import EvaluationProtocols


def main():
    print("=" * 60)
    print("EMBODIED LEARNING FRAMEWORK DEMO")
    print("=" * 60)

    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"\nRandom seed: {seed}")

    # ===================================================================
    # 1. Create Environment
    # ===================================================================
    print("\n" + "-" * 60)
    print("1. CREATING PHYSICS ENVIRONMENT")
    print("-" * 60)

    env_config = EnvironmentConfig(
        dt=0.02,
        max_episode_steps=500,
        observation_noise=0.01,
        partial_observability=True,
        vision_range=10.0
    )

    env = PhysicsEnvironment(env_config)
    print("Environment created successfully!")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    # Test environment
    obs, _ = env.reset()
    print("\nTesting environment with random actions...")
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Position: {info['agent_position'][:2]}, Reward: {reward:.3f}")

    # ===================================================================
    # 2. Initialize Agent
    # ===================================================================
    print("\n" + "-" * 60)
    print("2. INITIALIZING LEARNING AGENT")
    print("-" * 60)

    # Configure models
    world_model_config = WorldModelConfig(
        latent_dim=128,  # Smaller for demo
        hidden_dim=256,
        learning_rate=3e-4
    )

    policy_config = PolicyConfig(
        latent_dim=128,
        hidden_dims=(128, 128),
        learning_rate=3e-4
    )

    intrinsic_config = IntrinsicMotivationConfig(
        prediction_error_weight=0.1,
        goal_imagination_weight=0.05
    )

    agent_config = AgentConfig(
        world_model=world_model_config,
        policy=policy_config,
        intrinsic_motivation=intrinsic_config,
        batch_size=32,
        buffer_size=10000,
        policy_update_freq=1024,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    agent = EmbodiedLearningAgent(env, agent_config)

    print("Agent initialized successfully!")
    print(f"  Device: {agent.device}")
    print(f"  World model parameters: {sum(p.numel() for p in agent.world_model.parameters()):,}")
    print(f"  Policy parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

    # ===================================================================
    # 3. Train Agent (short demo)
    # ===================================================================
    print("\n" + "-" * 60)
    print("3. TRAINING AGENT (DEMO - SHORT)")
    print("-" * 60)

    print("Training for 5000 steps...")
    metrics = agent.train(num_steps=5000)

    print("\nTraining metrics:")
    if len(metrics['episode_returns']) > 0:
        print(f"  Episodes completed: {len(metrics['episode_returns'])}")
        print(f"  Average return: {np.mean(metrics['episode_returns']):.2f}")
        print(f"  Average episode length: {np.mean(metrics['episode_lengths']):.1f}")
    if len(metrics['world_model_loss']) > 0:
        print(f"  World model loss: {np.mean(metrics['world_model_loss']):.4f}")
    if len(metrics['policy_loss']) > 0:
        print(f"  Policy loss: {np.mean(metrics['policy_loss']):.4f}")

    # ===================================================================
    # 4. Evaluate Capabilities
    # ===================================================================
    print("\n" + "-" * 60)
    print("4. EVALUATING AGENT CAPABILITIES")
    print("-" * 60)

    evaluator = EvaluationProtocols(agent, env)

    print("\nRunning comprehensive evaluation...")
    results = evaluator.evaluate_all(num_episodes=3)

    report = evaluator.generate_report(results)
    print("\n" + report)

    # ===================================================================
    # 5. Demonstrate Components
    # ===================================================================
    print("\n" + "-" * 60)
    print("5. DEMONSTRATING FRAMEWORK COMPONENTS")
    print("-" * 60)

    obs, _ = env.reset()
    obs_tensors = agent._obs_to_tensors(obs)

    with torch.no_grad():
        # World Model: Encode observation
        print("\na) World Model - Encoding observation:")
        latent, mean, logstd = agent.world_model.encode(
            obs_tensors['proprioception'],
            obs_tensors['exteroception'],
            obs_tensors['touch']
        )
        print(f"   Latent state shape: {latent.shape}")
        print(f"   Latent mean: {mean[0, :5].cpu().numpy()}")

        # World Model: Reconstruct observation
        print("\nb) World Model - Reconstructing observation:")
        proprio_recon, vision_recon, touch_recon = agent.world_model.decode(latent)
        print(f"   Proprio reconstruction shape: {proprio_recon.shape}")
        print(f"   Vision reconstruction shape: {vision_recon.shape}")
        print(f"   Touch reconstruction shape: {touch_recon.shape}")

        # World Model: Predict next state
        print("\nc) World Model - Predicting next state:")
        action = torch.randn(1, 4, device=agent.device)
        next_latent, reward_pred, hidden = agent.world_model.predict_next(latent, action)
        print(f"   Next latent shape: {next_latent.shape}")
        print(f"   Predicted reward: {reward_pred.item():.3f}")

        # Policy: Sample action
        print("\nd) Policy - Sampling action:")
        action, log_prob = agent.policy(latent)
        print(f"   Action: {action[0].cpu().numpy()}")
        print(f"   Log probability: {log_prob.item():.3f}")

        # Policy: Get value estimate
        value = agent.policy.get_value(latent)
        print(f"   Value estimate: {value.item():.3f}")

    # Intrinsic Motivation
    print("\ne) Intrinsic Motivation - Computing rewards:")
    obs2, _, _, _, _ = env.step(action[0].cpu().numpy())
    obs2_tensors = agent._obs_to_tensors(obs2)

    with torch.no_grad():
        latent2, _, _ = agent.world_model.encode(
            obs2_tensors['proprioception'],
            obs2_tensors['exteroception'],
            obs2_tensors['touch'],
            deterministic=True
        )

        prediction_error = agent.world_model.get_prediction_error(
            obs_tensors, action, obs2_tensors
        )

        intrinsic_reward, info = agent.intrinsic_motivation.compute_intrinsic_reward(
            prediction_error, latent, latent2
        )

    print(f"   Prediction error: {prediction_error.item():.3f}")
    print(f"   Novelty reward: {info['novelty_reward'].item():.3f}")
    print(f"   Goal reward: {info['goal_reward'].item():.3f}")
    print(f"   Total intrinsic reward: {intrinsic_reward.item():.3f}")

    # ===================================================================
    # 6. Summary
    # ===================================================================
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    print("\nFramework components demonstrated:")
    print("  ✓ Physics-based environment with partial observability")
    print("  ✓ World model with encoding, prediction, and reconstruction")
    print("  ✓ Policy network with action sampling and value estimation")
    print("  ✓ Intrinsic motivation (prediction error + goal imagination)")
    print("  ✓ Joint training algorithm combining all components")
    print("  ✓ Evaluation protocols for perception, control, exploration, planning")

    print("\nNext steps:")
    print("  • Train for longer: python scripts/train.py --config configs/default_config.yaml")
    print("  • Evaluate trained model: python scripts/evaluate.py --checkpoint checkpoints/best_model.pt")
    print("  • Customize configuration: Edit configs/default_config.yaml")

    print("\nThank you for trying the Embodied Learning Framework!")


if __name__ == '__main__':
    main()
