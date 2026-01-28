# Embodied Learning Framework

A reproducible framework for embodied learning that combines world model learning, policy optimization, and intrinsic motivation. The agent learns to control a simulated body in a physics-based environment through interaction, using self-supervised learning and reinforcement learning.

## Overview

This framework implements an embodied learning setting where:

- **Environment**: A physics-based 3D world with partial observability, where agents perceive the world through limited, noisy sensors
- **Body**: An actuated body with proprioception (internal state sensing) and exteroception (external sensing via vision and touch)
- **Learning System**: Joint training of:
  - A **world model** (self-supervised from interaction data)
  - A **policy** (optimized with reinforcement learning from task rewards)
  - **Intrinsic motivation** via prediction-error novelty and goal-conditioned imagination

The framework provides tools to disentangle and evaluate perception, control, exploration, and long-horizon planning capabilities.

## Key Features

✓ **Physics-based environment** with configurable dynamics and partial observability
✓ **Multimodal sensing**: Proprioception, vision, and touch sensors
✓ **World model** with encoding, dynamics prediction, and reconstruction
✓ **Policy network** with actor-critic architecture
✓ **Intrinsic motivation** combining prediction error novelty and goal imagination
✓ **Joint training algorithm** with PPO and world model optimization
✓ **Evaluation protocols** for perception, control, exploration, and planning
✓ **Reproducible**: Configurable via YAML, seed control, deterministic training

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Model-1.git
cd Model-1

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Gymnasium
- MuJoCo (optional, for advanced physics)
- PyYAML
- TensorBoard

## Quick Start

### Run the Demo

```bash
python examples/demo.py
```

This will demonstrate all framework components:
- Environment creation and interaction
- World model encoding, prediction, and reconstruction
- Policy action sampling and value estimation
- Intrinsic motivation computation
- Training and evaluation

### Train an Agent

```bash
python scripts/train.py --config configs/default_config.yaml
```

Training options:
- `--config`: Path to configuration file
- `--resume`: Resume from a checkpoint
- `--eval-only`: Run evaluation only (no training)

### Evaluate a Trained Agent

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --num-episodes 10 \
    --visualize
```

Evaluation options:
- `--checkpoint`: Path to model checkpoint
- `--num-episodes`: Number of episodes to evaluate
- `--visualize`: Show episode visualizations
- `--deterministic`: Use deterministic policy

## Architecture

### 1. Environment (`embodied_learning/environment/`)

**PhysicsEnvironment**: A 3D physics simulation with:
- Gravity, friction, collision dynamics
- Partial observability (limited vision range and field of view)
- Sensor noise
- Configurable episode length and dynamics

**Observations** (multimodal):
- **Proprioception** (10D): position, velocity, orientation, angular velocity
- **Exteroception** (64×64×3): RGB vision (top-down view)
- **Touch** (8D): Proximity sensors in 8 directions

**Actions** (4D): [forward, lateral, rotation, jump/interact]

### 2. Body (`embodied_learning/body/`)

**ActuatedBody**: Represents the agent's physical embodiment with:
- 4 actuators with dynamics and delays
- Proprioceptive sensors (internal state)
- Exteroceptive sensors (vision, touch)
- Sensor noise simulation

### 3. Models (`embodied_learning/models/`)

**World Model** (`world_model.py`):
- **Encoder**: Multimodal observation → latent state (256D)
- **Dynamics Model**: (latent, action) → next latent (GRU-based)
- **Decoder**: latent → reconstructed observations
- **Reward Predictor**: latent → expected reward

**Policy** (`policy.py`):
- **Actor**: latent → action distribution (Gaussian)
- **Critic**: latent → value estimate
- PPO-compatible with clipped objective
- Optional goal-conditioned variant

### 4. Learning (`embodied_learning/learning/`)

**Intrinsic Motivation** (`intrinsic_motivation.py`):

1. **Prediction Error Novelty**:
   - Rewards states where world model predictions are inaccurate
   - Encourages exploration of novel/surprising states

2. **Goal-Conditioned Imagination**:
   - Samples goal states from experience buffer
   - Rewards progress towards imagined goals
   - Enables directed exploration

### 5. Training Algorithm (`embodied_learning/algorithm/`)

**EmbodiedLearningAgent** (`joint_training.py`):

The main training loop alternates between:

1. **Environment interaction**: Collect experience using current policy
2. **World model update**: Self-supervised learning from transitions
3. **Policy update**: PPO with combined extrinsic + intrinsic rewards

**Training Objective**:
```
Total Reward = α × Extrinsic Reward + β × Intrinsic Reward

Intrinsic Reward = w₁ × Prediction Error + w₂ × Goal Progress

World Model Loss = Reconstruction Loss + Dynamics Loss + Reward Loss + KL Loss
Policy Loss = PPO Clipped Objective + Value Loss + Entropy Bonus
```

### 6. Evaluation (`embodied_learning/evaluation/`)

**EvaluationProtocols** (`protocols.py`):

Disentangles different capabilities:

- **Perception**: Reconstruction error, prediction accuracy, latent consistency
- **Control**: Action smoothness, velocity tracking, stability
- **Exploration**: State space coverage, latent diversity, distance traveled
- **Planning**: Goal-reaching success rate, planning efficiency

## Configuration

All hyperparameters are configurable via YAML files (see `configs/default_config.yaml`):

```yaml
environment:
  dt: 0.02                    # Simulation timestep
  partial_observability: true # Enable partial observability
  vision_range: 10.0          # Visual range (meters)
  observation_noise: 0.01     # Sensor noise level

world_model:
  latent_dim: 256            # Latent state dimension
  learning_rate: 0.0003      # Learning rate
  kl_weight: 0.1             # KL divergence weight

policy:
  hidden_dims: [256, 256]    # Hidden layer sizes
  entropy_coef: 0.01         # Entropy bonus coefficient

intrinsic_motivation:
  prediction_error_weight: 0.1      # Novelty bonus weight
  goal_imagination_weight: 0.05     # Goal reward weight

training:
  num_steps: 1000000         # Total training steps
  batch_size: 64             # Batch size
  gamma: 0.99                # Discount factor
  ppo_epochs: 4              # PPO update epochs
```

## Training Objective (Mathematical Formulation)

### World Model

The world model learns to:
1. Encode observations: `z_t = Encode(o_t)`
2. Predict dynamics: `z_{t+1} = Dynamics(z_t, a_t)`
3. Decode observations: `ô_t = Decode(z_t)`
4. Predict rewards: `r̂_t = RewardHead(z_t, a_t)`

**Loss function**:
```
L_wm = L_recon + L_dyn + L_reward + β_KL × KL(q(z|o) || p(z))

L_recon = ||o_t - ô_t||²
L_dyn = ||z_{t+1} - ẑ_{t+1}||²
L_reward = ||r_t - r̂_t||²
```

### Policy

The policy is optimized with PPO:

```
L_policy = L_clip + c₁ × L_value + c₂ × L_entropy

L_clip = min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)

where r_t(θ) = π_θ(a_t|z_t) / π_θ_old(a_t|z_t)
```

### Combined Reward

```
R_total = w_ext × R_extrinsic + w_int × R_intrinsic

R_intrinsic = w₁ × ||ẑ_{t+1} - z_{t+1}||₂ + w₂ × max_g(||z_t - g|| - ||z_{t+1} - g||)
```

## Implementation-Ready Algorithm

```python
# Pseudocode for the main training loop

for step in range(total_steps):
    # 1. Collect experience
    observation = env.get_observation()
    latent = world_model.encode(observation)
    action = policy.sample(latent)
    next_observation, reward, done = env.step(action)

    # 2. Compute intrinsic reward
    prediction_error = world_model.get_prediction_error(
        observation, action, next_observation
    )
    intrinsic_reward = intrinsic_motivation.compute(
        prediction_error, latent, next_latent
    )

    total_reward = extrinsic_weight * reward + intrinsic_weight * intrinsic_reward

    # 3. Store in replay buffer
    replay_buffer.push(observation, action, total_reward, next_observation)

    # 4. Update world model
    if len(replay_buffer) >= batch_size:
        batch = replay_buffer.sample(batch_size)
        world_model_loss = world_model.compute_loss(batch)
        world_model_loss.backward()
        world_model_optimizer.step()

    # 5. Update policy (every N steps)
    if step % policy_update_freq == 0:
        advantages, returns = compute_gae(rollout_buffer)
        for epoch in range(ppo_epochs):
            policy_loss = policy.compute_ppo_loss(
                latents, actions, advantages, returns
            )
            policy_loss.backward()
            policy_optimizer.step()
```

## Evaluation Results

The evaluation protocols measure:

**Perception Metrics**:
- Reconstruction error (lower is better)
- One-step prediction accuracy (lower is better)
- Latent consistency across time (lower is better)

**Control Metrics**:
- Action smoothness (lower is better)
- Velocity tracking error (lower is better)
- Stability (maintaining height)

**Exploration Metrics**:
- State space coverage (higher is better)
- Latent diversity (higher is better)
- Distance traveled (higher is better)

**Planning Metrics**:
- Goal-reaching success rate (higher is better)
- Average steps to goal (lower is better)
- Planning efficiency (higher is better)

## Reproducibility

To ensure reproducibility:

1. **Set seeds**: The framework sets random seeds for NumPy, PyTorch, and CUDA
2. **Deterministic mode**: Enable via config: `deterministic: true`
3. **Configuration files**: All hyperparameters saved in YAML
4. **Checkpoints**: Models saved with full state (optimizer states, step counts)
5. **Evaluation protocol**: Standardized metrics and evaluation episodes

## Project Structure

```
Model-1/
├── embodied_learning/          # Main package
│   ├── environment/            # Physics environment
│   │   ├── physics_env.py      # Environment implementation
│   │   └── observation.py      # Observation processing
│   ├── body/                   # Embodied agent
│   │   ├── actuated_body.py    # Body with actuators
│   │   └── sensors.py          # Proprioception & exteroception
│   ├── models/                 # Neural networks
│   │   ├── world_model.py      # Predictive world model
│   │   ├── policy.py           # Actor-critic policy
│   │   └── networks.py         # Network building blocks
│   ├── learning/               # Learning algorithms
│   │   └── intrinsic_motivation.py  # Intrinsic rewards
│   ├── algorithm/              # Training algorithm
│   │   └── joint_training.py   # Main training loop
│   └── evaluation/             # Evaluation protocols
│       └── protocols.py        # Perception/control/exploration/planning
├── configs/                    # Configuration files
│   └── default_config.yaml     # Default hyperparameters
├── scripts/                    # Executable scripts
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── examples/                   # Example usage
│   └── demo.py                 # Demo script
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{embodied_learning_framework,
  title={Embodied Learning Framework: A Reproducible Implementation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/Model-1}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

This framework implements concepts from:
- World Models (Ha & Schmidhuber, 2018)
- Dreamer (Hafner et al., 2019)
- PPO (Schulman et al., 2017)
- Curiosity-driven exploration (Pathak et al., 2017)
- Goal-conditioned RL (Andrychowicz et al., 2017)

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
