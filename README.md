# Door Opening with Reinforcement Learning

Training a 7-DoF Panda robot arm to open a door using Soft Actor-Critic (SAC) in the robosuite simulation environment.

**CS 188 — Computational Approaches to Robot Autonomy**

## Project Overview

This project trains an RL agent to complete the robosuite `Door` task — reaching for a door handle, grasping it, rotating the latch, and pulling the door open. The agent learns entirely from trial and error using the SAC algorithm, with no hand-coded motion planning.

## Project Structure

```
├── train_door.py          # Training script: environment wrapper, SAC training, evaluation
├── train_door.ipynb       # Notebook interface for training
├── visuals.ipynb          # Generates all figures, plots, and website exports
├── test.py                # Quick test/evaluation script
├── models/
│   ├── door_sac/          # Best and final trained models
│   └── door_sac_checkpoints/  # Checkpoints at 50k, 100k, 150k, 200k steps
├── logs/
│   ├── door_sac/          # Evaluation logs (evaluations.npz)
│   └── door_sac_tb/       # TensorBoard event files
├── figures/               # Generated plots (training curves, diagnostics, etc.)
├── docs/                  # Project website (GitHub Pages)
└── project_1/             # PID control assignment (separate)
```

## Setup

### Prerequisites

- Python 3.10+
- MuJoCo (included with robosuite)

### Installation

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install robosuite gymnasium stable-baselines3 tensorboard numpy matplotlib plotly
```

### Verify Installation

```bash
python -c "import robosuite; import stable_baselines3; print('Ready')"
```

## Usage

### Train from Scratch

```bash
python train_door.py --timesteps 200000
```

This trains a SAC agent for 200k steps with:
- Evaluation every 10k steps (10 episodes)
- Checkpoints saved every 50k steps
- TensorBoard logging

### Resume Training from Checkpoint

```bash
python train_door.py --timesteps 100000 --resume-from models/door_sac_checkpoints/door_200000_steps
```

### Evaluate a Trained Model

```bash
mjpython train_door.py --eval --model-path models/door_sac/best_model
```

Note: Use `mjpython` (not `python`) for the MuJoCo viewer to render correctly on macOS.

### Monitor Training

```bash
tensorboard --logdir logs/door_sac_tb/
```

Open `http://localhost:6006` to view live training curves, loss plots, and entropy diagnostics.

## Method

### Algorithm: Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic algorithm that maximizes both expected reward and entropy. Key advantages for this task:

- **Sample-efficient** — learns from a replay buffer, reusing past experience
- **Entropy regularization** — encourages exploration, preventing early convergence to poor strategies
- **Continuous actions** — natively handles the 8-dim action space (7 joints + 1 gripper)

### Hyperparameters

| Parameter | Value |
|---|---|
| Policy network | MLP (256, 256) |
| Learning rate | 3e-4 |
| Replay buffer | 1,000,000 |
| Batch size | 256 |
| Discount (γ) | 0.99 |
| Soft update (τ) | 0.005 |
| Learning starts | 10,000 steps |
| Total training | 200,000 steps |

### Reward Design

The base robosuite reward provides reaching (distance to handle) and rotation (handle angle) signals. We added three shaping terms via a Gymnasium wrapper:

1. **Grasp reward (+0.5)** — fills the gap between "hand near handle" and "handle rotating" by rewarding gripper contact with handle geoms
2. **Action penalty (-0.01·||a||²)** — discourages large/jerky joint commands for smoother motion
3. **Time penalty (-0.002/step)** — incentivizes finishing the task faster

These additions are toggled with the `extra_shaping` flag in `RobosuiteGymEnv`.

## Results

| Metric | Value |
|---|---|
| Mean Reward | 303 ± 148 |
| Success Rate | 80% |
| Best Episode | 453 |
| Training Time | ~40 min (MacBook Pro, 32GB, CPU) |

## Generating Figures

Open `visuals.ipynb` and run all cells. This generates:

- Training reward curve with ±1 std
- Episode length over training
- Reward decomposition (reaching vs rotating)
- 3D gripper trajectory
- Action heatmap
- Checkpoint comparison (reward + success rate)
- SAC diagnostics (actor/critic loss, entropy)
- Rollout consistency
- Interactive Plotly charts for the website

All figures are saved to `figures/` and website exports to `docs/`.

## Website

The project website deployed at https://monkeypunkey21.github.io/cs188_project/. To preview locally:

```bash
cd docs && python -m http.server 8000
```

Then open `http://localhost:8000`.

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| robosuite | 1.5.2 | Task environment + MuJoCo |
| gymnasium | 1.2.3 | Standard RL environment API |
| stable-baselines3 | 2.7.1 | SAC implementation |
| tensorboard | 2.20.0 | Training monitoring |
| numpy | 1.26.4 | Numerical computation |
| matplotlib | 3.9.1 | Static plots |
| plotly | 5.x | Interactive visualizations |
| mujoco | 3.4.0 | Physics simulation |
