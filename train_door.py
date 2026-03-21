import numpy as np
import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


class RobosuiteGymEnv(gym.Env):
    """Thin wrapper to make robosuite fully compatible with Gymnasium/SB3."""

    def __init__(self, render_mode=None, horizon=500, extra_shaping=True):
        """Initialize the robosuite Door environment with Gymnasium compatibility.

        Args:
            render_mode: Set to "human" to open the MuJoCo viewer, None for headless.
            horizon: Maximum number of timesteps per episode.
            extra_shaping: If True, adds grasp reward, action penalty, and time penalty
                on top of the base robosuite reward.
        """
        super().__init__()
        self.render_mode = render_mode
        self.extra_shaping = extra_shaping
        self.prev_action = None
        self.env = suite.make(
            env_name="Door",
            robots="Panda",
            has_renderer=(render_mode == "human"),
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=horizon,
            reward_shaping=True,
        )
        self.gym_env = GymWrapper(self.env)

        # Force float32 for SB3 compatibility
        obs_space = self.gym_env.observation_space
        self.observation_space = gym.spaces.Box(
            low=obs_space.low.astype(np.float32),
            high=obs_space.high.astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = self.gym_env.action_space

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation.

        Args:
            seed: Random seed for reproducibility (unused by robosuite).
            options: Additional reset options (unused).

        Returns:
            Tuple of (observation as float32 array, info dict).
        """
        obs, info = self.gym_env.reset()
        self.prev_action = None
        return obs.astype(np.float32), info

    def step(self, action):
        """Execute one environment step with the given action.

        Applies the base robosuite reward and, if extra_shaping is enabled, adds:
            - Grasp reward: multi-stage bonus for approaching and gripping the handle.
            - Action penalty: -0.01 * ||a||^2 to discourage jerky movements.
            - Time penalty: -0.002 per step to incentivize faster completion.

        Args:
            action: 8-dim array (7 joint velocities + 1 gripper command).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.gym_env.step(action)

        if self.extra_shaping:
            raw = self.env

            # gradual reward for getting close AND closing gripper
            handle_geoms = [raw.door.naming_prefix + name for name in ["handle", "handle_base", "latch", "latch_tip"]]

            # Binary grasp detection (bonus when fully grasping)
            grasped = raw._check_grasp(
                gripper=raw.robots[0].gripper,
                object_geoms=handle_geoms,
            )

            dist = np.linalg.norm(raw._gripper_to_handle)

            # Reward closing the gripper when near the handle
            gripper_action = action[-1]
            near_handle = dist < 0.05
            closing_gripper = gripper_action < 0

            if grasped:
                grasp_reward = 0.5  # strong bonus for actual grasp
            elif near_handle and closing_gripper:
                grasp_reward = 0.15  # partial credit: close + trying to grip
            elif near_handle:
                grasp_reward = 0.05  # near but not closing gripper
            else:
                grasp_reward = 0.0

            # Action penalty: discourage large/jerky actions
            action_penalty = -0.01 * np.sum(action ** 2)

            # Time penalty: incentivize finishing faster
            time_penalty = -0.002

            reward += grasp_reward + action_penalty + time_penalty
            info["grasp_reward"] = grasp_reward
            info["grasped"] = grasped
            info["dist_to_handle"] = dist
            info["action_penalty"] = action_penalty
            info["time_penalty"] = time_penalty

        self.prev_action = action
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        """Render the environment. Only displays if render_mode is 'human'."""
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        """Clean up the robosuite environment and free resources."""
        self.env.close()


def make_env(horizon=500, extra_shaping=True):
    """Factory function to create a RobosuiteGymEnv instance for training.

    Args:
        horizon: Maximum timesteps per episode.
        extra_shaping: Whether to enable custom reward shaping (grasp, action, time).

    Returns:
        A RobosuiteGymEnv instance (headless, no rendering).
    """
    return RobosuiteGymEnv(horizon=horizon, extra_shaping=extra_shaping)


def train(timesteps=500_000, eval_freq=10_000, save_freq=50_000, resume_from=None):
    """Train a SAC agent on the robosuite Door task.

    Sets up the training and evaluation environments, configures SB3 callbacks
    for periodic evaluation and checkpointing, and runs SAC training.

    Args:
        timesteps: Total number of environment steps to train for.
        eval_freq: How often (in steps) to run evaluation episodes.
        save_freq: How often (in steps) to save a model checkpoint.
        resume_from: Path to a saved model to resume training from. If None,
            trains from scratch.

    Returns:
        The trained SAC model.
    """
    env = make_env()
    eval_env = make_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/door_sac/",
        log_path="logs/door_sac/",
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="models/door_sac_checkpoints/",
        name_prefix="door",
    )

    if resume_from:
        print(f"Resuming training from {resume_from}...")
        model = SAC.load(resume_from, env=env, tensorboard_log="logs/door_sac_tb/")
    else:
        print("Starting training from scratch...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            learning_starts=10_000,
            tensorboard_log="logs/door_sac_tb/",
        )

    print(f"Training SAC on Door for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=(resume_from is None),
    )
    model.save("models/door_sac/final_model")
    print("Training complete. Model saved to models/door_sac/")
    env.close()
    eval_env.close()
    return model


def evaluate(model_path="models/door_sac/best_model", n_episodes=20, render_mode="human"):
    """Load a trained SAC model and run evaluation episodes.

    Loads the model from disk, runs it in the Door environment with deterministic
    actions, and prints the total reward per episode. Use render_mode="human" to
    open the MuJoCo viewer.

    Args:
        model_path: Path to the saved SB3 model (without .zip extension).
        n_episodes: Number of evaluation episodes to run.
        render_mode: "human" for visual rendering, None for headless.
    """
    env = RobosuiteGymEnv(render_mode=render_mode, horizon=200)
    model = SAC.load(model_path)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            if terminated or truncated:
                break
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--eval", action="store_true", help="Evaluate a saved model")
    parser.add_argument("--model-path", type=str, default="models/door_sac/best_model")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint path to resume training from")
    args = parser.parse_args()

    if args.eval:
        evaluate(model_path=args.model_path)
    else:
        train(timesteps=args.timesteps, eval_freq=args.eval_freq, save_freq=args.save_freq, resume_from=args.resume_from)
