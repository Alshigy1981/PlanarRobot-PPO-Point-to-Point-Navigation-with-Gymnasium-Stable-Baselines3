# ================== IMPORTS ==================
import gymnasium as gym
from gymnasium import spaces
print("Gymnasium version:", gym.__version__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ================== UTIL ==================
def wrap_angle(a):
    # Wrap to (-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

# ================== ENV ==================
class PlanarRobotEnv(gym.Env):
    """
    Simple planar robot with state [x, y, psi].
    Action is 2D in [-1, 1]: we map to [0..v_max] for speed and [-w_max..w_max] for yaw rate.
    Reward uses progress-based shaping toward a point goal.
    """
    metadata = {"render_modes": []}

    def __init__(self, dt=0.05, v_max=1.5, w_max=2.0, goal=None, max_steps=400, seed=None):
        super().__init__()
        self.dt = dt
        self.v_max = v_max
        self.w_max = w_max
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.goal = np.array([0.0, 0.0]) if goal is None else np.array(goal, dtype=np.float32)

        # For progress shaping
        self.prev_dist = None

        # Observation: [x, y, psi]
        high = np.array([np.inf, np.inf, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Action (symmetric): a_v, a_w in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Initialize
        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Randomize start around a ring to diversify initial conditions
        r = self.rng.uniform(1.0, 3.0)
        th = self.rng.uniform(-np.pi, np.pi)
        self.state = np.array(
            [r * np.cos(th), r * np.sin(th), self.rng.uniform(-np.pi, np.pi)],
            dtype=np.float32
        )
        self.steps = 0

        # Track initial distance
        pos = self.state[:2]
        self.prev_dist = np.linalg.norm(pos - self.goal)

        obs = self.state.copy()
        return obs, {}

    def step(self, action):
        # Clip action to [-1, 1]
        a_v = float(np.clip(action[0], -1.0, 1.0))
        a_w = float(np.clip(action[1], -1.0, 1.0))

        # Map to physical actuation
        v = 0.5 * (a_v + 1.0) * self.v_max      # speed in [0, v_max]
        w = a_w * self.w_max                    # yaw rate in [-w_max, w_max]

        # Unpack state
        x, y, psi = self.state

        # Integrate
        x  += self.dt * v * np.cos(psi)
        y  += self.dt * v * np.sin(psi)
        psi = wrap_angle(psi + self.dt * w)

        self.state = np.array([x, y, psi], dtype=np.float32)
        self.steps += 1

        # Shaping: reward progress toward goal
        pos  = self.state[:2]
        dist = np.linalg.norm(pos - self.goal)

        progress = (self.prev_dist - dist) if (self.prev_dist is not None) else 0.0
        reward   = 2.0 * progress                      # encourage getting closer
        reward  -= 0.01 * (v**2 + w**2)                # small effort penalty

        # Success bonus
        terminated = dist < 0.1
        if terminated:
            reward += 50.0

        # Timeout
        truncated = self.steps >= self.max_steps

        info = {"distance": dist, "progress_made": progress}

        # Update after computing progress/info
        self.prev_dist = dist

        return self.state.copy(), reward, terminated, truncated, info

# ================== ENV FACTORIES ==================
def make_env(goal=(2.0, 2.0), seed=None):
    # Wrap with Monitor for logging
    env = PlanarRobotEnv(dt=0.05, goal=list(goal), seed=seed)
    return Monitor(env)

# ================== TRAINING SETUP ==================
# Vectorized training env + normalization
train_env = DummyVecEnv([lambda: make_env()])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

# Stronger PPO configuration
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=1e-3,
    n_steps=4096,
    batch_size=256,
    n_epochs=20,
    gamma=0.99,
    ent_coef=1e-3,       # a touch of exploration
    clip_range=0.2,
    target_kl=0.02,
    verbose=1,
)

print("Training the agent...")
model.learn(total_timesteps=500_000)
print("Training completed!")

# Save normalization stats (and optionally the model)
train_env.save("vecnormalize_planar.pkl")
model.save("ppo_planar.zip")

# ================== EVAL / VISUALIZATION ENV ==================
# Load normalization stats into a fresh eval env
eval_env = DummyVecEnv([lambda: make_env()])
eval_env = VecNormalize.load("vecnormalize_planar.pkl", eval_env)
eval_env.training = False       # don't update running stats during eval
eval_env.norm_reward = False    # show raw returns

# Helper to get the underlying base (unwrapped) env to read true state for plotting
def get_base_planar_env(vec_env):
    """
    Traverse wrappers: VecNormalize -> DummyVecEnv -> Monitor -> PlanarRobotEnv
    and return the PlanarRobotEnv instance for reading true state/goal.
    """
    base = vec_env.venv.envs[0]   # Monitor
    while hasattr(base, "env"):
        base = base.env
    return base  # PlanarRobotEnv

# ================== VISUALIZATION: PERFORMANCE ==================
def visualize_agent_performance_vec(model, vec_env, num_episodes=5, save_plots=True):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    all_trajectories, all_rewards, all_distances = [], [], []
    success_count = 0

    for ep in range(num_episodes):
        obs = vec_env.reset()  # <-- only obs
        base_env = get_base_planar_env(vec_env)
        traj = [base_env.state[:2].copy()]
        ep_rewards, ep_distances = [], []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            done = bool(dones[0])

            base_env = get_base_planar_env(vec_env)
            traj.append(base_env.state[:2].copy())

            ep_rewards.append(float(reward[0]))
            ep_distances.append(float(infos[0].get("distance", np.nan)))

        all_trajectories.append(np.array(traj))
        all_rewards.append(ep_rewards)
        all_distances.append(ep_distances)

        if len(ep_distances) > 0 and ep_distances[-1] < 0.1:
            success_count += 1

    # (rest of your plotting code unchanged) ...


    # ---- Plot 1: Trajectories
    ax1 = axes[0, 0]
    for i, traj in enumerate(all_trajectories):
        total_R = sum(all_rewards[i])
        color = 'green' if total_R > -1000 else 'red'
        ax1.plot(traj[:, 0], traj[:, 1], '-', alpha=0.8, linewidth=2, color=color, label=f'Episode {i+1}')
        ax1.plot(traj[0, 0], traj[0, 1], 'bo', markersize=8)   # start
        ax1.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8) # end

    # Goal
    base_env = get_base_planar_env(vec_env)
    goal_circle = plt.Circle(base_env.goal, 0.1, color='gold', alpha=0.7, label='Goal')
    ax1.add_patch(goal_circle)
    ax1.plot(base_env.goal[0], base_env.goal[1], 'g*', markersize=15)

    ax1.set_xlabel('X Position'); ax1.set_ylabel('Y Position')
    ax1.set_title(f'Agent Trajectories (Success: {success_count}/{num_episodes})')
    ax1.grid(True, alpha=0.3); ax1.axis('equal'); ax1.legend()

    # ---- Plot 2: Distance to Goal
    ax2 = axes[0, 1]
    for i, dists in enumerate(all_distances):
        ax2.plot(dists, alpha=0.8, linewidth=2, label=f'Episode {i+1}')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Goal Threshold')
    ax2.set_xlabel('Time Steps'); ax2.set_ylabel('Distance to Goal')
    ax2.set_title('Distance to Goal Over Time'); ax2.grid(True, alpha=0.3); ax2.legend()

    # ---- Plot 3: Cumulative Rewards
    ax3 = axes[1, 0]
    for i, rewards in enumerate(all_rewards):
        ax3.plot(np.cumsum(rewards), alpha=0.8, linewidth=2, label=f'Episode {i+1}')
    ax3.set_xlabel('Time Steps'); ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Cumulative Rewards Over Episodes'); ax3.grid(True, alpha=0.3); ax3.legend()

    # ---- Plot 4: Performance Stats
    ax4 = axes[1, 1]
    episode_lengths = [len(r) for r in all_rewards]
    total_rewards    = [sum(r) for r in all_rewards]
    final_distances  = [d[-1] for d in all_distances]

    stats_data  = [episode_lengths, total_rewards, final_distances]
    stats_labels = ['Episode Length', 'Total Reward', 'Final Distance']
    x_pos = np.arange(len(stats_labels))
    means = [np.mean(d) for d in stats_data]
    stds  = [np.std(d) for d in stats_data]

    bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'green', 'orange'])
    ax4.set_xlabel('Metrics'); ax4.set_ylabel('Values'); ax4.set_title('Performance Statistics')
    ax4.set_xticks(x_pos); ax4.set_xticklabels(stats_labels); ax4.grid(True, alpha=0.3)

    for bar, mean in zip(bars, means):
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., h + 0.02*abs(h), f'{mean:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_plots:
        plt.savefig('agent_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Console summary
    print(f"\n=== PERFORMANCE SUMMARY ===")
    print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Total Reward:   {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"Average Final Distance: {np.mean(final_distances):.3f} ± {np.std(final_distances):.3f}")

    return all_trajectories, all_rewards, success_count

# ================== VISUALIZATION: ANIMATION ==================
def create_animated_trajectory_vec(model, vec_env, max_steps=400):
    obs = vec_env.reset()  # <-- only obs
    base_env = get_base_planar_env(vec_env)

    trajectory = [base_env.state.copy()]
    rewards = []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)

        base_env = get_base_planar_env(vec_env)
        trajectory.append(base_env.state.copy())
        rewards.append(float(reward[0]))

        if bool(dones[0]):
            break

    # (rest of your animation code unchanged) ...


    trajectory = np.array(trajectory)

    # Figure + axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Main trajectory plot
    ax1.set_xlim(trajectory[:, 0].min() - 0.5, trajectory[:, 0].max() + 0.5)
    ax1.set_ylim(trajectory[:, 1].min() - 0.5, trajectory[:, 1].max() + 0.5)
    ax1.set_aspect('equal')

    # Goal
    goal_circle = plt.Circle(base_env.goal, 0.1, color='gold', alpha=0.7)
    ax1.add_patch(goal_circle)
    ax1.plot(base_env.goal[0], base_env.goal[1], 'g*', markersize=15, label='Goal')

    # Robot trail and current position
    trail_line, = ax1.plot([], [], 'b-', alpha=0.6, linewidth=2, label='Path')
    robot_pos,  = ax1.plot([], [], 'ro', markersize=10, label='Robot')

    # Robot orientation arrow
    arrow = patches.FancyArrowPatch((0, 0), (0, 0), arrowstyle='->', mutation_scale=20, color='red')
    ax1.add_patch(arrow)

    ax1.set_xlabel('X Position'); ax1.set_ylabel('Y Position')
    ax1.set_title('Robot Navigation Animation'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Reward plot
    ax2.set_xlim(0, len(rewards))
    ax2.set_ylim(min(rewards) - 1, max(rewards) + 1 if len(rewards) > 0 else 1)
    reward_line, = ax2.plot([], [], 'g-', linewidth=2)
    ax2.set_xlabel('Time Step'); ax2.set_ylabel('Reward')
    ax2.set_title('Reward Over Time'); ax2.grid(True, alpha=0.3)

    def animate(frame):
        if frame < len(trajectory):
            # Update trail
            trail_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])

            # Update robot position
            robot_pos.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])

            # Update orientation arrow
            x, y, psi = trajectory[frame]
            arrow_length = 0.3
            dx = arrow_length * np.cos(psi)
            dy = arrow_length * np.sin(psi)
            arrow.set_positions((x, y), (x + dx, y + dy))

            # Rewards
            if frame > 0:
                reward_line.set_data(range(frame), rewards[:frame])

        return trail_line, robot_pos, arrow, reward_line

    anim = FuncAnimation(fig, animate, frames=len(trajectory), interval=50, blit=True, repeat=True)
    plt.tight_layout()
    return anim

# ================== RUN VISUALIZATIONS ==================
print("\n" + "="*50)
print("VISUALIZING AGENT PERFORMANCE")
print("="*50)

trajectories, rewards, success_count = visualize_agent_performance_vec(model, eval_env, num_episodes=5)

print("\nCreating animated trajectory...")
anim = create_animated_trajectory_vec(model, eval_env)
plt.show()
# To display inline in Jupyter/Colab:
HTML(anim.to_jshtml())

print("\nVisualization complete!")
from matplotlib import rc
rc('animation', html='jshtml')
anim = create_animated_trajectory_vec(model, eval_env)

from IPython.display import HTML, display
display(HTML(anim.to_jshtml()))  # <-- shows the animation inline in Colab
anim = create_animated_trajectory_vec(model, eval_env)
anim.save('robot_navigation.gif', writer='pillow', fps=20)

from IPython.display import Image, display
display(Image('robot_navigation.gif'))
