# PlanarRobot PPO — Point-to-Point Navigation with Gymnasium & Stable-Baselines3

A minimal end-to-end reinforcement learning project that trains a PPO agent to drive a simple planar robot (state `[x, y, ψ]`) toward a goal using progress-based reward shaping. Includes training, evaluation, static plots, and an animated trajectory visualization.

---

## ✨ Features
- **Custom Gymnasium env**: `PlanarRobotEnv` with continuous actions `[a_v, a_w]`.
- **Reward shaping**: Distance-to-goal progress + success bonus + light effort penalty.
- **Stable-Baselines3 (PPO)**: Vectorized training with **VecNormalize** for observation/reward normalization.
- **Evaluation suite**: Success rate, distance curves, cumulative rewards, and summary stats.
- **Visualization**: Static performance plots and a live animation of the agent’s trajectory + rewards.

---

## 📦 Requirements
- Python 3.10+
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- numpy, matplotlib, IPython

```bash
# Recommended: use a fresh virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install gymnasium[box2d] stable-baselines3 numpy matplotlib ipython
```
> Note: `gymnasium[box2d]` is optional; it ensures common extras are available. The code here doesn’t use Box2D directly.

---

## 📁 Suggested Project Structure
```
planar_robot_rl/
├─ README.md                # ← this file
├─ planar_robot.py          # ← the provided script
└─ outputs/
   ├─ ppo_planar.zip        # SB3 model
   ├─ vecnormalize_planar.pkl
   └─ agent_performance.png
```
You can combine the code and README into a single repo or keep the script standalone.

---

## 🚀 Quickstart
1. **Save** the provided Python script as `planar_robot.py`.
2. **Run training** (500k steps by default):
   ```bash
   python planar_robot.py
   ```
3. **Artifacts** produced:
   - `ppo_planar.zip` — Trained PPO policy.
   - `vecnormalize_planar.pkl` — VecNormalize stats for consistent evaluation.
   - `agent_performance.png` — Summary plots (trajectories, distance, rewards, stats).

> If you run in a headless server, set `matplotlib` backend to `Agg` (or skip `plt.show()`), and the plot will still be saved.

---

## 🧠 Environment Summary (`PlanarRobotEnv`)
- **State**: `[x, y, ψ]` (position and yaw).  
- **Action**: `[a_v, a_w] ∈ [-1,1]^2`. Internally mapped to:
  - Speed `v ∈ [0, v_max]` via `v = 0.5 (a_v + 1) v_max`  
  - Yaw rate `w ∈ [-w_max, w_max]` via `w = a_w · w_max`
- **Dynamics**: Simple unicycle-like integration with step `dt`.
- **Reset**: Starts on a ring around the goal with random yaw for diversity.
- **Termination**: `‖pos - goal‖ < 0.1` (success).  
- **Truncation**: Reaches `max_steps` (timeout).
- **Reward**:
  - `+ 2.0 × (prev_dist - dist)` (progress toward goal)
  - `- 0.01 × (v² + w²)` (effort penalty)
  - `+ 50.0` on success

---

## ⚙️ Training Configuration (SB3 PPO)
Key hyperparameters used in the script:
- `learning_rate=1e-3`
- `n_steps=4096`, `batch_size=256`, `n_epochs=20`
- `gamma=0.99`, `clip_range=0.2`, `target_kl=0.02`
- `ent_coef=1e-3` (encourages modest exploration)
- `VecNormalize(..., norm_obs=True, norm_reward=True)`

You can adjust these for speed vs. stability. For tougher tasks, consider larger networks or curriculum on `goal` and start states.

---

## 🧪 Evaluation & Plots
The script constructs a fresh **eval env** and **loads** the saved `VecNormalize` statistics:
```python
eval_env = DummyVecEnv([lambda: make_env()])
eval_env = VecNormalize.load("vecnormalize_planar.pkl", eval_env)
eval_env.training = False       # freeze stats
eval_env.norm_reward = False    # report raw returns
```
Then it runs:
- `visualize_agent_performance_vec(...)` — produces four subplots and prints a performance summary.
- `create_animated_trajectory_vec(...)` — returns a `matplotlib.animation.FuncAnimation`.

In notebooks, you can display the animation inline:
```python
from IPython.display import HTML
HTML(anim.to_jshtml())
```
In scripts, `plt.show()` will open a window (if not headless).

---

## 🧪 Reproducibility Tips
- **Seeds**: `PlanarRobotEnv` accepts `seed`; set it at creation to fix initial states.
- **Deterministic eval**: `model.predict(obs, deterministic=True)` during evaluation.
- **Log versions**: The script prints `Gymnasium` version. You can also log `stable_baselines3.__version__`.

---

## 🧰 CLI Cheatsheet
- Train for fewer steps (quick smoke test):
  ```bash
  python planar_robot.py --quick    # (add an argparse flag if you prefer)
  ```
- Just evaluate a pre-trained model (skip training):
  ```python
  model = PPO.load("ppo_planar.zip", env=eval_env)
  trajectories, rewards, success = visualize_agent_performance_vec(model, eval_env)
  ```

---

## 🩺 Troubleshooting
- **Empty/NaN plots**: Ensure `VecNormalize` is loaded for eval and `eval_env.training=False`.
- **Animation doesn’t display**: In Jupyter, use `HTML(anim.to_jshtml())`; in headless mode, save via `anim.save("traj.mp4", fps=20)` (requires ffmpeg).
- **No progress in training**: Try increasing `ent_coef` (e.g., `3e-3`), raise `clip_range` to `0.3`, or use smaller `w_max` to simplify control early on.
- **Exploding value loss**: Lower `learning_rate`, reduce `gamma`, or increase `n_epochs` moderately.

---

## 📊 Interpreting Logs (PPO)
- **`ep_rew_mean`** rising and stabilizing indicates learning.
- **`entropy_loss`** magnitude trending toward 0 → more deterministic actions (normal upon convergence).
- **`approx_kl`** staying small and non-zero indicates stable updates; if it’s ~0 for long, the policy may have plateaued.
- **`clip_fraction`** near 0 suggests small policy changes per update; consider slightly increasing `learning_rate`/`clip_range` if you need more exploration.

---

## 🔒 License
This template is provided as-is under the MIT License. Adapt freely.

---

## 🙌 Acknowledgments
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

---

## ✅ Next Steps / Extensions
- Add obstacles and collision penalties.
- Randomize `goal` per-episode for generalization.
- Use curriculum to shrink the success radius over time.
- Swap in `SAC`/`TD3` for continuous control comparisons.
- Log to **TensorBoard** (`tensorboard_log` arg in PPO) for deeper diagnostics.

