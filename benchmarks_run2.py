"""
benchmark2_run.py  -  OUT-OF-DISTRIBUTION EVALUATION

Second-round (“Benchmark 2”) tests for DPO and baseline models.
 * keeps reward functions & model weights unchanged
 * supplies harder / noisier initial states
 * logs aggregate numbers + GIF visualisations
"""

import cv2
import warnings, math, collections, time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from utils import (
    get_environment,          # returns env with default settings
    setup_dpo_model,          # loads our DPO weights
    plot_eval_benchmarks      # helper to plot val–time curves
)
from benchmarks.sb3_utils import setup_benchmark_model
from tests import test_model_through_vals   # original evaluator

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. Experimental set‑up
# ------------------------------------------------------------------------------

# Seeds *not* used during the first benchmark
SEEDS = [997, 1009, 1231, 1759, 1993, 2221, 2601, 3011, 3307, 4001,
         4321, 4567, 4789, 4999, 5201, 5432, 5678, 5891, 5999, 6007]

# SEEDS = [997]

DATASETS = ['shape_boundary', 'shape', 'molecule']

ALGORITHMS = [
    'DPO_zero_order',
    'TRPO', 'PPO', 'SAC', 'DDPG', 'CrossQ', 'TQC',
    'S-TRPO', 'S-PPO', 'S-SAC', 'S-DDPG', 'S-CrossQ', 'S-TQC'
]

# gamma values from the original training run
ENV_TO_GAMMA = {'shape_boundary': 0.99, 'shape': 0.81, 'molecule': 0.0067}
DEFAULT_GAMMA = 0.99      # for all S‑variants

# rollout lengths
NUM_TRAJ =     {'shape_boundary': 200, 'shape': 200, 'molecule': 200}
NUM_STEPS =    {'shape_boundary': 20,  'shape': 20,  'molecule':  12}



# custom initial‑state modes for Benchmark 2
RESET_MODE = {
    'shape_boundary': 'random',
    'shape':          'hole',
    'molecule':       'random'        # scale widened below
}

# directory layout
#os.makedirs("output/videos",  exist_ok=True)
os.makedirs("output",       exist_ok=True)

# ------------------------------------------------------------------------------
# 2. Utility: override env.reset() so every rollout starts with our harder state
# ------------------------------------------------------------------------------
def patch_env_reset(env, env_name):
    """
    Replace env.reset with a version that produces a harder OOD start state.
    Works transparently with test_model_through_vals().
    """
    mode = RESET_MODE[env_name]

    # widen molecule torsion range – five‑times larger than during training
    if env_name == 'molecule':
        env.reset_scale *= 5.0 


    def custom_reset(*args, seed=None, **kwargs):
        #reset internal counters
        env.num_step   = 0
        env.discount   = 1.0
        if seed is not None:
            env.seed(seed)
        state = env.reset_at(mode=mode)
        # Gym vs Gymnasium return conventions
        return (state, {}) if kwargs.get("return_info", True) else state

    env.reset = custom_reset

# ------------------------------------------------------------------------------
# 3. Main loop – evaluate every algorithm on the new inputs
# ------------------------------------------------------------------------------
results      = collections.defaultdict(list)  # algo → list[(mean,std)]
eval_history = collections.defaultdict(dict)  # env  → {algo: val‑trajectories}

print("=== BENCHMARK 2 : Out-of-distribution generalisation ===")
start = time.time()

for algo in ALGORITHMS:
    is_dpo      = algo.startswith("DPO")
    is_s_variant= algo.startswith("S-")
    base_algo   = algo.replace("S-", "")

    print(f"\n▶ {algo}")
    for env_name in DATASETS:
        # ------------------------------------------------------------------ env
        env = get_environment(env_name)
        patch_env_reset(env, env_name)       # ← our custom start states

        # ---------------------------------------------------------------- model
        try:
            if is_dpo:
                gamma = ENV_TO_GAMMA[env_name]
                model = setup_dpo_model(algo, env, env_name)
            else:
                gamma = DEFAULT_GAMMA if is_s_variant else ENV_TO_GAMMA[env_name]
                prefix = "naive_" if is_s_variant else ""
                mpath  = f"benchmarks/models/{prefix}{env_name}_{base_algo}_{str(gamma).replace('.','_')}"
                model  = setup_benchmark_model(algo, env, mpath)
        except Exception as e:
            print(f"Failed to load model for {algo} on {env_name}: {e}")
            continue

            

        # -------------------------------------------------------- evaluation
        print(f"   ↳ {env_name}  (γ={gamma})")
        vals = test_model_through_vals(
            SEEDS, env, model,
            num_traj=NUM_TRAJ[env_name],
            num_step_per_traj=NUM_STEPS[env_name],
            benchmark_model=not is_dpo
        )

        key_algo = 'DPO' if algo == 'DPO_zero_order' else algo
        # average over trajs & seeds
        final_vals = vals[:, -1].reshape(len(SEEDS), NUM_TRAJ[env_name])
        mean, std  = np.mean(final_vals), np.std(final_vals)


        results[key_algo].append((mean, std))
        eval_history[env_name][key_algo] = vals
        print(f"      mean ± std final value  :  {mean:.3f} ± {std:.3f}")

# ------------------------------------------------------------------------------
# 4. Save summary CSV
# ------------------------------------------------------------------------------
display_names = {
    'shape_boundary': 'Materials deformation',
    'shape':          'Topological materials deformation',
    'molecule':       'Molecular dynamics'
}
df = pd.DataFrame(
    {k: [f"{μ:.3f} ± {σ:.3f}" for μ,σ in v] for k,v in results.items()},
    index=[display_names[d] for d in DATASETS]
).T
df.to_csv("output/benchmarks2.csv")
print("\nTop performer per task:")
for i, task in enumerate(display_names.values()):
    col = df.iloc[:, i]
    best_row = col.apply(lambda s: float(s.split('±')[0])).idxmin()
    print(f"  ✓ {task}: {best_row} → {col[best_row]}")

print("\n✔ Aggregate table written to  output/benchmarks2.csv")

# ------------------------------------------------------------------------------
# 5. Per-algorithm videos – legacy “frames-folder” pipeline
# ------------------------------------------------------------------------------

# shorter rollout lengths just for video generation
VIDEO_NUM_TRAJ = {'shape_boundary': 5, 'shape': 5, 'molecule': 5}
VIDEO_NUM_STEPS = {'shape_boundary': 10, 'shape': 10, 'molecule': 6}


def get_model_action(model, obs, prev_action=None):
    if hasattr(model, "predict"):
        act, _ = model.predict(obs, deterministic=True)
        return act
    if hasattr(model, "act"):
        return model.act(obs)
    if hasattr(model, "get_action"):
        try:
            return model.get_action(obs, prev_action)
        except TypeError:
            return model.get_action(obs)
    if callable(model):
        return model(obs)
    raise AttributeError("Unrecognized model interface")


def rollout_and_stream_video(env, model, env_name, algo, *,
                              steps_per_traj: int,
                              num_traj: int,
                              output_path: str,
                              fps: int = 15):
    """
    Runs trajectories and directly writes rendered frames into an MP4 file.
    No frames are saved to disk as images.
    """
    print(f"▶ Streaming video: {env_name} / {algo}")

    obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
    frame = env.render()
    h, w = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for traj_idx in range(num_traj):
        obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        prev_action = np.zeros_like(obs)

        for step_idx in range(steps_per_traj):
            action = get_model_action(model, obs, prev_action)
            obs, reward, done, *_ = env.step(action)
            frame = env.render()
            if frame is None:
                continue

            # Overlay step/traj info
            frame_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame_pil)
            draw.text((6, 6), f"step {step_idx:02d} | traj {traj_idx}", fill="red")
            frame_np = np.array(frame_pil)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            writer.write(frame_bgr)
            if done:
                break

        if traj_idx % 20 == 0:
            print(f"▶ stored trajectory {traj_idx}")

    writer.release()
    print(f"✔ wrote MP4 → {output_path}")

"""
print("\nGenerating rollout videos…")

for algo in eval_history['shape'].keys():  
    is_dpo      = algo.startswith("DPO")
    is_s_variant= algo.startswith("S-")
    base_algo   = algo.replace("S-", "")

    for env_name in DATASETS:
        env = get_environment(env_name)
        patch_env_reset(env, env_name)

        # make sure env returns RGB frames
        if hasattr(env, "render_mode"):
            env.render_mode = "rgb_array"

        gamma = DEFAULT_GAMMA if is_s_variant else ENV_TO_GAMMA.get(env_name, DEFAULT_GAMMA)
        model = (
            setup_dpo_model(algo, env, env_name)
            if algo in {"DPO", "DPO_zero_order"}
            else setup_benchmark_model(
                algo,
                env,
                f"benchmarks/models/{'naive_' if is_s_variant else ''}"
                f"{env_name}_{base_algo}_{str(gamma).replace('.', '_')}"
            )
        )
        
        output_path = os.path.join("output/videos", f"{env_name}_{algo}.mp4")
        rollout_and_stream_video(
            env, model, env_name, algo,
            steps_per_traj=VIDEO_NUM_STEPS[env_name],
            num_traj=VIDEO_NUM_TRAJ[env_name],
            output_path=output_path,
            fps=15
        )
"""
# ------------------------------------------------------------------------------
# 6. Optional: line‑plots of val(t) like before
# ------------------------------------------------------------------------------
TIMESTEPS = {
    'shape_boundary': np.linspace(0, 1, NUM_STEPS['shape_boundary']),
    'shape':          np.linspace(0, 1, NUM_STEPS['shape']),
    'molecule':       np.linspace(0, 1, NUM_STEPS['molecule'])
}


for env_name, disp in display_names.items():
    plot_eval_benchmarks(
        eval_history[env_name],
        time_steps=TIMESTEPS[env_name],
        title=f"Benchmark-2 on {disp}",
        mode='bootstrap',
        plot_dir=f"benchmark2_{env_name}.png"
    )

elapsed = (time.time() - start) / 3600
print(f"\nFinished Benchmark 2 in {elapsed:.2f} hours")
