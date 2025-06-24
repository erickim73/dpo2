import numpy as np
from envs.shape_boundary import ShapeBoundary

def eval_config(alpha_ctrl, alpha_knot, alpha_weight, alpha_repulsion, episodes=5):
    rewards = []
    for _ in range(episodes):
        env = ShapeBoundary(
            naive=False,              # Use shaped reward, not just raw negative value
            step_size=5e-2,           # How much an action perturbs the spline
            ctrl_state_dim=36,        # 18 control points × 2D = 36-dimensional state
            max_num_step=150,         # Total number of steps to run the simulation
            render_mode="rgb_array",  # Output images (used for making frames)
            degree=3,                 # Use cubic splines (degree 3)
            n_internal_knots=14,      # Number of internal knots controls spline flexibility
            train_ctrl=True,          # Train control points
            train_weight=True,        # Train weights
            train_knot=True,          # Train knot positions
            alpha_ctrl=0.4,       # give geometry a strong pull
            alpha_knot=0.2,       # moderate knot alignment
            alpha_weight=0.2,     # moderate weight alignment
            alpha_repulsion=0.2,  # match repulsion scale to geometry
            alpha_vel=0.1,
            alpha_energy=0.05,
        )
    
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, done, *_ = env.step(action)
            total += r
        rewards.append(total)
    return np.mean(rewards)

if __name__ == "__main__":
    # simple random search
    best = (-1e9, None)
    for _ in range(30):
        cfg = np.random.rand(4)  # values in [0,1]
        mean_r = eval_config(*cfg)
        if mean_r > best[0]:
            best = (mean_r, cfg)
    print("Best mean reward", best[0], "with alphas", best[1])
