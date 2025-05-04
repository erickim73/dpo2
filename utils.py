import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import torch
import pyrosetta
from envs import ShapeBoundary, Shape, Molecule
from common_nets import Mlp
from policy import Policy

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get correct environment
def get_environment(env_name):
    if env_name == 'naive_shape_boundary':
        return ShapeBoundary(naive=True, render_mode='rgb_array')
    if env_name == 'shape_boundary':
        return ShapeBoundary(render_mode='rgb_array')
    if env_name == 'naive_shape':
        return Shape(naive=True, render_mode='rgb_array')
    if env_name == 'shape':
        return Shape(render_mode='rgb_array')
    pose = pyrosetta.pose_from_sequence('A'*8)
    # ('TTCCPSIVARSNFNVCRLPGTSEAICATYTGCIIIPGATCPGDYAN')
    # pyrosetta.pose_from_pdb("molecule_files/1AB1.pdb") #pyrosetta.pose_from_sequence('A' * 10)
    if env_name == 'naive_molecule':
        return Molecule(pose=pose, naive=True, render_mode='rgb_array')
    if env_name == 'molecule':
        return Molecule(pose=pose, render_mode='rgb_array')

def from_str_to_2D_arr(s):
    tokens = s[2:-2].split("],[")
    ans = []
    for arr_str in tokens:
        arr = []
        elem_str = arr_str.split(",")
        for e in elem_str:
            arr.append(int(e))
        ans.append(arr)
    return ans

def str_to_list(s):
    tokens = s[1:-1].split(",")
    ans = []
    for token in tokens:
        ans.append(int(token))
    return ans

# Get neural net architecture
def get_architectures(env_name, zero_order, arch_file='arch.csv'):
    # Get architecture info from arch_file
    df = pd.read_csv(arch_file)
    net_info = df[df['env_name']==env_name]
    if zero_order:
        layer_dims = str_to_list(net_info['derivative_layer_dims'].values[0])
    else:
        layer_dims = str_to_list(net_info['val_layer_dims'].values[0])

    return layer_dims

# Get predefined training parameters from file for a specific environment. 
def get_train_params(env_name, param_file='params.csv'):
    # Get parameter info from param file
    df = pd.read_csv(param_file)
    info = df[df['env_name']==env_name]

    # Rate underlying hamiltonian dynamics formula.
    rate = float(info['rate'].values[0])

    # Number of trajectories per stage
    num_traj = int(info['num_traj'].values[0])
    
    # Step size for discretized ODE
    step_size = float(info['step_size'].values[0])
    
    # Optimization params: learning rate, batch size, how often logging
    # and number of optimization steps per each sampling stage.
    lr = float(info['lr'].values[0])
    batch_size = int(info['batch_size'].values[0])
    log_interval = int(info['log_interval'].values[0])

    return rate, num_traj, step_size, lr, batch_size, log_interval

def setup_main_net(env_name, zero_order, state_dim):
    layer_dims = get_architectures(env_name, zero_order)
    if zero_order:
        output_dim = 1
    else:
        output_dim = state_dim
    main_net = Mlp(input_dim=state_dim, output_dim=output_dim, 
                    layer_dims=layer_dims, activation='relu').to(DEVICE)
    
    return main_net


# Convert e.g. algo = "DPO_zero_order" to match your actual filename suffix
def setup_dpo_model(algo, env, env_name):
    import os
    import torch
    from policy import Policy

    # 🛠 Patch here
    if algo == "DPO":
        algo = "DPO_zero_order"

    path = os.path.join("models", f"{env_name}_{algo}.pth")

    zero_order = 'zero_order' in algo
    input_dim = env.observation_space.shape[0]
    main_net = setup_main_net(env_name, zero_order, input_dim)

    main_net.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    main_net.eval()

    model = Policy(env, main_net)
    return model




# Plotting
def _bootstrap(data, n_boot=2000, ci=68, random_state=42):
    rng = np.random.default_rng(random_state)
    boot_dist = []
    for _ in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return (s1,s2)

def _tsplot(ax, x, data, mode='bootstrap', label=None, color=None):
    est = np.mean(data, axis=0)
    if mode == 'bootstrap':
        cis = _bootstrap(data)
    else:
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.15, color=color)
    line = ax.plot(x, est, label=label, color=color)[0]
    ax.margins(x=0)
    ax.grid(True, linestyle='--', alpha=0.4)
    return line

def plot_eval_benchmarks(eval_dict, time_steps, title, mode='bootstrap', 
                         colors=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'olive'],
                         plot_dir='tmp.png'):
    methods = list(eval_dict.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))  # wider layout
    for i, method in enumerate(methods):
        data = eval_dict[method]
        color = colors[i % len(colors)]
        _tsplot(ax, np.array(time_steps), data, mode, label=method, color=color)

    ax.legend(frameon=False, bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.subplots_adjust(right=0.75)
    ax.set_title(title)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Evaluation cost')
    # ---- cosmetic axis tweaks ----
    if 'Molecular dynamics' in title:
        ax.set_yscale('log')
    elif 'Topological materials' in title:      # shape task stays 5‑10
        ax.set_ylim(5, 10)
    elif 'Materials deformation' in title:      # shape‑boundary needs larger window
        ax.set_ylim(12, 20)        # or use auto‑range and clip spike instead
    plt.tight_layout()
    plt.savefig(f'output/{plot_dir}', dpi=140)
    plt.close()