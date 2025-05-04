import os, sys
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from sb3_contrib import TRPO
from sb3_contrib import CrossQ, TQC

# current directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
# path to main repo (locally)
sys.path.append(os.path.dirname(cur_dir))
from utils import get_environment, str_to_list, DEVICE

def get_RL_nets_architectures(env_name, on_policy=True):
    # Get architecture info from arch_file
    df = pd.read_csv('arch.csv')
    net_info = df[df['env_name']==env_name]
    actor_dims = str_to_list(net_info['actor_dims'].values[0])
    critic_dims_field = 'v_critic_dims' if on_policy else 'q_critic_dims'
    critic_dims = str_to_list(net_info[critic_dims_field].values[0])
    return actor_dims, critic_dims

def train_benchmark_model(method, gamma, env_name, total_samples, common_dims=[], 
          activation='ReLU', lr=3e-3, log_interval=10):
    # On-policy mean optimize policy directly using current policy being optimized.
    on_policy = method in set(['PPO', 'TRPO', 'A2C'])
    # off_policy = set('SAC', 'DDPG', 'TD3')

    # Construct environment
    env = get_environment(env_name)
    env.set_gamma(gamma) 
    actor_dims, critic_dims = get_RL_nets_architectures(env_name, on_policy=on_policy)

    # Net architecture for actor and critic networks
    if on_policy:
        net_arch_dict = dict(pi=actor_dims, vf=critic_dims)
    else:
        net_arch_dict = dict(pi=actor_dims, qf=critic_dims)
    
    # Add common processing nets from state to both actor & critic.
    if len(common_dims) != 0:
        net_arch = []
        for dim in common_dims:
            net_arch.append(dim)
        net_arch.append(net_arch_dict)
    else:
        net_arch = net_arch_dict
        
    # Set the policy args
    activation_fn = torch.nn.ReLU if activation == 'ReLU' else torch.nn.Tanh
    policy_kwargs = dict(activation_fn=activation_fn,
                         net_arch=net_arch)
    
    # Build the model using SB3.
    if method == 'TRPO':
        model = TRPO("MlpPolicy", env, learning_rate=lr, gamma=gamma, policy_kwargs=policy_kwargs, verbose=1)
    elif method == 'PPO':
        model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma, policy_kwargs=policy_kwargs, verbose=1)
    elif method == 'SAC':
        model = SAC("MlpPolicy", env, learning_rate=lr, gamma=gamma, policy_kwargs=policy_kwargs, verbose=1)
    elif method == 'DDPG':
        n_actions = env.action_space.shape[-1]
        scale = env.max_act
        #action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=(scale**2) * np.ones(n_actions))
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=(scale**2)*np.ones(n_actions))
        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    elif method == 'CrossQ':
        model = CrossQ("MlpPolicy", env, learning_rate=lr, gamma=gamma, policy_kwargs=policy_kwargs, verbose=1)
    elif method == 'TQC':
        policy_kwargs.update({'n_critics': 5, 'n_quantiles': 10})
        model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs)
    model.device = DEVICE

    # Train and save model with SB3.
    model.learn(total_timesteps=total_samples, log_interval=log_interval)
    model_path ="models/" + env_name + '_' + method + '_' + str(gamma).replace('.', '_')
    model.save(model_path)

def _load_benchmark_model(method, model_path):
    if 'TRPO' in method:
        model = TRPO.load(model_path)
    elif 'PPO' in method:
        model = PPO.load(model_path)
    elif 'SAC' in method:
        model = SAC.load(model_path)
    elif 'DDPG' in method:
        model = DDPG.load(model_path)
    elif 'CrossQ' in method:
        model = CrossQ.load(model_path)
    elif 'TQC' in method:
        model = TQC.load(model_path)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return model

def setup_benchmark_model(method, env, model_path):
    model = _load_benchmark_model(method, model_path)
    model.set_env(env)

    return model