import os
import json
import wandb
wandb.require("core") #Bunch of improvements in core.
import torch
import random
import numpy as np
from datetime import datetime
from ppo_alg import PPO, Memory
from config import get_config
from design_env import DesignEnv
from control_env import ControlEnv
import torch.multiprocessing as mp
from wandb_sweep import HyperParameterTuner
from config import classify_and_return_args
from torch.utils.tensorboard import SummaryWriter

def save_config(config, SEED, save_path):
    """
    Save hyperparameters to json.
    """
    config_to_save = {
        "hyperparameters": config,
        "global_seed": SEED,
        }
    with open(save_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)

def train(train_config, is_sweep=False, sweep_config=None):
    """
    High level training orchestration.
    The control agent actors are parallelized. However, all aspects of training in both agents are centralized.
    """

    SEED = train_config['seed'] if train_config['seed'] else random.randint(0, 1000000)
    print(f"Random seed: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    worker_device = torch.device("cuda") if train_config['gpu'] and torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {worker_device}")
    design_args, control_args, lower_ppo_args, higher_ppo_args = classify_and_return_args(train_config, worker_device)

    # Before setup, print stats from dummy environments.
    dummy_envs = {
        'lower': ControlEnv(control_args, worker_id=None), # This is not a standard way to access control env (has to go through control agent). This is only for setup.
        'higher': DesignEnv(design_args, control_args, lower_ppo_args)
        }
    
    for env_type, env in dummy_envs.items():
        print(f"\nEnvironment for {env_type} level agent:")
        print(f"\tDefined observation space: {env.observation_space}")
        print(f"\tObservation space shape: {env.observation_space.shape}")
        print(f"\tDefined action space: {env.action_space}")
        if env_type == 'lower':
            print(f"\tOptions per action dimension: {env.action_space.nvec}")
        elif env_type == 'higher':
            print(f"\tNumber of proposals: {env.action_space['num_proposals'].n}")
            print(f"\tProposal space: {env.action_space['proposals']}")
    
    # Dont need these anymore
    dummy_envs['lower'].close() 
    dummy_envs['higher'].close()

    # Actual agents
    print(f"\nHigher level agent: \n\tIn channels: {train_config['higher_in_channels']}, Action dimension: {train_config['max_proposals']}\n")
    print(f"\nLower level agent: \n\tState dimension: {dummy_envs['lower'].observation_space.shape}, Action dimension: {train_config['lower_action_dim']}")

    higher_ppo = PPO(**higher_ppo_args)

    # TensorBoard setup
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time)
    os.makedirs('runs', exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Save hyperparameters 
    config_path = os.path.join(log_dir, f'config_{current_time}.json')
    save_config(train_config, SEED, config_path)
    print(f"Configuration saved to {config_path}")

    # Model saving setup
    save_dir = os.path.join('saved_models', current_time)
    os.makedirs(save_dir, exist_ok=True)
    best_reward_higher = float('-inf')

    control_args.update({'writer': writer})
    control_args.update({'save_dir': save_dir})
    control_args.update({'global_seed': SEED})
    control_args.update({'total_action_timesteps_per_episode': train_config['max_timesteps'] // train_config['action_duration']})

    # Instead of using total_episodes, we will use total_iterations. 
    # Every iteration, num_process lower level agents interact with the environment for total_action_timesteps_per_episode steps (which further internally contains action_duration steps)
    # Each iteration is equivalent to a single timestep for the higher agent.
    total_iterations = train_config['total_timesteps'] // (train_config['max_timesteps'] * train_config['lower_num_processes'])
    global_step = 0

    # Initialize higher level environment and get initial state
    higher_env = DesignEnv(design_args, control_args, lower_ppo_args, is_sweep=is_sweep, is_eval=False)
    higher_env.lower_ppo.total_iterations = total_iterations # For lr annealing
    higher_state = higher_env.reset() # state includes batch.
    higher_memory = Memory()

    for iteration in range(1, total_iterations + 1): # Starting from 1 to prevent policy update in the very first iteration.
        
        global_step = iteration * train_config['lower_num_processes']*control_args['total_action_timesteps_per_episode']*train_config['action_duration']
        print(f"\nStarting iteration: {iteration}/{total_iterations} with {global_step} total steps so far\n")
        #print(f"Higher state: {higher_state}")

        # Higher level agent takes node features, edge index, edge attributes and batch (to make single large graph) as input 
        # To produce padded fixed-sized actions num_actual_proposals is also returned.
        higher_action, num_actual_proposals, higher_logprob = higher_ppo.policy_old.act(higher_state, iteration, visualize=True) 

        # Since the higher agent internally takes a step where a number of parallel lower agents take their own steps, 
        # We return things relevant to both the higher and lower agents. First, for higher.
        higher_next_state, higher_reward, higher_done, higher_info = higher_env.step(higher_action, num_actual_proposals, iteration, global_step)
        higher_memory.append(higher_state, higher_action, higher_logprob, higher_reward, higher_done)

        if iteration % train_config['higher_update_freq'] == 0:
            higher_ppo.update(higher_memory, agent_type='higher')
            higher_memory.clear_memory()

        higher_state = higher_next_state

        # Log higher level agent stuff.
        if is_sweep:
            wandb.log({    "higher_avg_reward": higher_reward,
                            "global_step": global_step          })
        else:
            writer.add_scalar('Higher/Average_Reward', higher_reward, global_step)
    
    if is_sweep:
        wandb.finish()
    else:
        writer.close()

def evaluate(config, design_env):
    """
    Evaluate "RL agents (design + control)" vs "real-world (original design + TL)".
    TODO: Make the evaluation run N number of times each with different seed. 
    """
    pass

def calculate_performance(run_data,):
    """
    Evaluation generates run_data files. Calculate performance metrics.
    Average results over N runs.
    """
    pass

def main(config):
    """
    Cannot create a bunch of connections in main and then pass them around. 
    Because each new worker needs a separate pedestrian and vehicle trips file.
    """

    # Set the start method for multiprocessing. It does not create a process itself but sets the method for creating a process.
    # Spawn means create a new process. There is a fork method as well which will create a copy of the current process.
    mp.set_start_method('spawn') 
    mp.set_sharing_strategy('file_system')

    if config['evaluate']: 
        if config['manual_demand_veh'] is None or config['manual_demand_ped'] is None:
            print("Manual demand is None. Please specify a demand for both vehicles and pedestrians.")
            return None
        else: 
            env = DesignEnv(config, is_eval=True) # Initialize the design environment with default design.
            run_data = evaluate(config, env)
            calculate_performance(run_data)
            env.close()

    elif config['sweep']:
        tuner = HyperParameterTuner(config)
        tuner.start()

    else:
        train(config)  

if __name__ == "__main__":
    config = get_config()
    main(config)