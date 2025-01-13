import os
import json
import wandb
wandb.require("core")
import torch
import random
import numpy as np
from datetime import datetime
from ppo import PPO, Memory
from config import get_config
from env import ControlEnv
import torch.multiprocessing as mp
from wandb_sweep import HyperParameterTuner
from config import classify_and_return_args
from torch.utils.tensorboard import SummaryWriter

from utils import *
from models import CNNActorCritic

def parallel_worker(rank, control_args, model_init_params, policy_old_dict, memory_queue, global_seed, worker_device, network_iteration):
    """
    At every iteration, a number of workers will each parallelly carry out one episode in control environment.
    - Worker environment runs in CPU (SUMO runs in CPU).
    - Worker policy inference runs in GPU.
    - memory_queue is used to store the memory of each worker and send it back to the main process.
    - A shared policy_old (dict copy passed here) is used for importance sampling.
    """

    shared_policy_old = CNNActorCritic(model_init_params['model_dim'], model_init_params['action_dim'], **model_init_params['kwargs'])
    shared_policy_old.load_state_dict(policy_old_dict)

    # Set seed for this worker
    worker_seed = global_seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    lower_env = ControlEnv(control_args, worker_id=rank, network_iteration=network_iteration)
    memory_transfer_freq = control_args['memory_transfer_freq']  # Get from config

    # The central memory is a collection of memories from all processes.
    # A worker instance must have their own memory 
    local_memory = Memory()
    shared_policy_old = shared_policy_old.to(worker_device)

    state, _ = lower_env.reset()
    ep_reward = 0
    steps_since_update = 0
    
    for _ in range(control_args['total_action_timesteps_per_episode']):
        state_tensor = torch.FloatTensor(state).to(worker_device)

        # Select action
        with torch.no_grad():
            action, logprob = shared_policy_old.act(state_tensor)
            action = action.cpu()  # Explicitly Move to CPU, Incase they were on GPU
            logprob = logprob.cpu() 

        print(f"\nAction: in worker {rank}: {action}")
        # Perform action
        # These reward and next_state are for the action_duration timesteps.
        next_state, reward, done, truncated, info = lower_env.step(action)
        ep_reward += reward

        # Store data in memory
        local_memory.append(torch.FloatTensor(state), action, logprob, reward, done)
        steps_since_update += 1

        if steps_since_update >= memory_transfer_freq or done or truncated:
            # Put local memory in the queue for the main process to collect
            memory_queue.put((rank, local_memory))
            local_memory = Memory()  # Reset local memory
            steps_since_update = 0

        if done or truncated:
            break

        state = next_state.flatten()

    # In PPO, we do not make use of the total reward. We only use the rewards collected in the memory.
    print(f"Worker {rank} finished. Total reward: {ep_reward}")
    lower_env.close()
    memory_queue.put((rank, None))  # Signal that this worker is done

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
    control_args, ppo_args = classify_and_return_args(train_config, worker_device)

    # Print stats from dummy environment
    dummy_env = ControlEnv(control_args, worker_id=None)
    print(f"\nEnvironment for control agent:")
    print(f"\tDefined observation space: {dummy_env.observation_space}")
    print(f"\tObservation space shape: {dummy_env.observation_space.shape}")
    print(f"\tDefined action space: {dummy_env.action_space}")
    print(f"\tOptions per action dimension: {dummy_env.action_space.nvec}")
    dummy_env.close()

    # Initialize control agent
    print(f"\nControl agent: \n\tState dimension: {dummy_env.observation_space.shape}, Action dimension: {train_config['action_dim']}")
    control_ppo = PPO(**ppo_args)

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
    control_args.update({'writer': writer})
    control_args.update({'save_dir': save_dir})
    control_args.update({'global_seed': SEED})
    control_args.update({'total_action_timesteps_per_episode': train_config['max_timesteps'] // train_config['action_duration']})

    # worker related args
    control_args_worker = {k: v for k, v in control_args.items() if k != 'writer'} # bug fix. writer is unpicklable
    model_init_params_worker = {'model_dim': control_ppo.policy.in_channels,
                                    'action_dim': control_ppo.action_dim,
                                    'kwargs': ppo_args['model_kwargs']}

    # Instead of using total_episodes, we will use total_iterations. 
    # Every iteration, num_process control agents interact with the environment for total_action_timesteps_per_episode steps (which further internally contains action_duration steps)
    total_iterations = train_config['total_timesteps'] // (train_config['max_timesteps'] * train_config['num_processes'])
    global_step = 0
    action_timesteps = 0

    for iteration in range(1, total_iterations + 1): # Starting from 1 to prevent policy update in the very first iteration.
        
        global_step = iteration * train_config['num_processes']*control_args['total_action_timesteps_per_episode']*train_config['action_duration']
        print(f"\nStarting iteration: {iteration}/{total_iterations} with {global_step} total steps so far\n")

        reward, done, info = 0, False, {}
        manager = mp.Manager()
        memory_queue = manager.Queue()
        processes = []
        
        for rank in range(control_args['num_processes']):
            p = mp.Process(
                target=parallel_worker,
                args=(
                    rank,
                    control_args_worker,
                    model_init_params_worker,
                    control_ppo.policy_old.state_dict(),
                    memory_queue,
                    control_args['global_seed'],
                    worker_device,
                    iteration
                )
            )
            p.start()
            processes.append(p)

        if control_args['anneal_lr']:
            current_lr = control_ppo.update_learning_rate(iteration)

        all_memories = []
        active_workers = set(range(control_args['num_processes']))

        while active_workers:
            try:
                rank, memory = memory_queue.get(timeout=60) # Add a timeout to prevent infinite waiting

                if memory is None:
                    active_workers.remove(rank)
                else:
                    all_memories.append(memory)
                    print(f"Memory from worker {rank} received. Memory size: {len(memory.states)}")

                    action_timesteps += len(memory.states)
                    # Update lower level PPO every n times action has been taken
                    if action_timesteps % control_args['update_freq'] == 0:
                        loss = control_ppo.update(all_memories, agent_type='lower')

                        total_lower_reward = sum(sum(memory.rewards) for memory in all_memories)
                        avg_lower_reward = total_lower_reward / control_args['num_processes'] # Average reward per process in this iteration
                        print(f"\nAverage Reward per process: {avg_lower_reward:.2f}\n")
                        
                        # clear memory to prevent memory growth (after the reward calculation)
                        for memory in all_memories:
                            memory.clear_memory()

                        # reset all memories
                        del all_memories #https://pytorch.org/docs/stable/multiprocessing.html
                        all_memories = []

                        # logging after update
                        if loss is not None:
                            if is_sweep: # Wandb for hyperparameter tuning
                                wandb.log({     "iteration": iteration,
                                                "lower_avg_reward": avg_lower_reward, # Set as maximize in the sweep config
                                                "lower_policy_loss": loss['policy_loss'],
                                                "lower_value_loss": loss['value_loss'], 
                                                "lower_entropy_loss": loss['entropy_loss'],
                                                "lower_total_loss": loss['total_loss'],
                                                "lower_current_lr": current_lr if control_args['anneal_lr'] else control_args['lr'],
                                                "global_step": global_step          })
                                
                            else: # Tensorboard for regular training
                                total_updates = int(action_timesteps / control_args['update_freq'])
                                writer.add_scalar('Lower/Average_Reward', avg_lower_reward, global_step)
                                writer.add_scalar('Lower/Total_Policy_Updates', total_updates, global_step)
                                writer.add_scalar('Lower/Policy_Loss', loss['policy_loss'], global_step)
                                writer.add_scalar('Lower/Value_Loss', loss['value_loss'], global_step)
                                writer.add_scalar('Lower/Entropy_Loss', loss['entropy_loss'], global_step)
                                writer.add_scalar('Lower/Total_Loss', loss['total_loss'], global_step)
                                writer.add_scalar('Lower/Current_LR', current_lr, global_step)
                                print(f"Logged lower agent data at step {global_step}")

                                # Save model every n times it has been updated (may not every iteration)
                                if control_args['save_freq'] > 0 and total_updates % control_args['save_freq'] == 0:
                                    torch.save(control_ppo.policy.state_dict(), os.path.join(control_args['save_dir'], f'control_model_iteration_{iteration+1}.pth'))

                                # Save best model so far
                                if avg_lower_reward > best_reward_lower:
                                    torch.save(control_ppo.policy.state_dict(), os.path.join(control_args['save_dir'], 'best_control_model.pth'))
                                    best_reward_lower = avg_lower_reward
                        
                        else: # For some reason..
                            print("Warning: loss is None")

            except queue.Empty:
                print("Timeout waiting for worker. Continuing...")
        
        # At the end of an iteration, wait for all processes to finish
        # The join() method is called on each process in the processes list. This ensures that the main program waits for all processes to complete before continuing.
        for p in processes:
            p.join()

        # Log higher level agent stuff.
        if is_sweep:
            wandb.log({
                "control_avg_reward": reward,
                "global_step": global_step
            })
        else:
            writer.add_scalar('Average_Reward', reward, global_step)
    
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
            # env = ControlEnv(config, is_eval=True) 
            # run_data = evaluate(config, env)
            # calculate_performance(run_data)
            # env.close()
            pass

    elif config['sweep']:
        tuner = HyperParameterTuner(config)
        tuner.start()
    else:
        train(config)

if __name__ == "__main__":
    config = get_config()
    main(config)