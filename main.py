import os
import json
import time
import wandb
import torch
import random
import numpy as np
from copy import deepcopy
from datetime import datetime
from ppo import PPO, Memory
from config import get_config
import torch.multiprocessing as mp

from wandb_sweep import HyperParameterTuner
from config import classify_and_return_args
from torch.utils.tensorboard import SummaryWriter

from utils import *
from env import ControlEnv

def parallel_train_worker(rank, shared_policy_old, control_args, queue, global_seed, worker_device):
    """
    At every iteration, a number of workers will each parallelly carry out one episode in control environment.
    - Worker environment runs in CPU (SUMO runs in CPU).
    - Worker policy inference runs in GPU.
    - memory_queue is used to store the memory of each worker and send it back to the main process.
    - A shared policy_old (dict copy passed here) is used for importance sampling.
    - 1 memory transfer happens every memory_transfer_freq * action_duration sim steps.
    """

    # Set seed for this worker
    worker_seed = global_seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    worker_env = ControlEnv(control_args, worker_id=rank)
    memory_transfer_freq = control_args['memory_transfer_freq']  # Get from config
    local_memory = Memory() # A worker instance must have their own memory 

    state, _ = worker_env.reset()
    ep_reward = 0
    steps_since_update = 0
    
    for _ in range(control_args['total_action_timesteps_per_episode']):
        state = torch.FloatTensor(state)
        # Select action
        with torch.no_grad():
            action, logprob = shared_policy_old.act(state.to(worker_device)) # sim runs in CPU, state will initially always be in CPU.
            state = state.detach().cpu()
            action = action.detach().cpu()
            logprob = logprob.detach().cpu()

        # Perform action
        # These reward and next_state are for the action_duration timesteps.
        next_state, reward, done, truncated, _ = worker_env.step(action) # need the returned state to be 2D
        ep_reward += reward

        # Store data in memory
        local_memory.append(state, action, logprob, reward, done) 
        steps_since_update += 1

        if steps_since_update >= memory_transfer_freq or done or truncated:
            # Put local memory in the queue for the main process to collect
            memory_copy = deepcopy(local_memory)
            queue.put((rank, memory_copy))
            local_memory = Memory()  # Reset local memory
            steps_since_update = 0

        state = next_state
        if done or truncated:
            break

    # In PPO, we do not make use of the total reward. We only use the rewards collected in the memory.
    print(f"Worker {rank} finished. Total reward: {ep_reward}")
    worker_env.close()
    time.sleep(10) # Essential
    del worker_env
    queue.put((rank, None))  # Signal that this worker is done

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

    # Set and save hyperparameters 
    if is_sweep:
        for key, value in sweep_config.items():
            train_config[key] = value

    os.makedirs('runs', exist_ok=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(log_dir, f'config_{current_time}.json')
    save_config(train_config, SEED, config_path)
    print(f"Configuration saved to {config_path}")

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
    control_ppo.policy.share_memory() # share across processes
    control_ppo.policy_old.share_memory() # Since workers share the old policy (used for importance sampling), they get the new one after each update. Policy is not stale. Verify.

    # Model saving and tensorboard 
    writer = SummaryWriter(log_dir=log_dir)
    save_dir = os.path.join('saved_models', current_time)

    os.makedirs(save_dir, exist_ok=True)
    control_args.update({
        'writer': writer,
        'save_dir': save_dir,
        'global_seed': SEED,
        'total_action_timesteps_per_episode': train_config['max_timesteps'] // train_config['action_duration']
    })
    # worker related args
    control_args_worker = {k: v for k, v in control_args.items() if k != 'writer'} # bug fix. writer is unpicklable
    
    # Instead of using total_episodes, we will use total_iterations. 
    # Every iteration, num_process control agents interact with the environment for total_action_timesteps_per_episode steps (which further internally contains action_duration steps)
    total_iterations = train_config['total_timesteps'] // (train_config['max_timesteps'] * train_config['num_processes'])
    control_ppo.total_iterations = total_iterations
    
    global_step = 0
    total_updates = 0
    action_timesteps = 0
    best_reward = float('-inf')

    all_memories = Memory()
    for iteration in range(0, total_iterations): # Starting from 1 to prevent policy update in the very first iteration.
        print(f"\nStarting iteration: {iteration + 1}/{total_iterations} with {global_step} total steps so far\n")

        #print(f"Shared policy weights: {control_ppo.policy_old.state_dict()}")
        queue = mp.Queue()
        processes = []
        active_workers = []
        for rank in range(control_args['num_processes']):
            p = mp.Process(
                target=parallel_train_worker,
                args=(
                    rank,
                    control_ppo.policy_old,
                    control_args_worker,
                    queue,
                    control_args['global_seed'],
                    worker_device)
                )
            p.start()
            processes.append(p)
            active_workers.append(rank)

        if control_args['anneal_lr']:
            current_lr = control_ppo.update_learning_rate(iteration)
        
        while active_workers:
            print(f"Active workers: {active_workers}")

            rank, memory = queue.get() #timeout=60) # Add a timeout to prevent infinite waiting
            if memory is None:
                print(f"Worker {rank} finished")
                active_workers.remove(rank)
            else:
                current_action_timesteps = len(memory.states)
                print(f"Memory from worker {rank} received. Memory size: {current_action_timesteps}")
                all_memories.actions.extend(memory.actions)
                all_memories.states.extend(memory.states)
                all_memories.logprobs.extend(memory.logprobs)
                all_memories.rewards.extend(memory.rewards)
                all_memories.is_terminals.extend(memory.is_terminals)

                action_timesteps += current_action_timesteps
                global_step += current_action_timesteps * train_config['action_duration'] 
                print(f"Action timesteps: {action_timesteps}, global step: {global_step}")
                del memory #https://pytorch.org/docs/stable/multiprocessing.html

                # Update PPO every n times (or close) action has been taken 
                if action_timesteps >= control_args['update_freq']:
                    total_updates += 1
                    print(f"Updating PPO with {len(all_memories.actions)} memories")

                    # Un-normalized reward is useful to track performance whereas normalized reward is used for training (stability and convergence).
                    # Compute average reward (un-normalized) across all processes (various demands) for logging and maximization in sweep (this is directly interpretable and indicative with the traffic performance we want)
                    avg_reward = sum(all_memories.rewards) / control_args['num_processes'] # Averaged across processes.
                    print(f"\nAverage Reward per process (unnormalized): {avg_reward:.2f}\n")
                    print(f"\nAll memories rewards: {all_memories.rewards}")

                    loss = control_ppo.update(deepcopy(all_memories))
        
                    # Reset all memories
                    del all_memories
                    all_memories = Memory() 
                    action_timesteps = 0
                    print(f"Size of all memories after update: {len(all_memories.actions)}")

                    # logging
                    if is_sweep: # Wandb for hyperparameter tuning
                        wandb.log({ "iteration": iteration,
                                        "avg_reward": avg_reward, # Set as maximize in the sweep config
                                        "total_updates": total_updates,
                                        "policy_loss": loss['policy_loss'],
                                        "value_loss": loss['value_loss'], 
                                        "entropy_loss": loss['entropy_loss'],
                                        "total_loss": loss['total_loss'],
                                        "current_lr": current_lr if control_args['anneal_lr'] else ppo_args['lr'],
                                        "global_step": global_step          })
                        
                    else: # Tensorboard for regular training
                        writer.add_scalar('Training/Average_Reward', avg_reward, global_step)
                        writer.add_scalar('Training/Total_Policy_Updates', total_updates, global_step)
                        writer.add_scalar('Training/Policy_Loss', loss['policy_loss'], global_step)
                        writer.add_scalar('Training/Value_Loss', loss['value_loss'], global_step)
                        writer.add_scalar('Training/Entropy_Loss', loss['entropy_loss'], global_step)
                        writer.add_scalar('Training/Total_Loss', loss['total_loss'], global_step)
                        writer.add_scalar('Training/Current_LR', current_lr if control_args['anneal_lr'] else ppo_args['lr'], global_step)

                        # Save model every n times it has been updated (may not every iteration)
                        if control_args['save_freq'] > 0 and total_updates % control_args['save_freq'] == 0:
                            torch.save(control_ppo.policy.state_dict(), os.path.join(control_args['save_dir'], f'control_model_iteration_{iteration+1}.pth'))

                        # Save best model so far
                        if avg_reward > best_reward:
                            torch.save(control_ppo.policy.state_dict(), os.path.join(control_args['save_dir'], 'best_control_model.pth'))
                            best_reward = avg_reward
                    print(f"Logged agent data at step {global_step}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Clean up. The join() method ensures that the main program waits for all processes to complete before continuing.
        for p in processes:
            p.join() 
        print(f"All processes joined\n\n")

    if not is_sweep:
        writer.close()

def parallel_eval_worker(rank, eval_worker_config, queue):
    """
    - For the same demand, each worker runs n_iterations number of episodes and measures performance metrics at each iteration.
    - Each episode runs on a different random seed.
    - Performance metrics: 
        - Average waiting time (Veh, Ped)
        - Average travel time (Veh, Ped)
    - Returns a dictionary with performance metrics in all iterations.
    - For PPO: 
        - Each worker create a copy of the policy and run it
    """
    ppo_args = eval_worker_config['policy_args']
    eval_control_ppo = PPO(**ppo_args)
    eval_control_ppo.policy.load_state_dict(torch.load(eval_worker_config['policy_path']))
    
    worker_demand_scale = eval_worker_config['worker_demand_scale']
    control_args = eval_worker_config['control_args']

    # We set the demand manually (so that automatic scaling does not happen)
    control_args['manual_demand_veh'] = worker_demand_scale
    control_args['manual_demand_ped'] = worker_demand_scale
    env = ControlEnv(control_args, worker_id=rank)

    worker_result = {}
    worker_result['demand_scale'] = worker_demand_scale

    # Run the worker
    for i in range(eval_worker_config['n_iterations']):
        worker_result[i] = {}
        SEED = random.randint(0, 1000000)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        worker_result[i]['SEED'] = SEED

        # Run the worker
        state, _ = env.reset()
        veh_waiting_time_this_episode = 0
        ped_waiting_time_this_episode = 0
        veh_unique_ids_this_episode = 0
        ped_unique_ids_this_episode = 0

        with torch.no_grad():
            for _ in range(eval_worker_config['total_action_timesteps_per_episode']):
            

                state = torch.FloatTensor(state).to(ppo_args['device'])
                action, _ = eval_control_ppo.policy.act(state)
                action = action.detach().cpu() # sim runs in CPU
                state, reward, done, truncated, _ = env.step(action)

                # During this step, get all vehicles and pedestrians
                veh_waiting_time_this_step = env.get_vehicle_waiting_time()
                ped_waiting_time_this_step = env.get_pedestrian_waiting_time()

                veh_waiting_time_this_episode += veh_waiting_time_this_step
                ped_waiting_time_this_episode += ped_waiting_time_this_step

                veh_unique_ids_this_episode, ped_unique_ids_this_episode = env.total_unique_ids()

        # gather performance metrics
        worker_result[i]['veh_avg_waiting_time'] = veh_waiting_time_this_episode / veh_unique_ids_this_episode
        worker_result[i]['ped_avg_waiting_time'] = ped_waiting_time_this_episode / ped_unique_ids_this_episode
    # After all iterations are complete. 
    queue.put((rank, worker_result))

def eval(config):
    """
    Evaluate RL agent vs real-world TL
    - Each demand is run on a different worker
    - First eval trained ppo policy, then TL
    - Results are put in a json and saved

    """
    n_workers = config['eval_n_workers']
    eval_worker_device = config['eval_worker_device']
    n_iterations = config['eval_n_iterations']

    policy_path = config['eval_model_path']
    eval_device = torch.device("cuda") if config['gpu'] and torch.cuda.is_available() else torch.device("cpu")
    control_args, ppo_args = classify_and_return_args(config, eval_device)
    eval_demand_scales = config['eval_demand_scales']

    # PPO
    # number of times the n_workers have to be repeated to cover all eval demands
    num_times_workers_recycle = len(eval_demand_scales) if len(eval_demand_scales) < n_workers else (len(eval_demand_scales) // n_workers) + 1
    for i in range(num_times_workers_recycle):
        start = n_workers * i   
        end = n_workers * (i + 1)
        demand_scales_evaluated_current_cycle = eval_demand_scales[start: end]

        queue = mp.Queue()
        processes = []  
        active_workers = []
        for j, demand_scale in enumerate(demand_scales_evaluated_current_cycle): 
            print(f"For demand: {demand_scale}")    
            rank = i * len(demand_scales_evaluated_current_cycle) + j # This j goes from 0 to len(demand_scales_evaluated_current_cycle) - 1
            worker_config = {
                'n_iterations': n_iterations,
                'total_action_timesteps_per_episode': config['eval_n_timesteps'] // control_args['action_duration'], # Each time
                'worker_demand_scale': demand_scale,
                'policy_path': policy_path,
                'policy_args': ppo_args,
                'control_args': control_args,
                'worker_device': eval_worker_device,
            }
            p = mp.Process(
                target=parallel_eval_worker,
                args=(rank, worker_config, queue))
            
            p.start()
            processes.append(p)
            active_workers.append(rank)

        all_results = {}
        while active_workers:
            rank, result = queue.get()#timeout=60) # Result is obtained after all iterations are complete
            print(f"Result from worker {rank}: {result}")
            all_results[rank] = result
            active_workers.remove(rank)

        for p in processes:
            p.join()

        print(f"All results: {all_results}")    
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        result_json_path = os.path.join(config['save_dir'], f'eval_results_{current_time}.json')
        with open(result_json_path, 'w') as f:
            json.dump(all_results, f, indent=4)

        plot_consolidated_results(result_json_path)

def main(config):
    """
    Cannot create a bunch of connections in main and then pass them around. 
    Because each new worker needs a separate pedestrian and vehicle trips file.
    """
    # Set the start method for multiprocessing. It does not create a process itself but sets the method for creating a process.
    # Spawn means create a new process. There is a fork method as well which will create a copy of the current process.
    mp.set_start_method('spawn') 
    if config['evaluate']: 
        eval(config)

    elif config['sweep']:
        tuner = HyperParameterTuner(config, train)
        tuner.start()
    else:
        train(config)

if __name__ == "__main__":
    config = get_config()
    main(config)