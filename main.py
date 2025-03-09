import os
import json
import time
import wandb
import torch
import random
import numpy as np
from datetime import datetime
from ppo.ppo import PPO, Memory, WelfordNormalizer
from config import get_config, classify_and_return_args
import torch.multiprocessing as mp

from sweep import HyperParameterTuner
from torch.utils.tensorboard import SummaryWriter

from utils import *
from simulation.env import ControlEnv

def parallel_train_worker(rank, shared_policy_old, control_args, train_queue, worker_seed, shared_state_normalizer, shared_reward_normalizer, worker_device):
    """
    At every iteration, a number of workers will each parallelly carry out one episode in control environment.
    - Worker environment runs in CPU (SUMO runs in CPU).
    - Worker policy inference runs in GPU.
    - memory_queue is used to store the memory of each worker and send it back to the main process.
    - A shared policy_old (dict copy passed here) is used for importance sampling.
    - 1 memory transfer happens every memory_transfer_freq * action_duration sim steps.
    """

    # Set seed for this worker
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
            state = shared_state_normalizer.normalize(state)
            state = state.to(worker_device)
            action, logprob = shared_policy_old.act(state) # sim runs in CPU, state will initially always be in CPU.
            value = shared_policy_old.critic(state.unsqueeze(0)) # add a batch dimension

            state = state.detach().cpu().numpy() # 2D
            action = action.detach().cpu().numpy() # 1D
            value = value.item() # Scalar
            logprob = logprob.item() # Scalar
            # print(f"State: {state}")

        # Perform action
        # These reward and next_state are for the action_duration timesteps.
        next_state, reward, done, truncated, _ = worker_env.train_step(action) # need the returned state to be 2D
        reward = shared_reward_normalizer.normalize(reward).item()
        ep_reward += reward

        # Store data in memory
        local_memory.append(state, action, value, logprob, reward, done) 
        steps_since_update += 1

        if steps_since_update >= memory_transfer_freq or done or truncated:
            # Put local memory in the queue for the main process to collect
            train_queue.put((rank, local_memory))
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
    train_queue.put((rank, None))  # Signal that this worker is done

def save_config(config, save_path):
    """
    Save hyperparameters to json.
    """
    config_to_save = {
        "hyperparameters": config,
    }
    with open(save_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)

def save_policy(policy, normalizer, save_path):  
    """
    Save policy state dict and welford normalizer stats.
    """
    torch.save({  
        'policy_state_dict': policy.state_dict(),  
        'state_normalizer_mean': normalizer.mean.numpy(),  
        'state_normalizer_M2': normalizer.M2.numpy(),  
        'state_normalizer_count': normalizer.count.value  
    }, save_path)

def load_policy(policy, normalizer, load_path):
    """
    Load policy state dict and welford normalizer stats.
    """
    checkpoint = torch.load(load_path)
    # In place operations
    policy.load_state_dict(checkpoint['policy_state_dict'])
    normalizer.manual_load(
        mean=torch.from_numpy(checkpoint['state_normalizer_mean']),  
        M2=torch.from_numpy(checkpoint['state_normalizer_M2']),  
        count=checkpoint['state_normalizer_count']
    )

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

    device = torch.device("cuda") if train_config['gpu'] and torch.cuda.is_available() else torch.device("cpu") 
    print(f"Using device: {device}")

    # Set and save hyperparameters 
    if is_sweep:
        for key, value in sweep_config.items():
            train_config[key] = value

    os.makedirs('runs', exist_ok=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(log_dir, f'config_{current_time}.json')
    save_config(train_config, config_path)
    print(f"Configuration saved to {config_path}")

    control_args, ppo_args, eval_args = classify_and_return_args(train_config, device)

    # Print stats from dummy environment
    dummy_env = ControlEnv(control_args, worker_id=None)
    print(f"\nEnvironment for control agent:")
    print(f"\tDefined observation space: {dummy_env.observation_space}")
    print(f"\tObservation space shape: {dummy_env.observation_space.shape}")
    print(f"\tDefined action space: {dummy_env.action_space}")
    print(f"\tOptions per action dimension: {dummy_env.action_space.nvec}")

    # use the dummy env shapes to init normalizers
    obs_shape = dummy_env.observation_space.shape
    eval_args['state_dim'] = obs_shape
    shared_state_normalizer = WelfordNormalizer(obs_shape)
    shared_reward_normalizer = WelfordNormalizer(1)
    dummy_env.close()

    # Initialize control agent
    print(f"\nControl agent: \n\tState dimension: {dummy_env.observation_space.shape}, Action dimension: {train_config['action_dim']}")
    control_ppo = PPO(**ppo_args)

    # Model saving and tensorboard 
    writer = SummaryWriter(log_dir=log_dir)
    save_dir = os.path.join('saved_models', current_time)
    os.makedirs(save_dir, exist_ok=True)
    eval_args['eval_save_dir'] = os.path.join('results', f'train_{current_time}')
    os.makedirs(eval_args['eval_save_dir'], exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    control_args.update({
        'writer': writer,
        'save_dir': save_dir,
        'total_action_timesteps_per_episode': train_config['max_timesteps'] // train_config['action_duration']
    })
    # worker related args
    control_args_worker = {k: v for k, v in control_args.items() if k != 'writer'} # bug fix. writer is unpicklable
    
    # Instead of using total_episodes, we will use total_iterations. 
    # Every iteration, num_process control agents interact with the environment for total_action_timesteps_per_episode steps (which further internally contains action_duration steps)
    total_iterations = train_config['total_timesteps'] // (train_config['max_timesteps'] * train_config['num_processes'])
    total_updates = train_config['total_timesteps'] // train_config['update_freq']
    control_ppo.total_iterations = total_iterations
    
    global_step = 0
    update_count = 0
    action_timesteps = 0
    best_reward = float('-inf') 
    best_loss = float('inf')
    best_eval = float('inf')
    avg_eval = 200.0 # arbitrary large number
    eval_veh_avg_wait = 200.0 
    eval_ped_avg_wait = 200.0    

    # Every iteration, save all the sampled actions to a json file (by appending to the file).
    # A newer policy does importance sampling only every iteration. 
    actions_file_path = os.path.join(save_dir, f'sampled_actions.json')
    open(actions_file_path, 'w').close()
    sampled_actions = []

    all_memories = Memory()
    for iteration in range(0, total_iterations): # Starting from 1 to prevent policy update in the very first iteration.
        print(f"\nStarting iteration: {iteration + 1}/{total_iterations} with {global_step} total steps so far\n")
        
        old_policy = control_ppo.policy_old.to(device)
        old_policy.share_memory() # Dont pickle separate policy_old for each worker. Despite this, the old policy is still stale.
        old_policy.eval() # So that dropout, batnorm, laternote etc. are not used during inference

        #print(f"Shared policy weights: {control_ppo.policy_old.state_dict()}")
        train_queue = mp.Queue()
        train_processes = []
        active_train_workers = []
        for rank in range(control_args['num_processes']):

            worker_seed = SEED + iteration * 1000 + rank
            p = mp.Process(
                target=parallel_train_worker,
                args=(
                    rank,
                    old_policy,
                    control_args_worker,
                    train_queue,
                    worker_seed,
                    shared_state_normalizer,
                    shared_reward_normalizer,
                    device)
                )
            p.start()
            train_processes.append(p)
            active_train_workers.append(rank)
        
        while active_train_workers:
            print(f"Active workers: {active_train_workers}")
            rank, memory = train_queue.get()

            if memory is None:
                print(f"Worker {rank} finished")
                active_train_workers.remove(rank)
            else:
                current_action_timesteps = len(memory.states)
                print(f"Memory from worker {rank} received. Memory size: {current_action_timesteps}")
                all_memories.actions.extend(torch.from_numpy(np.asarray(memory.actions)))
                all_memories.states.extend(torch.from_numpy(np.asarray(memory.states)))
                all_memories.values.extend(memory.values)
                all_memories.logprobs.extend(memory.logprobs)
                all_memories.rewards.extend(memory.rewards)
                all_memories.is_terminals.extend(memory.is_terminals)

                sampled_actions.append(memory.actions[0].tolist())
                action_timesteps += current_action_timesteps
                global_step += current_action_timesteps * train_config['action_duration'] 
                print(f"Action timesteps: {action_timesteps}, global step: {global_step}")
                del memory #https://pytorch.org/docs/stable/multiprocessing.html

                # Update PPO every n times (or close to n) action has been taken 
                if action_timesteps >= control_args['update_freq']:
                    print(f"Updating PPO with {len(all_memories.actions)} memories") 

                    update_count += 1
                    # Anneal after every update
                    if control_args['anneal_lr']:
                        current_lr = control_ppo.update_learning_rate(update_count, total_updates)

                    avg_reward = sum(all_memories.rewards) / len(all_memories.rewards)
                    print(f"\nAverage Reward (across all memories): {avg_reward}\n")
                    #print(f"\nAll memories rewards: {all_memories.rewards}")

                    loss = control_ppo.update(all_memories)

                    # Reset all memories
                    del all_memories
                    all_memories = Memory() 
                    action_timesteps = 0
                    print(f"Size of all memories after update: {len(all_memories.actions)}")

                    # Save both during sweep and non-sweep
                    # Save (and evaluate the latest policy) every save_freq updates
                    if update_count % control_args['save_freq'] == 0:
                        latest_policy_path = os.path.join(control_args['save_dir'], f'policy_at_step_{global_step}.pth')
                        save_policy(control_ppo.policy, shared_state_normalizer, latest_policy_path)
                    
                        print(f"Evaluating policy: {latest_policy_path} at step {global_step}")
                        eval_json = eval(control_args_worker, ppo_args, eval_args, policy_path=latest_policy_path, tl= False) # which policy to evaluate?
                        _, eval_veh_avg_wait, eval_ped_avg_wait, _, _ = get_averages(eval_json)
                        eval_veh_avg_wait = np.mean(eval_veh_avg_wait)
                        eval_ped_avg_wait = np.mean(eval_ped_avg_wait)
                        avg_eval = ((eval_veh_avg_wait + eval_ped_avg_wait) / 2)
                        print(f"Eval veh avg wait: {eval_veh_avg_wait}, eval ped avg wait: {eval_ped_avg_wait}, avg eval: {avg_eval}")

                    # Save best policies 
                    if avg_reward > best_reward:
                        save_policy(control_ppo.policy, shared_state_normalizer, os.path.join(control_args['save_dir'], 'best_reward_policy.pth'))
                        best_reward = avg_reward
                    if loss['total_loss'] < best_loss:
                        save_policy(control_ppo.policy, shared_state_normalizer, os.path.join(control_args['save_dir'], 'best_loss_policy.pth'))
                        best_loss = loss['total_loss']
                    if avg_eval < best_eval:
                        save_policy(control_ppo.policy, shared_state_normalizer, os.path.join(control_args['save_dir'], 'best_eval_policy.pth'))
                        best_eval = avg_eval

                    # logging
                    if is_sweep: # Wandb for hyperparameter tuning
                        wandb.log({ "iteration": iteration,
                                        "avg_reward": avg_reward, # Set as maximize in the sweep config
                                        "update_count": update_count,
                                        "policy_loss": loss['policy_loss'],
                                        "value_loss": loss['value_loss'], 
                                        "entropy_loss": loss['entropy_loss'],
                                        "total_loss": loss['total_loss'],
                                        "current_lr": current_lr if control_args['anneal_lr'] else ppo_args['lr'],
                                        "approx_kl": loss['approx_kl'],
                                        "eval_veh_avg_wait": eval_veh_avg_wait,
                                        "eval_ped_avg_wait": eval_ped_avg_wait,
                                        "avg_eval": avg_eval,
                                        "global_step": global_step })
                        
                    else: # Tensorboard for regular training
                        writer.add_scalar('Training/Average_Reward', avg_reward, global_step)
                        writer.add_scalar('Training/Total_Policy_Updates', update_count, global_step)
                        writer.add_scalar('Training/Policy_Loss', loss['policy_loss'], global_step)
                        writer.add_scalar('Training/Value_Loss', loss['value_loss'], global_step)
                        writer.add_scalar('Training/Entropy_Loss', loss['entropy_loss'], global_step)
                        writer.add_scalar('Training/Total_Loss', loss['total_loss'], global_step)
                        writer.add_scalar('Training/Current_LR', current_lr if control_args['anneal_lr'] else ppo_args['lr'], global_step)
                        writer.add_scalar('Training/Approx_KL', loss['approx_kl'], global_step)
                        writer.add_scalar('Evaluation/Veh_Avg_Wait', eval_veh_avg_wait, global_step)
                        writer.add_scalar('Evaluation/Ped_Avg_Wait', eval_ped_avg_wait, global_step)
                        writer.add_scalar('Evaluation/Avg_Eval', avg_eval, global_step)
                    print(f"\nLogged data at step {global_step}\n")

                    # At the end of update, save normalizer stats
                    state_normalizer_mean = shared_state_normalizer.mean.numpy()  
                    state_normalizer_M2 = shared_state_normalizer.M2.numpy()  
                    state_normalizer_count = shared_state_normalizer.count.value  

        # Clean up. The join() method ensures that the main program waits for all processes to complete before continuing.
        for p in train_processes:
            p.join() 
        print(f"All processes joined\n\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del train_queue

        # Save all the sampled actions to a json file
        with open(actions_file_path, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            data[iteration] = sampled_actions
            f.seek(0)
            #print(f"Sampled actions: {sampled_actions}")
            json.dump(data, f, indent=4)
            f.truncate()
            f.close()
        sampled_actions = []

    if not is_sweep:
        writer.close()

def parallel_eval_worker(rank, eval_worker_config, eval_queue, tl=False, unsignalized=False):
    """
    - For the same demand, each worker runs n_iterations number of episodes and measures performance metrics at each iteration.
    - Each episode runs on a different random seed.
    - Performance metrics: 
        - Average waiting time (Veh, Ped)
        - Average travel time (Veh, Ped)
    - Returns a dictionary with performance metrics in all iterations.
    - For PPO: 
        - Create a single shared policy, and share among workers.
    - For TL:
        - Just pass tl = True
        - If unsignalized, all midblock TLs have no lights (equivalent to having all phases green)
    """
    
    shared_policy = eval_worker_config['shared_policy']
    worker_demand_scale = eval_worker_config['worker_demand_scale']
    control_args = eval_worker_config['control_args']

    # We set the demand manually (so that automatic scaling does not happen)
    control_args['manual_demand_veh'] = worker_demand_scale
    control_args['manual_demand_ped'] = worker_demand_scale
    env = ControlEnv(control_args, worker_id=rank)
    worker_result = {}
    
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
        worker_device = eval_worker_config['worker_device']
        shared_eval_normalizer = eval_worker_config['shared_eval_normalizer']
        # Run the worker (reset includes warmup)
        state, _ = env.reset(tl = tl)
        veh_waiting_time_this_episode = 0
        ped_waiting_time_this_episode = 0
        veh_unique_ids_this_episode = 0
        ped_unique_ids_this_episode = 0

        with torch.no_grad():
            for _ in range(eval_worker_config['total_action_timesteps_per_episode']):
                state = torch.FloatTensor(state)
                state = shared_eval_normalizer.normalize(state)
                state = state.to(worker_device)

                action, _ = shared_policy.act(state)
                action = action.detach().cpu() # sim runs in CPU
                state, reward, done, truncated, _ = env.eval_step(action, tl, unsignalized=unsignalized)

                # During this step, get all vehicles and pedestrians
                veh_waiting_time_this_step = env.get_vehicle_waiting_time()
                ped_waiting_time_this_step = env.get_pedestrian_waiting_time()

                veh_waiting_time_this_episode += veh_waiting_time_this_step
                ped_waiting_time_this_episode += ped_waiting_time_this_step

                veh_unique_ids_this_episode, ped_unique_ids_this_episode = env.total_unique_ids()

        # gather performance metrics
        worker_result[i]['total_veh_waiting_time'] = veh_waiting_time_this_episode
        worker_result[i]['total_ped_waiting_time'] = ped_waiting_time_this_episode
        worker_result[i]['veh_avg_waiting_time'] = veh_waiting_time_this_episode / veh_unique_ids_this_episode
        worker_result[i]['ped_avg_waiting_time'] = ped_waiting_time_this_episode / ped_unique_ids_this_episode
        worker_result[i]['total_conflicts'] = env.total_conflicts
        worker_result[i]['total_switches'] = env.total_switches

    # After all iterations are complete. 
    env.close()
    time.sleep(10) # Essential
    del env
    eval_queue.put((worker_demand_scale, worker_result))

def eval(control_args, ppo_args, eval_args, policy_path=None, tl=False, unsignalized=False):
    """
    Works to evaluate a policy during training as well as stand-alone policy vs real-world TL (tl = True) evaluation.
    - Each demand is run on a different worker
    - Results saved as json dict. 
    """
    n_workers = eval_args['eval_n_workers']
    n_iterations = eval_args['eval_n_iterations']
    eval_device = torch.device("cuda") if eval_args['eval_worker_device']=='gpu' and torch.cuda.is_available() else torch.device("cpu")
    eval_demand_scales = eval_args['in_range_demand_scales'] + eval_args['out_of_range_demand_scales']
    all_results = {}

    eval_ppo = PPO(**ppo_args)
    shared_eval_normalizer = WelfordNormalizer(eval_args['state_dim'])
    shared_eval_normalizer.eval()
    if policy_path:
        load_policy(eval_ppo.policy, shared_eval_normalizer, policy_path)

    shared_policy = eval_ppo.policy.to(eval_device)
    shared_policy.share_memory()
    shared_policy.eval()
    
    # number of times the n_workers have to be repeated to cover all eval demands
    num_times_workers_recycle = len(eval_demand_scales) if len(eval_demand_scales) < n_workers else (len(eval_demand_scales) // n_workers) + 1
    for i in range(num_times_workers_recycle):
        start = n_workers * i   
        end = n_workers * (i + 1)
        demand_scales_evaluated_current_cycle = eval_demand_scales[start: end]

        eval_queue = mp.Queue()
        eval_processes = []  
        active_eval_workers = []
        demand_scale_to_rank = {}
        for rank, demand_scale in enumerate(demand_scales_evaluated_current_cycle): 
            demand_scale_to_rank[demand_scale] = rank
            print(f"For demand: {demand_scale}")    
            worker_config = {
                'n_iterations': n_iterations,
                'total_action_timesteps_per_episode': config['eval_n_timesteps'] // control_args['action_duration'], # Each time
                'worker_demand_scale': demand_scale,
                'shared_policy': shared_policy,
                'control_args': control_args,
                'worker_device': eval_device,
                'shared_eval_normalizer': shared_eval_normalizer
            }
            p = mp.Process(
                target=parallel_eval_worker,
                args=(rank, worker_config, eval_queue, tl, unsignalized))
            
            p.start()
            eval_processes.append(p)
            active_eval_workers.append(rank)

        while active_eval_workers:
            worker_demand_scale, result = eval_queue.get() #timeout=60) # Result is obtained after all iterations are complete
            print(f"Result from worker with demand scale: {worker_demand_scale}: {result}")
            all_results[worker_demand_scale] = result
            active_eval_workers.remove(demand_scale_to_rank[worker_demand_scale])

        for p in eval_processes:
            p.join()

    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    del eval_queue
    del shared_policy

    print(f"All results: {all_results}")    
    if tl and unsignalized:
        tl_state = "unsignalized"
    elif tl:
        tl_state = "tl"
    else:
        tl_state = "ppo"
    
    result_json_path = os.path.join(eval_args['eval_save_dir'], f'{policy_path.split("/")[-1].split(".")[0]}_{tl_state}.json') # f'eval_{policy_path.split("/")[2].split(".")[0]}_{tl_state}.json
    with open(result_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    f.close()
    return result_json_path
    
def main(config):
    """
    Cannot create a bunch of connections in main and then pass them around. 
    Because each new worker needs a separate pedestrian and vehicle trips file.
    """
    # Set the start method for multiprocessing. It does not create a process itself but sets the method for creating a process.
    # Spawn means create a new process. There is a fork method as well which will create a copy of the current process.
    mp.set_start_method('spawn') 
    if config['evaluate']: 
        device = torch.device("cuda") if config['eval_worker_device']=='gpu' and torch.cuda.is_available() else torch.device("cpu")
        control_args, ppo_args, eval_args = classify_and_return_args(config, device)
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        os.makedirs(f'./results', exist_ok=True)
        os.makedirs(f'./results/eval_{current_time}', exist_ok=True)
        eval_args['eval_save_dir'] = os.path.join('results', f'eval_{current_time}')

        dummy_env = ControlEnv(control_args, worker_id=None)
        eval_args['state_dim'] = dummy_env.observation_space.shape
        
        ppo_results_path = eval(control_args, ppo_args, eval_args, policy_path=config['eval_model_path'], tl= False)
        tl_results_path = eval(control_args, ppo_args, eval_args, policy_path=None, tl= True, unsignalized=False) 
        unsignalized_results_path = eval(control_args, ppo_args, eval_args, policy_path=None, tl= True, unsignalized=True)

        plot_main_results(unsignalized_results_path, 
                          tl_results_path,
                          ppo_results_path,
                          in_range_demand_scales = eval_args['in_range_demand_scales'])

    elif config['sweep']:
        tuner = HyperParameterTuner(config, train)
        tuner.start()
    else:
        train(config)

if __name__ == "__main__":
    config = get_config()
    main(config)