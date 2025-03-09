def get_config():
    """
    Set config here.
    """
    config = {
        # Simulation
        "sweep": False,  # Use wandb sweeps for hyperparameter tuning
        "gui": True,  # Use SUMO GUI (default: False)
        "evaluate": False,  
        
        "step_length": 1.0,  # Real-world time in seconds per simulation timestep (default: 1.0). 
        "action_duration": 10,  # Number of simulation timesteps for each action (default: 10)
        "warmup_steps": [100, 250],  # Number of steps to run before collecting data
        "auto_start": True,  # Automatically start the simulation
        "vehicle_input_trips": "./simulation/original_vehtrips.xml",  # Original Input trips file
        "vehicle_output_trips": "./simulation/scaled_trips/scaled_vehtrips.xml",  # Output trips file
        "pedestrian_input_trips": "./simulation/original_pedtrips.xml",  # Original Input pedestrian trips file
        "pedestrian_output_trips": "./simulation/scaled_trips/scaled_pedtrips.xml",  # Output pedestrian trips file
        "original_net_file": "./simulation/Craver_traffic_lights_wide.net.xml",  # Original net file

        # Demand scaling
        "manual_demand_veh": None,  # Manually scale vehicle demand before starting the simulation (veh/hr)
        "manual_demand_ped": None,  # Manually scale pedestrian demand before starting the simulation (ped/hr)
        "demand_scale_min": 1.0,  # Minimum demand scaling factor for automatic scaling
        "demand_scale_max": 2.25,  # Maximum demand scaling factor for automatic scaling

        # PPO (general params)
        "seed": None,  # Random seed (default: None)
        "gpu": True,  # Use GPU if available (default: use CPU)
        "total_timesteps": 8000000,  # Total number of timesteps the simulation will run
        "max_timesteps": 600,  # Maximum number of steps in one episode (make this multiple of 16*10)
        "total_sweep_trials": 128,  # Total number of trials for the wandb sweep
        "memory_transfer_freq": 16,  # Frequency of memory transfer from worker to main process 
        "per_timestep_state_dim": 96,  # Number of features per timestep
        "model_type": "mlp",  # Model type: 'cnn' or 'mlp'

        # PPO
        "anneal_lr": True,  # Anneal learning rate
        "gae_lambda": 0.95,  # GAE lambda
        "max_grad_norm": 0.75,  # Maximum gradient norm for gradient clipping
        "vf_clip_param": 0.5,  # Value function clipping parameter
        "update_freq": 1024,  # Number of action timesteps between each policy update
        "lr": 1e-4,  # Learning rate
        "gamma": 0.99,  # Discount factor
        "K_epochs": 4,  # Number of epochs to update policy
        "eps_clip": 0.2,  # Clip parameter for PPO
        "save_freq": 5,  # Save model after every n updates (0 to disable). Also decided how often to evaluate
        "ent_coef": 0.01,  # Entropy coefficient
        "vf_coef": 0.5,  # Value function coefficient
        "batch_size": 64,  # Batch size
        "num_processes": 8,  # Number of parallel processes to use (agent has multiple workers)
        "kernel_size": 3,  # Kernel size for CNN
        "model_size": "medium",  # Model size for CNN: 'small' or 'medium'
        "dropout_rate": 0.25,  # Dropout rate for CNN
        "action_dim": 7 + 4,  # 7 + 4 for simple action. Number of action logits (not the same as number of actions. think)
        "in_channels": 1, # in_channels for cnn
        "activation": "tanh",  # Policy activation function

        # Evaluation
        "eval_model_path": "./saved_models/best_eval_policy.pth",  # Path to the saved PPO model for evaluation
        "eval_save_dir": None,
        "eval_n_timesteps": 600,  # Number of timesteps to each episode. Warmup not counted.
        "eval_n_workers": 8,  # Parallelizes how many demands can be evaluated at the same time.
        "eval_worker_device": "gpu",  # Policy during eval can be run in GPU 
    }
    return config

def classify_and_return_args(train_config, device):
    """
    Classify config and return. 
    """

    control_args = {
        'vehicle_input_trips': train_config['vehicle_input_trips'],
        'vehicle_output_trips': train_config['vehicle_output_trips'],
        'pedestrian_input_trips': train_config['pedestrian_input_trips'],
        'pedestrian_output_trips': train_config['pedestrian_output_trips'],
        'manual_demand_veh': train_config['manual_demand_veh'],
        'manual_demand_ped': train_config['manual_demand_ped'],
        'step_length': train_config['step_length'],
        'action_duration': train_config['action_duration'],
        'warmup_steps': train_config['warmup_steps'],
        'per_timestep_state_dim': train_config['per_timestep_state_dim'], 
        'gui': train_config['gui'],
        'auto_start': train_config['auto_start'],
        'max_timesteps': train_config['max_timesteps'],
        'demand_scale_min': train_config['demand_scale_min'],
        'demand_scale_max': train_config['demand_scale_max'],
        'memory_transfer_freq': train_config['memory_transfer_freq'],
        'save_freq': train_config['save_freq'],
        'writer': None, # Need dummy values for dummy envs init.
        'save_dir': None,
        'total_action_timesteps_per_episode': None,
        'num_processes': train_config['num_processes'],
        'anneal_lr': train_config['anneal_lr'],
        'update_freq': train_config['update_freq'],
        'model_type': train_config['model_type'],
    }

    model_kwargs = { # This is not to be returned on its own
        'action_duration': train_config['action_duration'],
        'model_size': train_config['model_size'],
        'kernel_size': train_config['kernel_size'],
        'dropout_rate': train_config['dropout_rate'],
        'per_timestep_state_dim': train_config['per_timestep_state_dim'],
        'activation': train_config['activation'],
    }

    ppo_args = {
        'model_dim': train_config['in_channels'], 
        'action_dim': train_config['action_dim'],
        'device': device,
        'lr': train_config['lr'],
        'gamma': train_config['gamma'],
        'K_epochs': train_config['K_epochs'],
        'eps_clip': train_config['eps_clip'],
        'ent_coef': train_config['ent_coef'],
        'vf_coef': train_config['vf_coef'],
        'batch_size': train_config['batch_size'],
        'gae_lambda': train_config['gae_lambda'],
        'max_grad_norm': train_config['max_grad_norm'],
        'vf_clip_param': train_config['vf_clip_param'],
        'model_type': train_config['model_type'],
        'model_kwargs': model_kwargs
    }
    
    if train_config['evaluate']:
        # during evaluation
        eval_n_iterations = 2
        in_range_demand_scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25] 
        out_of_range_demand_scales = [0.5, 0.75, 2.5, 2.75]
    else: 
        # during training
        eval_n_iterations = 10
        in_range_demand_scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25] # The demand scales that are used for training.
        out_of_range_demand_scales = [] # The demand scales that are used ONLY for evaluation.
    
    eval_args = {
        'state_dim': None,
        'eval_model_path': train_config['eval_model_path'],
        'eval_save_dir': train_config['eval_save_dir'],
        'eval_n_timesteps': train_config['eval_n_timesteps'],
        'eval_n_workers': train_config['eval_n_workers'],
        'eval_worker_device': train_config['eval_worker_device'],
        'eval_n_iterations': eval_n_iterations,
        'in_range_demand_scales': in_range_demand_scales,
        'out_of_range_demand_scales': out_of_range_demand_scales,
    }

    return control_args, ppo_args, eval_args