def get_config():
    """
    Set config here.
    """
    config = {
        # Simulation
        "sweep": False,  # Use wandb sweeps for hyperparameter tuning
        "gui": True,  # Use SUMO GUI (default: False)
        "step_length": 1.0,  # Simulation step length (default: 1.0). Since we have pedestrians, who walk slow. A value too small is not required.
        "action_duration": 10,  # Duration of each action (default: 10.0)
        "auto_start": True,  # Automatically start the simulation
        "vehicle_input_trips": "./SUMO_files/original_vehtrips.xml",  # Original Input trips file
        "vehicle_output_trips": "./SUMO_files/scaled_trips/scaled_vehtrips.xml",  # Output trips file
        "pedestrian_input_trips": "./SUMO_files/original_pedtrips.xml",  # Original Input pedestrian trips file
        "pedestrian_output_trips": "./SUMO_files/scaled_trips/scaled_pedtrips.xml",  # Output pedestrian trips file
        "original_net_file": "./SUMO_files/Craver_traffic_lights.net.xml",  # Original net file

        # Demand scaling
        "manual_demand_veh": None,  # Manually scale vehicle demand before starting the simulation (veh/hr)
        "manual_demand_ped": None,  # Manually scale pedestrian demand before starting the simulation (ped/hr)
        "demand_scale_min": 0.5,  # Minimum demand scaling factor for automatic scaling
        "demand_scale_max": 4.0,  # Maximum demand scaling factor for automatic scaling

        # PPO (general params)
        "seed": None,  # Random seed (default: None)
        "gpu": True,  # Use GPU if available (default: use CPU)
        "total_timesteps": 1500000,  # Total number of timesteps the simulation will run
        "max_timesteps": 250,  # Maximum number of steps in one episode 
        "total_sweep_trials": 128,  # Total number of trials for the wandb sweep
        "memory_transfer_freq": 16,  # Frequency of memory transfer from worker to main process 

        # PPO
        "anneal_lr": True,  # Anneal learning rate
        "gae_lambda": 0.95,  # GAE lambda
        "update_freq": 128,  # Number of action timesteps between each policy update
        "lr": 0.002,  # Learning rate
        "gamma": 0.99,  # Discount factor
        "K_epochs": 4,  # Number of epochs to update policy
        "eps_clip": 0.2,  # Clip parameter for PPO
        "save_freq": 2,  # Save model after every n updates (0 to disable), for both design and control agents
        "ent_coef": 0.01,  # Entropy coefficient
        "vf_coef": 0.5,  # Value function coefficient
        "batch_size": 32,  # Batch size
        "num_processes": 6,  # Number of parallel processes to use (agent has multiple workers)
        "kernel_size": 3,  # Kernel size for CNN
        "model_size": "medium",  # Model size for CNN: 'small' or 'medium'
        "dropout_rate": 0.2,  # Dropout rate for CNN
        "action_dim": 6,  # Number of action logits (not the same as number of actions. think)
        "in_channels": 1, # in_channels for cnn

        # Evaluation
        "evaluate": None,  # Evaluation mode: 'tl' (traffic light), 'ppo', or None
        "model_path": None,  # Path to the saved PPO model for evaluation
    }

    return config

def classify_and_return_args(train_config, worker_device):
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
        'gui': train_config['gui'],
        'auto_start': train_config['auto_start'],
        'max_timesteps': train_config['max_timesteps'],
        'demand_scale_min': train_config['demand_scale_min'],
        'demand_scale_max': train_config['demand_scale_max'],
        'memory_transfer_freq': train_config['memory_transfer_freq'],
        'save_freq': train_config['save_freq'],
        'writer': None, # Need dummy values for dummy envs init.
        'save_dir': None,
        'global_seed': None,
        'total_action_timesteps_per_episode': None,
        'num_processes': train_config['num_processes'],
        'anneal_lr': train_config['anneal_lr'],
        'update_freq': train_config['update_freq'],
    }

    model_kwargs = { # This is not to be returned on its own
        'action_duration': train_config['action_duration'],
        'model_size': train_config['model_size'],
        'kernel_size': train_config['kernel_size'],
        'dropout_rate': train_config['dropout_rate'],
        'per_timestep_state_dim': 40, # Circular dependency, hardcoded here
    }

    ppo_args = {
        'model_dim': train_config['in_channels'], 
        'action_dim': train_config['action_dim'],
        'device': worker_device,
        'lr': train_config['lr'],
        'gamma': train_config['gamma'],
        'K_epochs': train_config['K_epochs'],
        'eps_clip': train_config['eps_clip'],
        'ent_coef': train_config['ent_coef'],
        'vf_coef': train_config['vf_coef'],
        'batch_size': train_config['batch_size'],
        'gae_lambda': train_config['gae_lambda'],
        'model_kwargs': model_kwargs
    }

    return control_args, ppo_args