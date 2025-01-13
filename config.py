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
        "component_dir": "./SUMO_files/component_SUMO_files",
        "network_dir": "./SUMO_files/network_iterations",

        # Demand scaling
        "manual_demand_veh": None,  # Manually scale vehicle demand before starting the simulation (veh/hr)
        "manual_demand_ped": None,  # Manually scale pedestrian demand before starting the simulation (ped/hr)
        "demand_scale_min": 0.5,  # Minimum demand scaling factor for automatic scaling
        "demand_scale_max": 4.0,  # Maximum demand scaling factor for automatic scaling

        # PPO (general params)
        "seed": None,  # Random seed (default: None)
        "gpu": True,  # Use GPU if available (default: use CPU)
        "total_timesteps": 1500000,  # Total number of timesteps the simulation will run
        "max_timesteps": 1500,  # Maximum number of steps in one episode (for the lower level agent)
        "total_sweep_trials": 128,  # Total number of trials for the wandb sweep
        "memory_transfer_freq": 16,  # Frequency of memory transfer from worker to main process (Only applicable for lower level agent)

        # PPO (higher level agent)
        "higher_anneal_lr": True,  # Anneal learning rate
        "higher_lr": 0.001,  # Learning rate for higher-level agent
        "higher_gamma": 0.99,  # Discount factor for higher-level agent
        "higher_K_epochs": 4,  # Number of epochs to update policy for higher-level agent
        "higher_eps_clip": 0.2,  # Clip parameter for PPO for higher-level agent
        "higher_ent_coef": 0.01,  # Entropy coefficient for higher-level agent
        "higher_vf_coef": 0.5,  # Value function coefficient for higher-level agent
        "higher_batch_size": 32,  # Batch size for higher-level agent
        "higher_gae_lambda": 0.95,  # GAE lambda for higher-level agent
        "higher_hidden_channels": 64,  # Number of hidden channels in GATv2 layers
        "higher_out_channels": 32,  # Number of output channels in GATv2 layers
        "higher_initial_heads": 8,  # Number of attention heads in first GATv2 layer
        "higher_second_heads": 1,  # Number of attention heads in second GATv2 layer
        "higher_action_hidden_channels": 32,  # Number of hidden channels in action layers
        "higher_update_freq": 2,  # Number of action timesteps between each policy update
        "higher_gmm_hidden_dim": 64,  # Hidden dimension for GMM layers
        "higher_num_mixtures": 3,  # Number of mixtures in GMM
        "higher_edge_dim": 2,  # Number of features per edge (location, width)
        "higher_in_channels": 2,  # Number of input features per node (x and y coordinates)
        

        # Higher-level agent specific arguments
        "max_proposals": 10,  # Maximum number of crosswalk proposals
        "min_thickness": 0.1,  # Minimum thickness of crosswalks
        "max_thickness": 10.0,  # Maximum thickness of crosswalks
        "min_coordinate": 0.0,  # Minimum coordinate for crosswalk placement
        "max_coordinate": 1.0,  # Maximum coordinate for crosswalk placement
        "save_graph_images": True, # Save graph image every iteration.
        "save_gmm_plots": True, # Save GMM visualization every iteration.

        # PPO (lower level agent)
        "lower_anneal_lr": True,  # Anneal learning rate
        "lower_gae_lambda": 0.95,  # GAE lambda
        "lower_update_freq": 128,  # Number of action timesteps between each policy update
        "lower_lr": 0.002,  # Learning rate
        "lower_gamma": 0.99,  # Discount factor
        "lower_K_epochs": 4,  # Number of epochs to update policy
        "lower_eps_clip": 0.2,  # Clip parameter for PPO
        "save_freq": 2,  # Save model after every n updates (0 to disable), for both design and control agents
        "lower_ent_coef": 0.01,  # Entropy coefficient
        "lower_vf_coef": 0.5,  # Value function coefficient
        "lower_batch_size": 32,  # Batch size
        "lower_num_processes": 1,  # Number of parallel processes to use (Lower level agent has multiple workers)
        "lower_kernel_size": 3,  # Kernel size for CNN
        "lower_model_size": "medium",  # Model size for CNN: 'small' or 'medium'
        "lower_dropout_rate": 0.2,  # Dropout rate for CNN
        "lower_action_dim": 6,  # Number of action logits (not the same as number of actions. think)
        "lower_in_channels": 1, # in_channels for cnn

        # Evaluation
        "evaluate": None,  # Evaluation mode: 'tl' (traffic light), 'ppo', or None
        "model_path": None,  # Path to the saved PPO model for evaluation
    }

    return config

def classify_and_return_args(train_config, worker_device):
    """
    Classify config and return. 
    """

    design_args = {
        'save_graph_images': train_config['save_graph_images'],
        'save_gmm_plots': train_config['save_gmm_plots'],
        'max_proposals': train_config['max_proposals'],
        'min_thickness': train_config['min_thickness'],
        'max_thickness': train_config['max_thickness'],
        'min_coordinate': train_config['min_coordinate'],
        'max_coordinate': train_config['max_coordinate'],
        'save_freq': train_config['save_freq'],
        'original_net_file': train_config['original_net_file'],
        'component_dir': train_config['component_dir'],
        'network_dir': train_config['network_dir'],
    }

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
        'lower_num_processes': train_config['lower_num_processes'],
        'lower_anneal_lr': train_config['lower_anneal_lr'],
        'lower_update_freq': train_config['lower_update_freq'],
    }

    higher_model_kwargs = {
        'hidden_channels': train_config['higher_hidden_channels'],
        'out_channels': train_config['higher_out_channels'],
        'initial_heads': train_config['higher_initial_heads'],
        'second_heads': train_config['higher_second_heads'],
        'edge_dim': train_config['higher_edge_dim'],
        'action_hidden_channels': train_config['higher_action_hidden_channels'],
        'gmm_hidden_dim': train_config['higher_gmm_hidden_dim'],
        'num_mixtures': train_config['higher_num_mixtures'],
    }

    lower_model_kwargs = {
        'action_duration': train_config['action_duration'],
        'model_size': train_config['lower_model_size'],
        'kernel_size': train_config['lower_kernel_size'],
        'dropout_rate': train_config['lower_dropout_rate'],
        'per_timestep_state_dim': 40, # Circular dependency, hardcoded here
    }

    higher_ppo_args = {
        'model_dim': train_config['higher_in_channels'],
        'action_dim': train_config['max_proposals'],  # Action dimension
        'device': worker_device,
        'lr': train_config['higher_lr'],
        'gamma': train_config['higher_gamma'],
        'K_epochs': train_config['higher_K_epochs'],
        'eps_clip': train_config['higher_eps_clip'],
        'ent_coef': train_config['higher_ent_coef'],
        'vf_coef': train_config['higher_vf_coef'],
        'batch_size': train_config['higher_batch_size'],
        'gae_lambda': train_config['higher_gae_lambda'],
        'agent_type': "higher",
        'model_kwargs': higher_model_kwargs
    }

    lower_ppo_args = {
        'model_dim': train_config['lower_in_channels'], 
        'action_dim': train_config['lower_action_dim'],
        'device': worker_device,
        'lr': train_config['lower_lr'],
        'gamma': train_config['lower_gamma'],
        'K_epochs': train_config['lower_K_epochs'],
        'eps_clip': train_config['lower_eps_clip'],
        'ent_coef': train_config['lower_ent_coef'],
        'vf_coef': train_config['lower_vf_coef'],
        'batch_size': train_config['lower_batch_size'],
        'gae_lambda': train_config['lower_gae_lambda'],
        'agent_type': "lower",
        'model_kwargs': lower_model_kwargs
    }

    return design_args, control_args, lower_ppo_args, higher_ppo_args