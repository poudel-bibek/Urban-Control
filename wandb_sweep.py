import wandb
import torch.nn as nn
class HyperParameterTuner: 
    def __init__(self, config, train_function):
        self.config = config
        self.project = "ppo-urban-control"
        self.train_function = train_function
        
    def start(self, ):
        sweep_id = self.create_sweep_config()
        wandb.agent(sweep_id, function= self.hyperparameter_tune_main, count= self.config["total_sweep_trials"]) 

    def hyperparameter_tune_main(self):
        try:
            wandb.init(project=self.project, settings=wandb.Settings(disable_git=True))
            config = wandb.config
            self.train_function(self.config, is_sweep=True, sweep_config=config)
        finally:
            wandb.finish() 

    def create_sweep_config(self, method="bayes"): # options: random, grid, bayes
        """
        If using random, max and min values are required.
        We do not want to get weird weights such as 0.192 for various params. Hence not using random search.
        However, if using grid search requires all parameters to be categorical, constant, int_uniform

        On using bayes method for hyperparameter tuning:
            - Works well for small number of continuous parameters. Scales poorly.

        # What to maximize?
        Keep in mind: 
            1. Every iteration, the policy gets updated. 
            2. Each episode runs in a parallel worker with a randomly sampled scaling factor (ped/ veh demands).
            3. An episode might not be over yet the policy might be updated. This is how PPO works.
        Best Choice: avg_reward i.e., Average reward per process in this iteration.
            1. Robustness: avg_reward considers the performance across all processes in an iteration, each with potentially different demand scaling factors. 
            2. Consistency: By averaging rewards across processes, we reduce the impact of potential overfitting to a specific demand scaling factor.
        """
        if method == "random":
            sweep_config = {
            'method': 'random', 
            'metric': {
                'name': 'avg_reward',
                'goal': 'maximize'
                },
            'parameters': {
                'gae_lambda': {'values': [0.9, 0.95, 0.99]},
                'update_freq': {'values': [64, 128, 256, 512]},
                'lr': {'values': [0.05, 0.01, 0.002, 0.001, 0.0005] }, # starting value of lr
                'gamma': {'values': [0.90, 0.95, 0.98, 0.999]},
                'K_epochs': {'values': [2, 4, 8] },
                'eps_clip': {'values': [0.1, 0.2, 0.3]},
                'ent_coef': {'values': [0.0001, 0.001, 0.005, 0.01]},
                'vf_coef': {'values': [0.25, 0.5, 0.75, 1.0]},
                'batch_size': {'values': [15, 32, 64, 128]},
                # policy specific
                'model_type': {'values': ['cnn', 'mlp']},
                'size': {'values': ['small', 'medium']},
                'kernel_size': {'values': [3, 5]},
                'dropout_rate': {'values': [0.1, 0.2, 0.3]},
                # Reward related lambda
                'l1': {'values': [-0.20, -0.33, -0.5]}, # intersection vehicle 
                'l2': {'values': [-0.20, -0.33, -0.5]}, # intersection pedestrian 
                'l3': {'values': [-0.20, -0.33, -0.5]}, # midblock vehicle 
                'l4': {'values': [-0.20, -0.33, -0.5]}, # midblock pedestrian
                'l5': {'values': [-0.10, -0.20, -0.33, -0.5]}, # switch penalty 
                }
            }
        
        elif method=="bayes":
            sweep_config = {
            'method': 'bayes', 
            'metric': {
                'name': 'avg_eval',
                'goal': 'minimize'
                },
            'parameters': {
                'lr': {
                    'values': [1e-4]
                },
                'gae_lambda': {
                    'values': [0.95]
                },
                'update_freq': {
                    'values': [1024]
                },
                'gamma': {
                    'values': [0.99]
                },
                'K_epochs': {
                    'values': [4]
                },
                'eps_clip': {
                    'values': [0.2]
                },
                'ent_coef': {
                    'values': [0.01]
                },
                'vf_coef': {
                    'values': [0.5]
                },
                'vf_clip_param': {
                    'values': [0.5]
                },
                'batch_size': {
                    'values': [64]
                },
                # policy:
                'model_type': {
                    'values': ['mlp']
                },
                'size': {
                    'values': ['medium']
                },
                'activation': {
                    'values': ["tanh"]
                },
                # 'kernel_size': { # ignored if model_type is mlp
                #     'values': [3, 5, 7]
                # },
                # Reward-related lambdas: continuous range 
                # 'l1': {
                #     'min': -1.0,
                #     'max': -0.1,
                #     'distribution': 'uniform'
                # },
                # 'l2': {
                #     'min': -1.0,
                #     'max': -0.1,
                #     'distribution': 'uniform'
                # },
                # 'l3': {
                #     'min': -1.0,
                #     'max': -0.1,
                #     'distribution': 'uniform'
                # },
                # 'l4': {
                #     'min': -1.0,
                #     'max': -0.1,
                #     'distribution': 'uniform'
                # },
                # 'l5': {
                #     'min': -1.0,
                #     'max': -0.1,
                #     'distribution': 'uniform'
                # }
                }
            }  
        sweep_id = wandb.sweep(sweep_config, entity="fluidic-city", project=self.project)
        return sweep_id
