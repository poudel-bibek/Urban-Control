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
                    'values': [0.95, 0.99]
                },
                'update_freq': {
                    'values': [512, 1024]
                },
                'gamma': {
                    'values': [0.98, 0.99]
                },
                'K_epochs': {
                    'values': [4, 8]
                },
                'eps_clip': {
                    'values': [0.2, 0.25]
                },
                'ent_coef': {
                    'values': [0.01, 0.02]
                },
                'vf_coef': {
                    'values': [0.5, 0.75]
                },
                'vf_clip_param': {
                    'values': [0.5, 0.75]
                },
                'batch_size': {
                    'values': [64, 128]
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
                }
            }  
        sweep_id = wandb.sweep(sweep_config, entity="fluidic-city", project=self.project)
        return sweep_id
