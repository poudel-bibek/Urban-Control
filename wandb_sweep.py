import os
import wandb
class HyperParameterTuner: 
    def __init__(self, args):
        self.args = args
        self.project = "ppo-urban-control"
        wandb.login(key="75fd76d4675a1868cd6ae899d1fbfb2eadaea843")

    def start(self):
        sweep_id = self.create_sweep_config()
        wandb.agent(sweep_id, function= self.hyperparameter_tune_main, count= self.args.total_sweep_trials) 

    def hyperparameter_tune_main(self):
        wandb.init(project=self.project) 
        config = wandb.config
        from main import train # Importing here to avoid circular import error.
        train(self.args, is_sweep=True, config=config)

    def create_sweep_config(self):
        """
        If using random, max and min values are required.
        We do not want to get weird weights such as 0.192 for various params. Hence not using random search.
        However, if using grid search requires all parameters to be categorical, constant, int_uniform

        # What to maximize?
        Keep in mind: 
            1. Every iteration, the policy gets updated. 
            2. Each episode runs in a parallel worker with a randomly sampled scaling factor (ped/ veh demands).
            3. An episode might not be over yet the policy might be updated. This is how PPO works.
        Best Choice: avg_reward i.e., Average reward per process in this iteration.
            1. Robustness: avg_reward considers the performance across all processes in an iteration, each with potentially different demand scaling factors. 
            2. Consistency: By averaging rewards across processes, we reduce the impact of potential overfitting to a specific demand scaling factor.
        """

        sweep_config = {
        'method': 'random', # options: random, grid, bayes
        'metric': {
            'name': 'avg_reward',
            'goal': 'maximize'
            },
        'parameters': {
            'gae_lambda': {'values': [0.9, 0.95, 0.99]},
            'update_freq': {'values': [64, 128, 256, 512]},
            'lr': {'values': [0.05, 0.01, 0.002, 0.001, 0.0005] }, # starting value of lr
            'gamma': {'values': [0.90, 0.95, 0.98, 0.999]},
            'K_epochs': {'values': [2, 4, 8, 16] },
            'eps_clip': {'values': [0.1, 0.2, 0.3]},
            'ent_coef': {'values': [0.0001, 0.001, 0.005, 0.01]},
            'vf_coef': {'values': [0.25, 0.5, 0.75, 1.0]},
            'batch_size': {'values': [15, 32, 64, 128]},
            # CNN policy specific
            'size': {'values': ['small', 'medium']},
            'kernel_size': {'values': [3, 5]},
            'dropout_rate': {'values': [0.1, 0.2, 0.3]},
            # Reward related lambda
            'l1': {'values': [-0.33, -0.5, -1]},
            'l2': {'values': [-0.33, -0.5, -1]},
            'l3': {'values': [-0.33, -0.5, -1]},
            'l4': {'values': [-0.33, -0.5, -1]},
            'l5': {'values': [-0.33, -0.5, -1]},
            }
        }
        sweep_id = wandb.sweep(sweep_config, entity="matmul", project=self.project)
        return sweep_id
