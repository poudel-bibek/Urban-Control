# Co-designing the built environment and traffic controller using reinforcement learning

## Abstract
This project explores the co-design of urban environments and traffic control systems using reinforcement learning techniques, specifically the Proximal Policy Optimization (PPO) algorithm. We utilize SUMO (Simulation of Urban MObility) for traffic simulation and implement our learning algorithms using PyTorch.

## Requirements
- SUMO version 1.20.0
- Python version 3.12.0
- PyTorch
- TensorBoard (optional, for logging)
- Weights & Biases (optional, for experiment tracking and hyperparameter tuning)

## Setup
1. Clone the repository
2. Install required packages:
   ```bash
   pip install torch traci numpy tensorboard wandb
   ```
3. Ensure SUMO is installed and properly configured in your system PATH

## Project Structure
- `SUMO_files/`: Contains SUMO configuration and trip files
- `models.py`: Neural network architecture definitions
- `ppo_alg.py`: PPO algorithm implementation
- `ppo_run.py`: Main script for running experiments
- `sim_run.py`: SUMO environment wrapper
- `wandb_sweep.py`: Hyperparameter tuning setup

## Key Parameters
- `--step_length`: Simulation step length (default: 1.0)
- `--action_duration`: Duration of each action (default: 10.0)
- `--total_timesteps`: Total training timesteps (default: 100000)
- `--max_timesteps`: Maximum steps per episode (default: 1000)
- `--num_processes`: Number of parallel processes (default: 8)

For a complete list of parameters, refer to the argument parser in `ppo_run.py`.

## Running Experiments

### Single Run
To start training with default settings:


This command may be different: 
```bash
command = f"netconvert --sumo-net-file {sumo_net_file} --plain-output-prefix {output_dir}/base_xml --plain-output.lanes true"
```