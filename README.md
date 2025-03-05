
  

## Joint Pedestrian and Vehicle Traffic Optimization in Urban Environments using Reinforcement Learning

<a  href='https://arxiv.org/'><img  src='https://img.shields.io/badge/arXiv--green'></a> <a  href='https://youtu.be/H5kRwGb_xz4'><img  src='https://img.shields.io/badge/YouTube--red'></a>

<p align="center">
  <a href="https://youtu.be/H5kRwGb_xz4"><img src="https://github.com/poudel-bibek/Urban-Control/blob/main/Images/Craver.gif" alt="craver-walk" style="width:800px"/></a>
</p>

  



---

### Data
- Training [logs in wandb](https://api.wandb.ai/links/Fluidic-city/kt1tlg8f) 
- Trained Policy and config files
- Main results reported in the paper
- Rollout videos:
	| Method |Demand (1x)  | Demand (2.5x)| 
	|--|--|--|
	| Unsignalized | [link](https://youtu.be/XWkNqePOXPo) | [link](https://youtu.be/VC9E25Ys5RY) |
	| Signalized | [link](https://youtu.be/j9cxdP3pj_c) | [link](https://youtu.be/JaxmSJG-B5E) |
	| RL (Ours) | [link](https://youtu.be/80-0g7RuBIg)  | [link](https://youtu.be/HHrltmck6l8) |

- *Please note:
	- The initial 100-250 timesteps (10 - 25 seconds) are warmup period.
	- Although the horizon timesteps is same between methods, RL rollouts are longer  because of policy inference time.

---
### Setup & Training:

 ####  Requirements:
- SUMO version: [1.21](https://github.com/eclipse-sumo/sumo/releases/tag/v1_21_0)
- Python version: 3.12:
	- If using Anaconda:
- Install required packages
	```bash
	pip  install  requirements.txt
	```

#### Project Structure

-  `SUMO_files/`: Contains SUMO configuration and trip files
-  `models.py`: Neural network architecture definitions
-  `ppo_alg.py`: PPO algorithm implementation
-  `ppo_run.py`: Main script for running experiments
-  `sim_run.py`: SUMO environment wrapper
-  `wandb_sweep.py`: Hyperparameter tuning setup
  
 #### To Train

Step 1: Complete the setup
Open terminal In linux or wsl, use: ulimit -n 20000
to increase the limit on the number of file descriptors that can be opened by a process.
config file
```bash
command  =  f"netconvert --sumo-net-file {sumo_net_file} --plain-output-prefix {output_dir}/base_xml --plain-output.lanes true"
```

#### Key Parameters
-  `--step_length`: Simulation step length (default: 1.0)
-  `--action_duration`: Duration of each action (default: 10.0)
-  `--total_timesteps`: Total training timesteps (default: 100000)
-  `--max_timesteps`: Maximum steps per episode (default: 1000)
-  `--num_processes`: Number of parallel processes (default: 8)
For a complete list of parameters, refer to the argument parser in `ppo_run.py`.

---
### Citation
If you find this work useful in your own research, please cite the following:
```
@inproceedings{poudel2025control,
title={Joint Pedestrian and Vehicle Traffic Optimization in Urban Environments using Reinforcement Learning},
author={Poudel, Bibek and Wang, Xuan and Li, Weizi and Zhu, Lei and Heaslip, Kevin},
booktitle={arXiv preprint},
pages={},
year={2025},
organization={}
}
```
---
### Notes: 
- Developed and tested on Ubuntu 24.04, Windows 11 + WSL2

