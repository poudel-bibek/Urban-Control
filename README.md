## Joint Pedestrian and Vehicle Traffic Optimization in Urban Environments using Reinforcement Learning

<a  href='https://arxiv.org/'><img  src='https://img.shields.io/badge/arXiv--green'></a> <a  href='https://youtu.be/Tec3H72cDT4'><img  src='https://img.shields.io/badge/YouTube--red'></a>

<p align="center">
  <a href="https://youtu.be/Tec3H72cDT4"><img src="https://github.com/poudel-bibek/Urban-Control/blob/main/images/craver_3d.gif" alt="craver" style="width:600px"/></a>
  <br>
  <em>The Craver road corridor in the city of Charlotte, North Carolina.</em>
</p>

### Overview

This project uses Reinforcement Learning to jointly optimize traffic signal control for pedestrians and vehicles along the Craver Road corridor, featuring one intersection (with four signalized crosswalks) and seven midblock crossings. Our approach reduces waiting times by up to 52% for vehicles and 67% for pedestrians compared to traditional fixed-time signal control. 


<p align="center">
  <img src="https://github.com/poudel-bibek/Urban-Control/blob/main/images/system_overview.png" alt="System Overview" style="width:600px"/>
  <br>
  <em>System overview and agent actions in the intersection and midblock crosswalks.</em>
</p>


---

### Data
- [Training logs in wandb](https://api.wandb.ai/links/Fluidic-city/kt1tlg8f) 
- [Trained policy and config file]()
- Original trips: [pedestrian](), [vehicle]()
- [Results reported in the paper]()
- Rollout videos:
	| Method |Demand (1x)  | Demand (2.5x)| 
	|--|--|--|
	| Unsignalized | [link](https://youtu.be/XWkNqePOXPo) | [link](https://youtu.be/VC9E25Ys5RY) |
	| Signalized | [link](https://youtu.be/j9cxdP3pj_c) | [link](https://youtu.be/JaxmSJG-B5E) |
	| RL (Ours) | [link](https://youtu.be/80-0g7RuBIg)  | [link](https://youtu.be/HHrltmck6l8) |

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

	-  `simulation/`: Contains SUMO configuration and trip files
	-  `models.py`: Neural network architecture definitions
	-  `ppo_alg.py`: PPO algorithm implementation
	-  `ppo_run.py`: Main script for running experiments
	-  `sim_run.py`: SUMO environment wrapper
	-  `sweep.py`: Hyperparameter tuning using wandb sweep
  
 #### To Train

	Step 1: Complete the setup

	Step 2: Open terminal In linux or wsl and run:
		```bash
		ulimit -n 20000
		```
	to increase the limit on the number of file descriptors that can be opened by a process.

	Step 3: In the config.py file, set the `sweep`,`evaluate`, and `gui` to `False`.

	Step 4: Run the following command:
		```bash
		python main.py
		```

	#### Some important parameters that you can add to the command:
	-  `--gui` to run the simulation with GUI.
	-  `--sweep` to run the hyperparameter tuning.
	-  `--evaluate` to evaluate the trained policy.
	-  `--eval_model_path` to specify the path to the trained policy.
	-  `--eval_worker_device` to specify the device to run the evaluation on.
	-  `--step_length`: Simulation step length (default: 1.0)
	-  `--action_duration`: Duration of each action (default: 10.0)
	-  `--total_timesteps`: Total training timesteps (default: 100000)
	-  `--max_timesteps`: Maximum steps per episode (default: 1000)
	-  `--num_processes`: Number of parallel processes (default: 8)
	For a complete list of parameters, refer to the argument parser in `ppo_run.py`.

 #### To Run a sweep
 

---
### Notes: 
- The initial 100-250 timesteps (10 - 25 seconds) are warmup period.
- Although the horizon timesteps is same between methods, rollouts for higher demands are longer because of increased CPU load.
- Developed and tested on Ubuntu 24.04, Windows 11 + WSL2

---
### Citation
If you find this work useful in your own research:
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


