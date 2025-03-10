## Joint Pedestrian and Vehicle Traffic Optimization in Urban Environments using Reinforcement Learning

<a  href='https://poudel-bibek.github.io/pdfs/projects/joint_control/'><img  src='https://img.shields.io/badge/arXiv--green'></a> <a  href='https://youtu.be/Tec3H72cDT4'><img  src='https://img.shields.io/badge/YouTube--red'></a>

<p align="center">
  <a href="https://youtu.be/Tec3H72cDT4"><img src="https://github.com/poudel-bibek/Urban-Control/blob/main/images/craver_3d.gif" alt="craver" style="width:800px"/></a>


### 📌 Overview

This project uses Proximal Policy Optimization (PPO) to jointly optimize traffic signal control for pedestrians and vehicles along the Craver Road corridor, featuring one intersection (with four signalized crosswalks) and seven midblock crossings. Our approach reduces waiting times by up to 52% for vehicles and 67% for pedestrians compared to traditional fixed-time signal control. 


<p align="center">
  <img src="https://github.com/poudel-bibek/Urban-Control/blob/main/images/system_overview.png" alt="System Overview" style="width:800px"/>
  <br>
  <em>(a) System overview and (b) agent actions in the Intersection and (c) Mid-Block crossings.</em>
</p>

 #### 📂 Project Structure

-  `simulation/`: Contains SUMO network, configuration and trip files
-  `simulation/env.py`: Python-SUMO interface
-  `ppo/ppo_alg.py`: PPO algorithm implementation (with parallelized sumo environments)
-  `ppo/models.py`: Policy network architectures
-  `main.py`: Main script for running experiments
-  `config.py`: Setup for training, evaluation
-  `sweep.py`: Hyperparameter tuning using wandb sweep

---

### 📊 Data
- [Training logs in wandb](https://api.wandb.ai/links/Fluidic-city/kt1tlg8f) 
- [Trained policy](https://github.com/poudel-bibek/Urban-Control/blob/main/saved_models) and [config file](https://github.com/user-attachments/files/19145941/config_Feb24_19-06-53.json)
- Unscaled trips: [pedestrian](https://github.com/poudel-bibek/Urban-Control/blob/main/simulation/original_pedtrips.xml), [vehicle](https://github.com/poudel-bibek/Urban-Control/blob/main/simulation/original_vehtrips.xml)
- Results reported in the paper (json files): [Traffic Signal](https://github.com/user-attachments/files/19145910/eval_tl.json), [Unsignalized](https://github.com/user-attachments/files/19145909/eval_unsignalized.json), [RL](https://github.com/user-attachments/files/19145911/eval_ppo.json).
- Rollout videos:
	| Method |Demand (1x)  | Demand (2.5x)| 
	|--|--|--|
	| Unsignalized | [link](https://youtu.be/XWkNqePOXPo) | [link](https://youtu.be/VC9E25Ys5RY) |
	| Signalized | [link](https://youtu.be/j9cxdP3pj_c) | [link](https://youtu.be/JaxmSJG-B5E) |
	| RL (Ours) | [link](https://youtu.be/80-0g7RuBIg)  | [link](https://youtu.be/HHrltmck6l8) |

---

### 🛠️ Setup & Training:

 #### 📋 Requirements:
- SUMO version: [1.21](https://github.com/eclipse-sumo/sumo/releases/tag/v1_21_0)
- Python version: 3.12 ([Anaconda 2024.06](https://repo.anaconda.com/archive/) recommended)
- Install required packages

	```bash
	pip install -r requirements.txt
	```

 ---
 #### 🏋️ Training:

- Step 1: Complete the setup

- Step 2: Open terminal, in linux or wsl run:

	```bash	
	ulimit -n 20000
	```

 to increase the limit on the number of file descriptors that can be opened by a process.

- Step 3: In the [config.py](https://github.com/poudel-bibek/Urban-Control/blob/main/config.py) file, set the `sweep`,`evaluate`, and `gui` to `False`.

- Step 4: Run the following command:

	```bash
	python main.py
	```

- Step 5: To view tensorboard logs, run the following command:

	```bash
	tensorboard --logdir=./runs
	```

#### Some important parameters that you can change in the [config.py](https://github.com/poudel-bibek/Urban-Control/blob/main/config.py) file during training:
-  `gui: True` to run the simulation with GUI.
-  `gpu: True` to run the simulation on GPU.
-  `sweep: True` to run the hyperparameter tuning.
-  `evaluate: True` to evaluate a trained policy.
-  `"step_length"`: Real-world time in seconds per simulation timestep (default: 1.0)
-  `"action_duration"`: Number of simulation timesteps for each action (default: 10)
-  `"total_timesteps"`: Total training timesteps (default: 8000000)
-  `"max_timesteps"`: Maximum simulation steps per episode (default: 600)
-  `"num_processes"`: Number of parallel processes (default: 8). Increase/ reduce this according to your CPU.

 ---
 #### 📈 Evaluation and Benchmarks 

- Set `eval_model_path` path in the [config.py](https://github.com/poudel-bibek/Urban-Control/blob/main/config.py) file. Modify other evaluation parameters as needed.
- Set `evaluate: True` in the [config.py](https://github.com/poudel-bibek/Urban-Control/blob/main/config.py) file.
- Run the following command:

	```bash
	python main.py
	```

- It will run benchmarks in the order: RL, Traffic Signal and Unsignalized as defined in the main.py file. If you want to run a specific benchmark, comment out the other two.

	```python
	ppo_results_path = eval(control_args, ppo_args, eval_args, policy_path=config['eval_model_path'], tl= False)
	tl_results_path = eval(control_args, ppo_args, eval_args, policy_path=None, tl= True, unsignalized=False) 
	unsignalized_results_path = eval(control_args, ppo_args, eval_args, policy_path=None, tl= True, unsignalized=True)
	```

- Benchmark results json files are saved in the `results` folder.

 ---
 #### 🔍 Hyperparameter sweep
- Set `sweep: True` in the [config.py](https://github.com/poudel-bibek/Urban-Control/blob/main/config.py) file.
- Modify the `create_sweep_config` method in [sweep.py](https://github.com/poudel-bibek/Urban-Control/blob/main/sweep.py) to set the parameters/ method to use.
- Run the following command:

	```bash
	python main.py
	```
- Create a [wandb account](https://wandb.ai/site) and [login/ authorize](https://wandb.ai/authorize). 
- You will also have to setup a project and set in the name in `self.project` in [sweep.py](https://github.com/poudel-bibek/Urban-Control/blob/main/sweep.py)

---
### 📝 Notes: 
- The initial `100-250` timesteps (randomly chosen) are warmup period. Defined in the `reset` method in [env.py](https://github.com/poudel-bibek/Urban-Control/blob/main/simulation/env.py)
- Although when episode horizon is same, rollouts for higher demands take longer because of higher CPU load.
- This code was developed and tested on Ubuntu 24.04 and Windows 11 + WSL2.
- ⚠️ If something fails, check the `sumo_logfile.txt` and `sumo_errorlog.txt` files in the `simulation` folder.

---
### 📖 Citation
If you find this work useful in your own research:
```
@inproceedings{poudel2025control,
  title={Joint Pedestrian and Vehicle Traffic Optimization in Urban Environments using Reinforcement Learning},
  author={Poudel, Bibek and Wang, Xuan and Li, Weizi and Zhu, Lei and Heaslip, Kevin},
  booktitle={arXiv preprint},
  volume={arXiv:},  
  year={2025},
  url={https://poudel-bibek.github.io/pdfs/projects/joint_control/},
}
```


