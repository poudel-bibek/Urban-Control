import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models import CNNActorCritic, MLPActorCritic

class Memory:
    """
    Storage class for saving experience from interactions with the environment.
    These memories will be made in CPU but loaded in GPU for the policy update.
    """
    def __init__(self,):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def append(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action) 
        self.logprobs.append(logprob)
        self.rewards.append(reward) # these are scalars
        self.is_terminals.append(done) # these are scalars

class PPO:
    """
    This implementation is parallelized using Multiprocessing i.e. multiple CPU cores each running a separate process.
    Multiprocessing vs Multithreading:
    - In the CPython implementation, the Global Interpreter Lock (GIL) is a mechanism used to prevent multiple threads from executing Python bytecodes at once. 
    - This lock is necessary because CPython is not thread-safe, i.e., if multiple threads were allowed to execute Python code simultaneously, they could potentially interfere with each other, leading to data corruption or crashes. 
    - The GIL prevents this by ensuring that only one thread can execute Python code at any given time.
    - Since only one thread can execute Python code at a time, programs that rely heavily on threading for parallel execution may not see the expected performance gains.
    - In contrast, multiprocessing allows multiple processes to execute Python code in parallel, bypassing the GIL and taking full advantage of multiple CPU cores.
    - However, multiprocessing has higher overhead than multithreading due to the need to create separate processes and manage inter-process communication.
    - In Multiprocessing, we create separate processes, each with its own Python interpreter and memory space
    """
    def __init__(self, 
                 model_dim, # could be state_dim for mlp, in_channels for cnn, in_channels for gatv2
                 action_dim, # is max_proposals for gatv2
                 device, 
                 lr, 
                 gamma, 
                 K_epochs, 
                 eps_clip, 
                 ent_coef, 
                 vf_coef, 
                 batch_size, 
                 gae_lambda,
                 max_grad_norm,
                 model_type,
                 model_kwargs):
        
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm

        if model_type == "cnn":
            self.policy = CNNActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device)
            self.policy_old = CNNActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device) # old policy network (used for importance sampling)
        elif model_type == "mlp":
            self.policy = MLPActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device)
            self.policy_old = MLPActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device) # old policy network (used for importance sampling)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}. Must be 'cnn' or 'mlp'.")
        
        param_counts = self.policy.param_count()
        print(f"\nModel parameters:")
        for k, v in param_counts.items():
            print(f"\t{k}: {v}")

        # Copy the parameters from the current policy to the old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Set up the optimizer for the current policy network
        self.initial_lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.initial_lr)
        self.total_iterations = None  # Will be set externally.
    
    def update_learning_rate(self, iteration):
        """
        Linear annealing. At the end of training, the learning rate is 0.
        """
        if self.total_iterations is None:
            raise ValueError("total_iterations must be set before calling update_learning_rate")
        
        frac = 1.0 - (iteration / self.total_iterations)
        new_lr = frac * self.initial_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    
    def compute_gae(self, rewards, values, is_terminals, gamma, gae_lambda):
        """
        Compute the Generalized Advantage Estimation (GAE) for the collected experiences.
        For most steps in the sequence, we use the value estimate of the next state to calculate the TD error.
        For the last step (step == len(rewards) - 1), we use the value estimate of the current state. 

        """ 
        advantages = [0] * len(rewards)
        gae = 0.0
        next_value = 0.0

        values = values.cpu().numpy()
        print(f"\nValues 2: {values.shape}")

        # First, we iterate through the rewards in reverse order.
        for step in reversed(range(len(rewards))):

            # If its the terminal step (which has no future) or if its the last step in our collected experiences (which may not be terminal).
            if is_terminals[step]:
                next_value = 0.0
                gae = 0.0

            # For each step, we calculate the TD error (delta). Equation 12 in the paper. delta = r + γV(s') - V(s)
            delta = rewards[step] + gamma * next_value * (1 - int(is_terminals[step])) - values[step]

            # Equation 11 in the paper. GAE(t) = δ(t) + (γλ)δ(t+1) + (γλ)²δ(t+2) + ...
            gae = delta + gamma * gae_lambda * (1 - int(is_terminals[step])) * gae 
            advantages[step] = gae

            # Update the next value for the next step
            next_value = values[step]

        #print(f"\nAdvantages: {advantages}, shape: {len(advantages)}")
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def update(self, memories):
        """
        Update the policy and value networks using the collected experiences.
        memories = combined memories from all processes. 
        - Includes GAE
        - For the choice between KL divergence vs. clipping, we use clipping.
    
        The paper expresses the loss as maximization objective. We convert it to minimization by changing the sign.
        """

        old_states = torch.stack(memories.states).to(self.device)
        # print(f"\nOld states before reshape: {old_states.shape}")

        # From [128, 10, 96] to [128, 1, 10, 96]
        old_states = old_states.unsqueeze(1) # Reshape states to have batch size first
        # print(f"\nOld states after reshape: {old_states.shape}")

        with torch.no_grad():
            values = self.policy_old.critic(old_states) # Use the old policy to get the value estimate.
            values = values.squeeze() # Shape is [128]

        print(f"\nStates: {old_states.shape}")
        print(f"\nActions: {len(memories.actions)}")
        print(f"\nLogprobs: {len(memories.logprobs)}")
        print(f"\nValues: {values.shape}")

        # Compute GAE
        advantages = self.compute_gae(memories.rewards, values, memories.is_terminals, self.gamma, self.gae_lambda)

        # Advantage = how much better is it to take a specific action compared to the average action. 
        # GAE = difference between the empirical return and the value function estimate.
        # advantages + val = Reconstruction of empirical returns. Because we want the critic to predict the empirical returns.
        returns = advantages + values
        print(f"\nAdvantages: {advantages.shape}")
        print(f"\nReturns: {returns.shape}")

        # Normalize the advantages (only for use in policy loss calculation) after they have been added to get returns.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Small constant to prevent division by zero
        print(f"\nAdvantages after normalization: {advantages.shape}")

        # Process actions and logprobs
        old_actions = torch.stack(memories.actions).to(self.device)
        old_logprobs = torch.stack(memories.logprobs).to(self.device)

        # Create a dataloader for mini-batching 
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_loss = 0

        # Optimize policy for K epochs (terminology used in PPO paper)
        for _ in range(self.K_epochs):
            for states_batch, actions_batch, old_logprobs_batch, advantages_batch, returns_batch in dataloader:

                # Evaluating old actions and values using current policy network
                logprobs, state_values, dist_entropy = self.policy.evaluate(states_batch, actions_batch)

                # Finding the ratio (pi_theta / pi_theta_old) for importance sampling (we want to use the samples obtained from old policy to get the new policy)
                ratios = torch.exp(logprobs - old_logprobs_batch.squeeze(-1)) # New log probs need to remain attached to the graph.
                # print(f"\nLogprobs: {logprobs.shape}")
                # print(f"\nOld logprobs: {old_logprobs_batch.squeeze(-1).shape}")
                # print(f"\nRatios: {ratios.shape}")

                # Finding Surrogate Loss
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_batch
                
                # Calculate policy and value losses
                policy_loss = torch.min(surr1, surr2).mean() # Equation 7 in the paper
                print(f"\nPolicy loss: {policy_loss.item()}")

                # The negative sign ensures that the optimizer maximizes the PPO objective by minimizing the loss function. It is correct and necessary.
                value_loss = 0.5 * ((state_values - returns_batch) ** 2).mean() # MSE. Value loss is clipped (0.5)
                print(f"\nValue loss: {value_loss.item()}")

                print(f"\nDist entropy: {dist_entropy.shape}, dist entropy: {dist_entropy}")
                entropy_loss = dist_entropy.mean()
                print(f"\nEntropy loss: {entropy_loss.item()}")

                # Total loss. Negate for minimization.
                loss = -1 * (policy_loss - self.vf_coef * value_loss + self.ent_coef * entropy_loss) # Equation 9 in the paper
                print(f"\nTotal Loss: {loss.item()}")

                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) # Clipping to prevent exploding gradients
                self.optimizer.step()

                # Accumulate losses
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy_loss += entropy_loss.item()
        
        num_batches = len(dataloader) * self.K_epochs
        avg_policy_loss /= num_batches
        avg_value_loss /= num_batches
        avg_entropy_loss /= num_batches

        # print("\nPolicy New params:")
        # for name, param in self.policy.named_parameters():
        #     print(f"{name}: {param.data}")

        # print("\n\n\nPolicy Old params:")
        # for name, param in self.policy_old.named_parameters():
        #     print(f"{name}: {param.data}")

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f"\nPolicy updated with avg_policy_loss: {avg_policy_loss}\n") 

        # Return the average batch loss per epoch
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_policy_loss + self.vf_coef * avg_value_loss - self.ent_coef * avg_entropy_loss
        }