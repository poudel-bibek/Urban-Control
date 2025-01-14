import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models import CNNActorCritic

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
        self.policy = CNNActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device)
        self.policy_old = CNNActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device) # old policy network (used for importance sampling)

        param_counts = self.policy.param_count()
        print(f"\tActor parameters: {param_counts['actor_total']}")
        print(f"\tCritic parameters: {param_counts['critic_total']}")
        print(f"\tShared parameters: {param_counts['shared']}\n")

        # Copy the parameters from the current policy to the old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # self.policy.share_memory() # Share the policy network across all processes. Any tensor can be shared across processes by calling this.
        # self.policy_old.share_memory() # Share the old policy network across all processes. 

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
        advantages = []
        gae = 0

        # First, we iterate through the rewards in reverse order.
        for step in reversed(range(len(rewards))):

            # If its the terminal step (which has no future) or if its the last step in our collected experiences (which may not be terminal).
            if is_terminals[step] or step == len(rewards) - 1:
                next_value = 0
                gae = 0
            else:
                next_value = values[step + 1]
            # For each step, we calculate the TD error (delta). Equation 12 in the paper. delta = r + γV(s') - V(s)
            delta = rewards[step] + gamma * next_value * (1 - is_terminals[step]) - values[step]

            # Equation 11 in the paper. GAE(t) = δ(t) + (γλ)δ(t+1) + (γλ)²δ(t+2) + ...
            gae = delta + gamma * gae_lambda * (1 - is_terminals[step]) * gae # (1 - dones[step]) term ensures that the advantage calculation stops at episode boundaries.
            advantages.insert(0, gae) # Insert the advantage at the beginning of the list so that it is in the same order as the rewards.
        #print(f"\nAdvantages: {advantages}, shape: {len(advantages)}")
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def update(self, memories):
        """
        Update the policy and value networks using the collected experiences.
        memories = combined memories from all processes. 
        - Includes GAE
        - For the choice between KL divergence vs. clipping, we use clipping.
    
         However, the policy network expects a tensor.
        Therefore, we need to convert the graph to a tensor.
        """

        print(f"\nUpdating policy")
        combined_memory = Memory()
        for memory in memories:
            combined_memory.actions.extend(memory.actions)
            combined_memory.states.extend(memory.states)
            combined_memory.logprobs.extend(memory.logprobs)
            combined_memory.rewards.extend(memory.rewards)
            combined_memory.is_terminals.extend(memory.is_terminals)

        old_states = torch.stack(combined_memory.states).to(self.device)
        # From [128, 10, 40] to [128, 1, 10, 40]
        old_states = old_states.unsqueeze(1) # Reshape states to have batch size first

        with torch.no_grad():
            values = self.policy.critic(old_states).squeeze()

        print(f"\nStates: {old_states.shape}")
        print(f"\nActions: {len(combined_memory.actions)}")
        print(f"\nLogprobs: {len(combined_memory.logprobs)}")
        print(f"\nValues: {values.shape}")

        # Compute GAE
        advantages = self.compute_gae(combined_memory.rewards, values, combined_memory.is_terminals, self.gamma, self.gae_lambda)

        # Advantage = how much better is it to take a specific action compared to the average action. 
        # GAE = difference between the empirical return and the value function estimate.
        # advantages + val = Reconstruction of empirical returns. Because we want the critic to predict the empirical returns.
        returns = advantages + values

        # Normalize the advantages (only for use in policy loss calculation) after they have been added to get returns.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Small constant to prevent division by zero
        
        # Process actions and logprobs (same for both agents)
        old_actions = torch.stack(combined_memory.actions).detach().to(self.device)
        old_logprobs = torch.stack(combined_memory.logprobs).detach().to(self.device)

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
                logprobs = torch.FloatTensor(logprobs)
                state_values = torch.FloatTensor(state_values)

                # Finding the ratio (pi_theta / pi_theta_old) for importance sampling (we want to use the samples obtained from old policy to get the new policy)
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_batch
                
                # Calculate policy and value losses
                # TODO: Is the mean necessary here? In policy loss and entropy loss. Probably yes, for averaging across the batch.
                policy_loss = -torch.min(surr1, surr2).mean() # Equation 7 in the paper
                # The negative sign ensures that the optimizer maximizes the PPO objective by minimizing the loss function. It is correct and necessary.
                value_loss = ((state_values - returns_batch) ** 2).mean() # MSE 
                entropy_loss = dist_entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss # Equation 9 in the paper
                
                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate losses
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy_loss += entropy_loss.item()
        
        num_batches = len(dataloader) * self.K_epochs
        avg_policy_loss /= num_batches
        avg_value_loss /= num_batches
        avg_entropy_loss /= num_batches

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