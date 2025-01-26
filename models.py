import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
from torch.distributions import  Categorical

class CNNActorCritic(nn.Module):
    def __init__(self, in_channels, action_dim, **kwargs):
        """
        CNN Actor-Critic network with configurable size (designed to be compatible with hyper-parameter tuning)
        we are applying conv2d, the state should be 2d with a bunch of channels.
        Choices: 
            Small: 4 Conv layers, 3 Linear layers
            Medium: 6 Conv layers, 5 Linear layers

        Regularization: Dropout and Batch Norm (mitigation of internal covariate shift)
        Conservatively using pooling layers. Every piece of information is important, however we also want to avoid overfitting and keep parameters modest. 
        Dilation: For the first layer, experiment with dilation. (Disabled for now)

        During hyper-param sweep, the model size changes based on one of the dimension of the input (action_duration). 
        Even at high action durations, the model size is around 4.1M parameters. 
        """
        super(CNNActorCritic, self).__init__()
        self.in_channels = in_channels
        self.action_dim = action_dim 
        self.action_duration = kwargs.get('action_duration')
        self.per_timestep_state_dim = kwargs.get('per_timestep_state_dim')

        model_size = kwargs.get('model_size', 'medium')
        kernel_size = kwargs.get('kernel_size', 3)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        padding = kernel_size // 2

        if model_size == 'small':
            self.shared_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Dropout(dropout_rate)
            )
            hidden_dim = 128

        else:  # medium
            self.shared_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Dropout(dropout_rate)
            )
            hidden_dim = 256

        # Calculate the size of the flattened CNN output
        with torch.no_grad():
            sample_input = torch.zeros(1, self.in_channels, self.action_duration, self.per_timestep_state_dim) # E.g., (1,1,10,74) batch size of 1, 1 channel, 10 timesteps, 74 state dims
            cnn_output_size = self.shared_cnn(sample_input).shape[1]
            #print(f"\n\nCNN output size: {cnn_output_size}\n\n")

        # Actor-specific layers
        self.actor_layers = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, self.action_dim)
        )
        
        # Critic-specific layers
        self.critic_layers = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def actor(self, state,):
        shared_features = self.shared_cnn(state)
        action_logits = self.actor_layers(shared_features)
        #print(f"\n\nAction logits: {action_logits}\n\n")
        return action_logits
    
    def critic(self, state):
        shared_features = self.shared_cnn(state)
        return self.critic_layers(shared_features)
    
    def act(self, state):
        """
        Select an action based on the current state:
        * Simple Action: 
        - First two bits:
            - Intersection control on 4 mutually exclusive choices
            - Logits passed through softmax: 
                - Converts a vector of logits to a vector of probabilities that sum to 1
            - Feed the probabilities to a Categorical distribution
                - Then we sample to get one of the 4 choices

        - Next seven bits:
            - Midblock TL/ crosswalk signals
            - Bernoulli distribution (single probability parameter p)
            - Logits passed through sigmoid: 
                - Converts a single logit to a single probability value between 0 and 1
                - The probability is of getting 1
            - Feed each of the 7 probabilities to independent Bernoulli distribution
                - Then we sample to get either 0 or 1 for each of the 7 choices.
        
        - Modeling: 
            - We are modeling these choices independently. 
            - The assumption is that optimal action in one of them does not affect the others i.e., there is no possibility of coordination. 
            - Log probabilities can be summed for independent events. To get the joint log probability:
                - Sum the log probabilities of the individual midblock choices.
                - Add the log probability of the intersection choice.
    
        * Advanced Action: 
        """
        state_tensor = state.reshape(1, self.in_channels, self.action_duration, self.per_timestep_state_dim) # 1= batch size
        action_logits = self.actor(state_tensor)
        #print(f"\nAction logits: {action_logits}")

        # Simple action
        intersection_logits = action_logits[:, :4]  # First 4 logits for traffic light (4 choices)
        midblock_logits = action_logits[:, 4:]  # Last 7 logits for crosswalks (binary choices)
        # print(f"\nIntersection logits: {intersection_logits}")
        # print(f"Midblock logits: {midblock_logits}")
        
        intersection_probs = torch.softmax(intersection_logits, dim=1)
        intersection_dist = Categorical(intersection_probs)
        intersection_action = intersection_dist.sample() # This predicts 0, 1, 2, or 3
        # print(f"Intersection action: {intersection_action}")

        # represent intersection action in binary form
        intersection_action_binary = torch.zeros(2).to(intersection_action.device)
        intersection_action_binary[0] = (intersection_action // 2) % 2  # First bit
        intersection_action_binary[1] = intersection_action % 2  # Second bit
        # print(f"Intersection action in binary: {intersection_action_binary}")

        midblock_probs = torch.sigmoid(midblock_logits)
        midblock_dist = Bernoulli(midblock_probs)
        midblock_actions = midblock_dist.sample() # This predicts 0 or 1
        # print(f"Midblock actions: {midblock_actions}")
        
        combined_action = torch.cat([intersection_action_binary, midblock_actions.squeeze(0)], dim=0)
        log_prob = intersection_dist.log_prob(intersection_action) + midblock_dist.log_prob(midblock_actions).sum()
        # print(f"\nCombined action: {combined_action}, shape: {combined_action.shape}")
        # print(f"\nLog probability: {log_prob}, shape: {log_prob.shape}")

        return combined_action.int(), log_prob
    
        # Advanced action

    def evaluate(self, states, actions):
        """
        Evaluate a batch of states and actions.
        * Simple action: 

        - 1. Use the state input to get the current distributions (dont get new actions).
        - 2. Use the distribution and input actions provided (actions that were already-sampled; convert binary intersection actions to decimal) to get the log probs (how likely is the distribution to get the action).
        - 3. Combine to get the joint log probs (sum across the 7 Bernoullis and sum the intersection log prob).
            - sum operation is valid because the actions are independent. 
        - 4. Get the individual entropy (used as a regularization term to encourage exploration) and sum them to get the total entropy.
        - 5. Get the state values from critic.
        """

        action_logits = self.actor(states)
        print(f"\nAction logits: {action_logits}")

        # Simple action
        # 1. Get distributions 
        intersection_logits = action_logits[:, :4]
        midblock_logits = action_logits[:, 4:]
        print(f"\nIntersection logits: {intersection_logits}")
        print(f"Midblock logits: {midblock_logits}")

        intersection_probs = torch.softmax(intersection_logits, dim=1)
        intersection_dist = Categorical(intersection_probs)
        midblock_probs = torch.sigmoid(midblock_logits)
        midblock_dist = Bernoulli(midblock_probs)

        # 2.1 Convert binary intersection actions to decimal
        intersection_bits = actions[:, :2]  # (batch_size, 2)
        intersection_int = (2 * intersection_bits[:, 0] + intersection_bits[:, 1]).long()
        midblock_bits = actions[:, 2:].float() # (batch_size, 7)
        print(f"\nIntersection bits: {intersection_bits}, shape: {intersection_bits.shape}")
        print(f"\nIntersection int: {intersection_int}, shape: {intersection_int.shape}")
        print(f"\nMidblock bits: {midblock_bits}, shape: {midblock_bits.shape}")

        # 2.2 Get log probs
        intersection_log_probs = intersection_dist.log_prob(intersection_int)
        midblock_log_probs = midblock_dist.log_prob(midblock_bits)
        print(f"\nIntersection log probs: {intersection_log_probs}, shape: {intersection_log_probs.shape}")
        print(f"\nMidblock log probs: {midblock_log_probs}, shape: {midblock_log_probs.shape}")

        # 3. Combine to get the joint log probs 
        action_log_probs = intersection_log_probs + midblock_log_probs.sum(dim=1)
        print(f"\nAction log probs: {action_log_probs}, shape: {action_log_probs.shape}")

        # 4. Get the total entropy of the distributions.
        total_entropy = intersection_dist.entropy() + midblock_dist.entropy().sum(dim=1)
        print(f"\nTotal entropy: {total_entropy}, shape: {total_entropy.shape}")       

        # 5. Get the state values
        state_values = self.critic(states)
        print(f"\nState values: {state_values}, shape: {state_values.shape}")
        return action_log_probs, state_values, total_entropy
        
        # Advanced action

    def param_count(self, ):
        """
        Return a dict
        """
        actor_params = sum(p.numel() for p in self.actor_layers.parameters())
        critic_params = sum(p.numel() for p in self.critic_layers.parameters())
        shared_params = sum(p.numel() for p in self.shared_cnn.parameters())
        
        return {
            "actor_total": actor_params + shared_params,
            "critic_total": critic_params + shared_params,
            "total": actor_params + critic_params + shared_params,
            "shared": shared_params
        }
