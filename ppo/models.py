import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
from torch.distributions import  Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogoal initialization of weights and Constant initialization of biases.
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNActorCritic(nn.Module):
    def __init__(self, in_channels, action_dim, **kwargs):
        """
        This CNN-based Actor-Critic architecture uses two separate convolutional
        backbones for the actor and the critic respectively while keeping the head's
        structure (MLP layers, normalization, and activation) the same.

        The choices:
          - Small: 3 Conv layers, 3 Linear layers with hidden_dim=128.
          - Medium: 5 Conv layers, 3 Linear layers with hidden_dim=256.

        Some kwargs:
            action_duration: height of the 2D input.
            per_timestep_state_dim: width of the 2D input.
            model_size: 'small' or 'medium'
            kernel_size: kernel size for the conv layers.
            activation: one of ['tanh', 'relu', 'leakyrelu']
        """
        super(CNNActorCritic, self).__init__()
        self.in_channels = in_channels
        self.action_dim = action_dim 
        self.action_duration = kwargs.get('action_duration')
        self.per_timestep_state_dim = kwargs.get('per_timestep_state_dim')

        model_size = kwargs.get('model_size')
        kernel_size = kwargs.get('kernel_size')
        padding = kernel_size // 2
        activation_str = kwargs.get('activation')

        if activation_str == "tanh":
            activation = nn.Tanh
        elif activation_str == "relu":
            activation = nn.ReLU
        elif activation_str == "leakyrelu":
            activation = nn.LeakyReLU
        else:
            raise ValueError("Unknown activation chosen.")

        # Build separate CNN backbones for actor and critic.
        if model_size == 'small':
            self.actor_cnn = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                activation(),
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(32),
                activation(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                activation(),
                nn.Flatten(),
            )
            self.critic_cnn = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                activation(),
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(32),
                activation(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                activation(),
                nn.Flatten(),
            )
            hidden_dim = 128

        else:  # medium
            self.actor_cnn = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                activation(),
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(32),
                activation(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                activation(),
                nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(128),
                activation(),
                nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                activation(),
                nn.Flatten(),
            )
            self.critic_cnn = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                activation(),
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(32),
                activation(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                activation(),
                nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(128),
                activation(),
                nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                activation(),
                nn.Flatten(),
            )
            hidden_dim = 256

        # Calculate CNN output dimensions separately for actor and critic.
        with torch.no_grad():
            sample_input = torch.zeros(1, self.in_channels, self.action_duration, self.per_timestep_state_dim)
            actor_cnn_output_size = self.actor_cnn(sample_input).shape[1]
            critic_cnn_output_size = self.critic_cnn(sample_input).shape[1]

        # Build the actor head.
        self.actor_layers = nn.Sequential(
            layer_init(nn.Linear(actor_cnn_output_size, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            activation(),
            layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LayerNorm(hidden_dim // 2),
            activation(),
            layer_init(nn.Linear(hidden_dim // 2, self.action_dim))
        )
        
        # Build the critic head.
        self.critic_layers = nn.Sequential(
            layer_init(nn.Linear(critic_cnn_output_size, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            activation(),
            layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LayerNorm(hidden_dim // 2),
            activation(),
            layer_init(nn.Linear(hidden_dim // 2, 1))
        )

    def actor(self, state):
        """
        Processes the input state through the actor's CNN backbone and MLP head
        to produce the raw action logits.
        """
        features = self.actor_cnn(state)
        logits = self.actor_layers(features)
        return logits
    
    def critic(self, state):
        """
        Processes the input state through the critic's CNN backbone and MLP head
        to produce the state-value estimate.
        """
        if state.ndim == 3:
            state = state.unsqueeze(0)
        features = self.critic_cnn(state)
        return self.critic_layers(features)
    
    def act(self, state):
        """
        Select an action based on the current state:
          - The first 4 logits correspond to an intersection (Categorical distribution).
          - The next 7 logits correspond to midblock signals (Bernoulli distribution).
        Adjusts the input shape if necessary and returns the combined action and the joint log probability.
        """
        # Adjust the state shape if needed.
        if state.ndim == 2:  # shape: [action_duration, per_timestep_state_dim]
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.ndim == 3:  # shape: [batch, action_duration, per_timestep_state_dim]
            state = state.unsqueeze(1)

        action_logits = self.actor(state)
        intersection_logits = action_logits[:, :4]  # first 4 logits for intersection choices
        midblock_logits = action_logits[:, 4:]        # last 7 logits for midblock binary choices

        intersection_dist = Categorical(logits=intersection_logits)
        intersection_action = intersection_dist.sample()

        midblock_dist = Bernoulli(logits=midblock_logits)
        midblock_actions = midblock_dist.sample()

        # Combine the outputs; note that squeeze is used to remove any extra dimension.
        combined_action = torch.cat([intersection_action, midblock_actions.squeeze(0)], dim=0)
        log_prob = intersection_dist.log_prob(intersection_action) + midblock_dist.log_prob(midblock_actions).sum()
        return combined_action.int(), log_prob

    def evaluate(self, states, actions):
        """
        For a batch of states and already-sampled actions, compute:
          1. The joint log probability of taking each action.
          2. The entropy of the actions' distributions (used for exploration regularization).
          3. The critic's state value estimates.
        """
        # Reshape states to [batch, channel, action_duration, per_timestep_state_dim]
        states = states.unsqueeze(0)
        states = states.permute(1, 0, 2, 3)
        action_logits = self.actor(states)

        # Split the logits
        intersection_logits = action_logits[:, :4]
        midblock_logits = action_logits[:, 4:]

        intersection_dist = Categorical(logits=intersection_logits)
        midblock_dist = Bernoulli(logits=midblock_logits)

        # Actions: intersection is the first column, midblock the rest.
        intersection_action = actions[:, :1].squeeze(1).long()  # Categorical expects long
        midblock_actions = actions[:, 1:].float()

        intersection_log_probs = intersection_dist.log_prob(intersection_action)
        midblock_log_probs = midblock_dist.log_prob(midblock_actions)
        action_log_probs = intersection_log_probs + midblock_log_probs.sum(dim=1)

        total_entropy = intersection_dist.entropy() + midblock_dist.entropy().sum(dim=1)

        state_values = self.critic(states)
        return action_log_probs, state_values, total_entropy

    def param_count(self):
        """
        Returns a dictionary with parameter counts of the actor and critic networks separately.
        """
        actor_params = sum(p.numel() for p in self.actor_layers.parameters()) + \
                       sum(p.numel() for p in self.actor_cnn.parameters())
        critic_params = sum(p.numel() for p in self.critic_layers.parameters()) + \
                        sum(p.numel() for p in self.critic_cnn.parameters())
        
        return {
            "actor_total": actor_params,
            "critic_total": critic_params,
            "total": actor_params + critic_params
        }
    
class CNNActorCriticShared(nn.Module):
    def __init__(self, in_channels, action_dim, **kwargs):
        """
        Model choices: 
            Small: 3 Conv layers, 3 Linear layers
            Medium: 5 Conv layers, 3 Linear layers

        - Applying conv2d, the state should be 2d with a bunch of channels (1)
        - Regularization: Dropout and Batch Norm
        - Using strided convolutions instead of pooling layers
        - Shared CNN backbone useful because "feature extraction" is useful for both actor and critic.
        """
        super(CNNActorCriticShared, self).__init__()
        self.in_channels = in_channels
        self.action_dim = action_dim 
        self.action_duration = kwargs.get('action_duration')
        self.per_timestep_state_dim = kwargs.get('per_timestep_state_dim')

        model_size = kwargs.get('model_size')
        kernel_size = kwargs.get('kernel_size')
        padding = kernel_size // 2
        # dropout_rate = kwargs.get('dropout_rate')
        activation = kwargs.get('activation')

        if activation == "tanh":
            activation = nn.Tanh
        elif activation == "relu":
            activation = nn.ReLU
        elif activation == "leakyrelu":
            activation = nn.LeakyReLU

        if model_size == 'small':
            self.shared_cnn = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                activation(),
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=2, padding=padding),  # Strided Conv 
                nn.BatchNorm2d(32),
                activation(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                activation(),
                nn.Flatten(),
                #nn.Dropout(dropout_rate)
                )
            hidden_dim = 128

        else:  # medium
            self.shared_cnn = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                activation(),
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=2, padding=padding),  # Strided Conv 
                nn.BatchNorm2d(32),
                activation(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                activation(),
                nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=padding), # Strided Conv 
                nn.BatchNorm2d(128),
                activation(),
                nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                activation(),
                nn.Flatten(),
                #nn.Dropout(dropout_rate)
                )
            hidden_dim = 256

        # Calculate the size of the flattened CNN output
        with torch.no_grad():
            sample_input = torch.zeros(1, self.in_channels, self.action_duration, self.per_timestep_state_dim)
            cnn_output_size = self.shared_cnn(sample_input).shape[1]
            #print(f"\n\nCNN output size: {cnn_output_size}\n\n")

        self.actor_layers = nn.Sequential(
            layer_init(nn.Linear(cnn_output_size, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            activation(),
            # nn.Dropout(dropout_rate),
            layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LayerNorm(hidden_dim // 2),
            activation(),

            # nn.Dropout(dropout_rate),
            layer_init(nn.Linear(hidden_dim // 2, self.action_dim))
        )
        
        self.critic_layers = nn.Sequential(
            layer_init(nn.Linear(cnn_output_size, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            activation(),
            # nn.Dropout(dropout_rate),
            layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LayerNorm(hidden_dim // 2),
            activation(),

            # nn.Dropout(dropout_rate),
            layer_init(nn.Linear(hidden_dim // 2, 1))
        )


    def actor(self, state):
        shared_features = self.shared_cnn(state)
        action_logits = self.actor_layers(shared_features)
        return action_logits
    
    def critic(self, state):
        if state.ndim == 3:
            state = state.unsqueeze(0)
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
        """
        # Check the number of dimensions of state.
        if state.ndim == 2: # coming from act
            # State is of shape [action_duration, per_timestep_state_dim].
            # Add batch and channel dimensions: becomes [1, 1, action_duration, per_timestep_state_dim].
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.ndim == 3: # coming from evaluate
            # State is of shape [batch, action_duration, per_timestep_state_dim].
            # Add the channel dimension: becomes [batch, 1, action_duration, per_timestep_state_dim].
            state = state.unsqueeze(1)

        action_logits = self.actor(state)
        #print(f"\nAction logits: {action_logits}")

        # Simple action
        intersection_logits = action_logits[:, :4]  # First 4 logits for traffic light (4 choices)
        midblock_logits = action_logits[:, 4:]  # Last 7 logits for crosswalks (binary choices)
        # print(f"\nIntersection logits: {intersection_logits}")
        # print(f"Midblock logits: {midblock_logits}")
        
        intersection_dist = Categorical(logits=intersection_logits)
        intersection_action = intersection_dist.sample() # This predicts 0, 1, 2, or 3
        # print(f"Intersection action: {intersection_action}, shape: {intersection_action.shape}")

        midblock_dist = Bernoulli(logits=midblock_logits)
        midblock_actions = midblock_dist.sample() # This predicts 0 or 1
        # print(f"Midblock actions: {midblock_actions}, shape: {midblock_actions.shape}")
        
        combined_action = torch.cat([intersection_action, midblock_actions.squeeze(0)], dim=0)
        log_prob = intersection_dist.log_prob(intersection_action) + midblock_dist.log_prob(midblock_actions).sum()
        # print(f"\nCombined action: {combined_action}, shape: {combined_action.shape}")
        return combined_action.int(), log_prob

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
        # print(f"\nStates shape: {states.shape}")
        states = states.unsqueeze(0)
        # reshape to have [C, B, H, W] instead of [B, C, H, W]
        states = states.permute(1, 0, 2, 3)
        # print(f"\nStates shape after unsqueeze: {states.shape}")
        
        action_logits = self.actor(states)
        # print(f"\nAction logits: {action_logits}")

        # Simple action
        # 1. Get distributions 
        intersection_logits = action_logits[:, :4]
        midblock_logits = action_logits[:, 4:]
        # print(f"\nIntersection logits: {intersection_logits}")
        # print(f"Midblock logits: {midblock_logits}")

        intersection_dist = Categorical(logits=intersection_logits)
        midblock_dist = Bernoulli(logits=midblock_logits)

        # 2. Get log probs
        intersection_action = actions[:, :1].squeeze(1).long()  # (batch_size, 1) # Categorical expects long
        midblock_actions = actions[:, 1:].float() # (batch_size, 7)
        # print(f"\nIntersection action: {intersection_action}, shape: {intersection_action.shape}")
        # print(f"\nMidblock actions: {midblock_actions}, shape: {midblock_actions.shape}")
        intersection_log_probs = intersection_dist.log_prob(intersection_action)
        midblock_log_probs = midblock_dist.log_prob(midblock_actions)
        # print(f"\nIntersection log probs: {intersection_log_probs}, shape: {intersection_log_probs.shape}")
        # print(f"\nMidblock log probs: {midblock_log_probs}, shape: {midblock_log_probs.shape}")

        # 3. Combine to get the joint log probs 
        action_log_probs = intersection_log_probs + midblock_log_probs.sum(dim=1)
        # print(f"\nAction log probs: {action_log_probs}, shape: {action_log_probs.shape}")

        # 4. Get the total entropy of the distributions.
        total_entropy = intersection_dist.entropy() + midblock_dist.entropy().sum(dim=1)
        # print(f"\nTotal entropy: {total_entropy}, shape: {total_entropy.shape}")       

        # 5. Get the state values
        state_values = self.critic(states)
        # print(f"\nState values: {state_values}, shape: {state_values.shape}")
        return action_log_probs, state_values, total_entropy

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

class MLPActorCritic(nn.Module):
    def __init__(self, in_channels, action_dim, **kwargs):
        """
        - MLP Actor-Critic network in two sizes: small, medium. 
        - Expects inputs of shape (B, in_channels, action_duration, per_timestep_state_dim) then flattens to (B, -1).
        - No shared backbone as there is no feature extraction.
        """
        super(MLPActorCritic, self).__init__()
        in_channels = in_channels
        action_duration = kwargs.get('action_duration')
        per_timestep_state_dim = kwargs.get('per_timestep_state_dim')

        with torch.no_grad():
            sample_input = torch.zeros(1, in_channels, action_duration, per_timestep_state_dim)
            self.input_dim = sample_input.numel()  # total number of features, e.g. 1 * c * d * s

        if kwargs.get('activation') == "tanh":
            activation = nn.Tanh
        elif kwargs.get('activation') == "relu":
            activation = nn.ReLU
        elif kwargs.get('activation') == "leakyrelu":
            activation = nn.LeakyReLU
        # dropout_rate = kwargs.get('dropout_rate')

        model_size = kwargs.get('model_size')
        if model_size == 'small':
            actor_hidden_sizes = [256, 128, 64, 32]
            critic_hidden_sizes = [256, 128, 64, 32]
        elif model_size == 'medium':
            actor_hidden_sizes = [512, 256, 128, 64, 32]
            critic_hidden_sizes = [512, 256, 128, 64, 32]
        # Build networks
        # actor
        actor_layers = []
        input_size_actor = self.input_dim
        for h in actor_hidden_sizes:
            actor_layers.append(layer_init(nn.Linear(input_size_actor, h)))
            actor_layers.append(nn.LayerNorm(h))  # Add LayerNorm after linear layer
            actor_layers.append(activation())
            # actor_layers.append(nn.Dropout(dropout_rate)) # Disabled for now
            input_size_actor = h
        self.actor_layers = nn.Sequential(*actor_layers)
        self.actor_logits = layer_init(nn.Linear(input_size_actor, action_dim)) # Last layer, no activation

        # critic 
        critic_layers = []
        input_size_critic = self.input_dim
        for h in critic_hidden_sizes:
            critic_layers.append(layer_init(nn.Linear(input_size_critic, h)))
            critic_layers.append(nn.LayerNorm(h))  # Add LayerNorm after linear layer
            critic_layers.append(activation())
            # critic_layers.append(nn.Dropout(dropout_rate)) # Disabled for now
            input_size_critic = h
        self.critic_layers = nn.Sequential(*critic_layers)
        self.critic_value = layer_init(nn.Linear(input_size_critic, 1)) # Last layer, no activation

    def actor(self, state):
        """
        First Flatten the input from 4D (B, C, D, S) to 2D (B, -1)
        Returns the raw action logits from the actor head.
        """
        bsz = state.size(0)
        flat = state.view(bsz, -1)  # shape: (B, in_channels*action_duration*per_timestep_state_dim)
        return self.actor_logits(self.actor_layers(flat))

    def critic(self, state):
        """
        First Flatten the input from 4D (B, C, D, S) to 2D (B, -1)
        Returns the scalar state-value V(s).
        """
        bsz = state.size(0)
        flat = state.view(bsz, -1)  # shape: (B, in_channels*action_duration*per_timestep_state_dim)
        return self.critic_value(self.critic_layers(flat))

    def act(self, state):
        """
        Sample an action exactly like in the CNN version:
          - intersection action from first 4 logits (Categorical)
          - midblock from next 7 logits (Bernoulli)
        """
        # print("Sampling...")
        state = state.reshape(1, 1, state.shape[0], state.shape[1])
        action_logits = self.actor(state)

        # The first 4 logits => intersection (Categorical)
        intersection_logits = action_logits[:, :4]
        # intersection_probs = torch.softmax(intersection_logits, dim=1)
        intersection_dist = Categorical(logits=intersection_logits)
        intersection_action = intersection_dist.sample()  # [1]


        # The next 7 logits => midblock (Bernoulli)
        midblock_logits = action_logits[:, 4:]
        # midblock_probs = torch.sigmoid(midblock_logits)
        midblock_dist = Bernoulli(logits=midblock_logits)
        midblock_actions = midblock_dist.sample()  # shape [1,7]

        # print(f"\nIntersection logits: {intersection_logits}")
        # print(f"\nMidblock logits: {midblock_logits}")

        combined_action = torch.cat([intersection_action, midblock_actions.squeeze(0)], dim=0)
        log_prob = intersection_dist.log_prob(intersection_action) + \
                   midblock_dist.log_prob(midblock_actions).sum()

        # print(f"\nAction Log probability: {log_prob}, shape: {log_prob.shape}")
        return combined_action.int(), log_prob


    def evaluate(self, states, actions):
        """
        Evaluate a batch of states and pre-sampled actions. Same logic as the CNN version.
        """
        # print("Evaluating...")
        action_logits = self.actor(states)
        intersection_logits = action_logits[:, :4]
        midblock_logits = action_logits[:, 4:]

        # Distributions
        # intersection_probs = torch.softmax(intersection_logits, dim=1)
        intersection_dist = Categorical(logits=intersection_logits)
        # midblock_probs = torch.sigmoid(midblock_logits)
        midblock_dist = Bernoulli(logits=midblock_logits)

        # Actions in shape (B,1) for intersection, (B,7) for midblock
        intersection_action = actions[:, :1].squeeze(1).long() # Categorical expects long
        midblock_actions = actions[:, 1:].float()

        intersection_log_probs = intersection_dist.log_prob(intersection_action)
        # print(f"\nIntersection log probs: {intersection_log_probs}, shape: {intersection_log_probs.shape}")
        midblock_log_probs = midblock_dist.log_prob(midblock_actions)
        # print(f"\nMidblock log probs: {midblock_log_probs}, shape: {midblock_log_probs.shape}")
        action_log_probs = intersection_log_probs + midblock_log_probs.sum(dim=1)
        # print(f"\nAction log probs: {action_log_probs}, shape: {action_log_probs.shape}")

        # Entropies
        # print(f"Entropies: intersection: {intersection_dist.entropy()}, midblock: {midblock_dist.entropy().sum(dim=1)}")
        total_entropy = intersection_dist.entropy() + midblock_dist.entropy().sum(dim=1)
        # print(f"Total entropy: {total_entropy}, shape: {total_entropy.shape}")

        # Critic value
        state_values = self.critic(states)
        return action_log_probs, state_values, total_entropy

    def param_count(self):
        """
        Return a dict describing the parameter counts, mirroring the CNN version.
        """
        actor_params = sum(p.numel() for p in self.actor_layers.parameters()) + sum(p.numel() for p in self.actor_logits.parameters())
        critic_params = sum(p.numel() for p in self.critic_layers.parameters()) + sum(p.numel() for p in self.critic_value.parameters())

        return {
            "Actor": actor_params,
            "Critic": critic_params,
            "Total": actor_params + critic_params,
        }