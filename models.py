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
        print(f"\n\nAction logits: {action_logits}\n\n")
        return action_logits
    
    def critic(self, state):
        shared_features = self.shared_cnn(state)
        return self.critic_layers(shared_features)
    
    def act(self, state):
        """
        Select an action based on the current state:
        - First action: 4-class classification for traffic light
        - Second and third actions: binary choices for crosswalks
        """
        state_tensor = state.reshape(1, self.in_channels, self.action_duration, self.per_timestep_state_dim)
        action_logits = self.actor(state_tensor)
        print(f"\nAction logits: {action_logits}")
        
        # Split logits into traffic light and crosswalk decisions
        traffic_logits = action_logits[:, :4]  # First 4 logits for traffic light (4-class)
        crosswalk_logits = action_logits[:, 4:]  # Last 2 logits for crosswalks (binary)
        print(f"\nTraffic logits: {traffic_logits}")
        print(f"Crosswalk logits: {crosswalk_logits}")
        
        # Multi-class classification for traffic light
        traffic_probs = F.softmax(traffic_logits, dim=1)
        traffic_dist = Categorical(traffic_probs)
        traffic_action = traffic_dist.sample() # This predicts 0, 1, 2, or 3
        print(f"\nTraffic probabilities: {traffic_probs}")
        print(f"Traffic action: {traffic_action}")
        
        # Binary choices for crosswalks
        crosswalk_probs = torch.sigmoid(crosswalk_logits)
        crosswalk_dist = Bernoulli(crosswalk_probs)
        crosswalk_actions = crosswalk_dist.sample() # This predicts 0 or 1
        print(f"\nCrosswalk probabilities: {crosswalk_probs}")
        print(f"Crosswalk actions: {crosswalk_actions}\n")
        
        # Combine actions
        combined_action = torch.cat([traffic_action, crosswalk_actions.squeeze(0)], dim=0)
        print(f"\nCombined action: {combined_action}")
        
        # Calculate log probabilities
        log_prob = traffic_dist.log_prob(traffic_action) + crosswalk_dist.log_prob(crosswalk_actions).sum()
        print(f"\nLog probability: {log_prob}")
        
        return combined_action.long(), log_prob

    def evaluate(self, states, actions):
        """
        Evaluates a batch of states and actions.
        States are passed to actor to get action logits, using which we get the probabilities and then the distribution. similar to act function.
        Then using the sampled actions, we get the log probabilities and the entropy. 
        Finally, we pass the states to critic to get the state values. (used to compute the value function component of the PPO loss)
        The entropy is used as a regularization term to encourage exploration.
        """
        action_logits = self.actor(states)
        
        # Split logits and actions
        traffic_logits = action_logits[:, 0:2]
        crosswalk_logits = action_logits[:, 2:]
        traffic_actions = actions[:, 0:2].argmax(dim=1)  # Convert one-hot back to index
        crosswalk_actions = actions[:, 2:]
        
        # Evaluate traffic direction actions
        traffic_probs = F.softmax(traffic_logits, dim=1)
        #TODO:Visualize this?
        print(f"\nTraffic probabilities: {traffic_probs}\n")

        traffic_dist = Categorical(traffic_probs)
        traffic_log_probs = traffic_dist.log_prob(traffic_actions)
        
        # Evaluate crosswalk actions
        crosswalk_probs = torch.sigmoid(crosswalk_logits)
        crosswalk_dist = Bernoulli(crosswalk_probs)
        crosswalk_log_probs = crosswalk_dist.log_prob(crosswalk_actions)
        
        # Combine log probabilities
        action_log_probs = traffic_log_probs + crosswalk_log_probs.sum(dim=1)
        
        # Calculate entropy 
        dist_entropy = traffic_dist.entropy() + crosswalk_dist.entropy().sum(dim=1)
        
        state_values = self.critic(states)
        
        return action_log_probs, state_values, dist_entropy

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
