import random
from collections import deque

# buffer for training
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class PPOAgent:
    def __init__(self, model, lr=1e-4, gamma=0.99, eps_clip=0.2):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def update(self, memory):
        states, actions, log_probs_old, returns = zip(*memory)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        log_probs_old = torch.tensor(log_probs_old)
        returns = torch.tensor(returns)

        policy, values = self.model(states)
        dist = torch.distributions.Categorical(policy)
        log_probs = dist.log_prob(actions)

        ratios = torch.exp(log_probs - log_probs_old)
        advantages = returns - values.detach().squeeze()

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        loss = -torch.min(surr1, surr2).mean() + F.mse_loss(values.squeeze(), returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQNAgent:
    def __init__(self, q_net, input_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=0.1):
        self.q_net = q_net
        self.target_net = DQN(input_dim, 128, action_dim)
        self.target_net.load_state_dict(q_net.state_dict())
        self.optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.q_net(state)
            return q_values.argmax().item()

    def update(self, buffer, batch_size=32):
        if len(buffer) < batch_size:
            return

        samples = buffer.sample(batch_size)
        states, actions, rewards, next_states = zip(*samples)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
