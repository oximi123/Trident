import torch

from env.serverless_env import HrlCloudEnv
from provisioner.agent import ReplayBuffer

# train logic
def select_ppo_action(model, state):
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        probs, _ = model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return action.item(), log_prob.item()


def train_hrl(higher_agent, lower_agents, env : HrlCloudEnv, num_episodes=1000):
    for episode in range(num_episodes):
        for T in range(env.horizon_length):
            state_high = env.get_high_level_state()
            action_high, log_prob = select_ppo_action(higher_agent.model, state_high)
            selected_vms = env.high_level_step(action_high)

            lower_buffers = [ReplayBuffer() for _ in selected_vms]
            lower_policies = lower_agents

            memory_high = []

            for t in range(env.period_length):
                states_low = env.get_low_level_state()
                actions_low = []
                for i, vm_id in enumerate(selected_vms):
                    state_low = states_low[i]
                    action_low = lower_policies[i].select_action(state_low)
                    actions_low.append(action_low)
                low_reward, next_state = env.low_level_step(actions_low)
                for i, vm_id in enumerate(selected_vms):
                    lower_buffers[i].push(states_low[i], actions_low[i], low_reward, next_state)

            for i, agent in enumerate(lower_policies):
                agent.update(lower_buffers[i])

            # Store memory for PPO
            high_reward = env.get_high_level_reward()
            memory_high.append((state_high, action_high, log_prob, torch.tensor(high_reward)))
            higher_agent.update(memory_high)
