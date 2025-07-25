import torch

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


def train_hrl(higher_agent, lower_agents, env, num_episodes=1000):
    for episode in range(num_episodes):
        state_high = env.get_high_state()
        action_high, log_prob = select_ppo_action(higher_agent.model, state_high)
        selected_vms = env.apply_vm_selection(action_high)

        lower_buffers = [ReplayBuffer() for _ in selected_vms]
        lower_policies = lower_agents

        memory_high = []

        for t in range(env.period_length):
            for i, vm_id in enumerate(selected_vms):
                state_low = env.get_low_state(vm_id)
                action_low = lower_policies[i].select_action(state_low)
                reward, next_state = env.apply_vm_scaling(vm_id, action_low)
                lower_buffers[i].push(state_low, action_low, reward, next_state)

        for i, agent in enumerate(lower_policies):
            agent.update(lower_buffers[i])

        # Store memory for PPO
        total_reward = env.compute_utility()
        memory_high.append((state_high, action_high, log_prob, torch.tensor(total_reward)))

        # PPO update
        higher_agent.update(memory_high)

        print(f"Episode {episode}: Utility = {total_reward:.3f}")
