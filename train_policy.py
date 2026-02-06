import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo_wrapper import AdversaryObsRewardWrapper, make_env
import matplotlib.pyplot as plt
from show_live_tag import plot_learning_curve, render_tag
from tournament_loader import load_default_policies


# Load student policy network
from group_A_policy import PolicyNet  # <<< students change this

# ------------------------
# Configuration
# ------------------------
GROUP_NAME = "group_C_predator"   # <<< students change this
AGENT_ROLE = "predator"  # <<< students change this: "predator" or "prey"
NUM_PREY = 1 # number of prey groups
NUM_PREDATORS = 2 # number of predator groups
NUM_EPOCHS = 10 # <<< students change this: number of training episodes
SAVE_PATH = f"{GROUP_NAME}_{AGENT_ROLE}.pt"
TIMESTEPS_PER_EPISODE = 300 # max timesteps per episode




if AGENT_ROLE == "prey":
    agent_id = "agent_0"  # student controls this agent
else:
    agent_id = "adversary_0"  # student controls this agent


# ------------------------
# Create environment
# ------------------------
env = make_env(TIMESTEPS_PER_EPISODE, num_predators=NUM_PREDATORS, num_preys=NUM_PREY)

obs_dim = env.observation_space(agent_id)
act_dim = env.action_space(agent_id)


# ------------------------
# Initialize policy network and optimizer
# ------------------------
policy = PolicyNet(obs_dim, act_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

# ------------------------
# Random policy or default policy for other agents
# ------------------------
def random_policy(obs):
    return torch.rand(act_dim) 

default_policies = load_default_policies(env, num_prey=NUM_PREY, num_predators=NUM_PREDATORS, random_policy=random_policy)


# ------------------------
# Training loop (REINFORCE)
# ------------------------
episode_rewards = []
for episode in range(NUM_EPOCHS):
    obs = env.reset()
    log_probs = []
    rewards = []
    steps = 0

    done = False
    while not done:
        actions = {}

        for agent, ob in obs.items():
            o = torch.tensor(ob, dtype=torch.float32)
            if agent == agent_id:

                action = policy(o)
                dist = torch.distributions.Bernoulli(action)
                sampled = dist.sample()
                log_prob = dist.log_prob(sampled).sum()

                actions[agent] = sampled.detach().numpy()
                log_probs.append(log_prob)
            else:
                actions[agent] = default_policies[agent](o).detach().numpy()

        obs, reward, term, trunc, info = env.step(actions)
        steps += 1
        # print('Step:', steps)
        # print('Rewards:', reward, 'observations:', obs, 'info', info, 'actions:', actions, 'done:', term, trunc)
        if not trunc[agent_id]:
            rewards.append(reward[agent_id])
        done = term[agent_id] or trunc[agent_id]

    # Return
    R = sum(rewards)
    episode_rewards.append(R)

    loss = -torch.tensor(R, dtype=torch.float32).detach() * torch.stack(log_probs).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, return = {R:.2f}")

# ------------------------
# Save model
# ------------------------
torch.save(policy.state_dict(), SAVE_PATH)
print("Saved:", SAVE_PATH)

# ------------------------
# Plot training curve
# ------------------------
plot_learning_curve(episode_rewards)


# ------------------------
# See example rollouts
# ------------------------
obs = env.reset()
scores = {agent: 0.0 for agent in env.agents}
group_names = {agent: "yours" if agent == agent_id else "CPU" for i, agent in enumerate(env.agents)}
all_policies = {agent: policy if agent == agent_id else default_policies[agent] for agent in env.agents}

render_tag(500, env, scores=scores, group_names=group_names, policies=all_policies)

