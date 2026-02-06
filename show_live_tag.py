import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import numpy as np

from tournament_loader import load_all_policies, load_all_group_names
from pettingzoo_wrapper import AdversaryObsRewardWrapper, make_env

from gymnasium.wrappers import RecordVideo





TIMESTEPS_PER_EPISODE = 300 # max timesteps per episode


prey_group_names = ['winners']
predator_group_names = ['trained_beast_2','default']


# prey_groups = list(load_all_group_names('prey', prey_group_names))
# predator_groups = list(load_all_group_names('predator', predator_group_names))




# torch.set_grad_enabled(False)

def get_agent_positions(env):
    world = env.unwrapped.world
    positions = {}
    for agent in world.agents:
        positions[agent.name] = agent.state.p_pos.copy()
    return positions


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
# import imageio

# import numpy as np
# import imageio
# from PIL import Image, ImageDraw, ImageFont
# import torch

# def render_tag(num_steps, env, scores, group_names, policies, save_path="tag_match.mp4"):
#     frames = []

#     obs = env.reset()
#     scores = {agent: 0.0 for agent in env.agents}

#     for t in range(num_steps):
#         actions = {}
#         for agent, ob in obs.items():
#             with torch.no_grad():
#                 act = policies[agent](torch.tensor(ob, dtype=torch.float32)).numpy()
#                 actions[agent] = act

#         obs, rewards, terms, truncs, infos = env.step(actions)

#         for agent in rewards:
#             scores[agent] += rewards[agent]

#         # === GET FRAME DIRECTLY FROM ENV ===
#         frame = env.render()#mode="rgb_array")
#         img = Image.fromarray(frame)
#         draw = ImageDraw.Draw(img)

#         # === SCOREBOARD ===
#         y = 10
#         for agent, score in scores.items():
#             label = group_names.get(agent, agent)
#             draw.text((10, y), f"{label}: {score:.2f}", fill=(255,255,255))
#             y += 15

#         frames.append(np.array(img))

#         if all(terms.values()) or all(truncs.values()):
#             obs = env.reset()

#     imageio.mimsave(save_path, frames, fps=20)
#     print(f"Saved video to {save_path}")



# def render_with_labels(env, scores, group_labels, ax):
#     world = env.unwrapped.world

#     ax.clear()
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_aspect("equal")
#     ax.axis("off")

#     # Draw landmarks
#     for landmark in world.landmarks:
#         pos = landmark.state.p_pos
#         ax.scatter(pos[0], pos[1], c="gray", s=2300)

#     # Draw agents
#     for agent in world.agents:
#         pos = agent.state.p_pos
#         name = agent.name
#         label = group_labels[name]
#         color = "red" if "adversary" in name else "green"

#         ax.scatter(pos[0], pos[1], c=color, s=200)
#         ax.text(
#             pos[0],
#             pos[1] + 0.05,
#             label,
#             ha="center",
#             va="bottom",
#             fontsize=9,
#             color="white",
#             bbox=dict(facecolor="black", alpha=0.6, pad=1),
#         )

#     # Scoreboard
#     score_text = "\n".join(
#         f"{group_labels[a]}: {scores[a]:.2f}" for a in scores
#     )
#     ax.text(
#         -0.95, 0.95,
#         score_text,
#         ha="left",
#         va="top",
#         fontsize=10,
#         bbox=dict(facecolor="black", alpha=0.7),
#         color="white"
#     )


# def render_tag(num_timesteps, env, scores, group_names, policies, save_path="tag_match.mp4"):
#     fig, ax = plt.subplots(figsize=(6, 6))
#     frames = []

#     obs = env.reset()
#     scores = {agent: 0.0 for agent in env.agents}

#     for t in range(num_timesteps):
#         actions = {}
#         for agent, ob in obs.items():
#             with torch.no_grad():
#                 actions[agent] = policies[agent](
#                     torch.tensor(ob, dtype=torch.float32)
#                 ).numpy()

#         obs, rewards, terms, truncs, infos = env.step(actions)

#         for agent in rewards:
#             scores[agent] += rewards[agent]

#         render_with_labels(env, scores, group_names, ax)

#         # 🔑 FORCE DRAW
#         fig.canvas.draw()
#         fig.canvas.flush_events()

#         # 🔑 CAPTURE UPDATED FRAME
#         # frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         # frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         # frames.append(frame)
# #         fig.canvas.draw()
#         w, h = fig.canvas.get_width_height()
#         buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#         buf = buf.reshape((h, w, 4))      # RGBA
#         image = buf[:, :, :3]             # drop alpha channel
#         frames.append(image)

#         if all(terms.values()) or all(truncs.values()):
#             obs = env.reset()

#     plt.close(fig)

#     imageio.mimsave(save_path, frames, fps=20)
#     print(f"Saved video to {save_path}")


# def draw_frame(ax, env, scores, group_labels):
#     world = env.unwrapped.world

#     ax.clear()
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_aspect("equal")
#     ax.axis("off")

#     # Draw landmarks
#     for landmark in world.landmarks:
#         pos = landmark.state.p_pos
#         ax.scatter(pos[0], pos[1], c="gray", s=2300)

#     # Draw agents
#     for agent in world.agents:
#         pos = agent.state.p_pos
#         name = agent.name
#         label = group_labels[name]

#         color = "red" if "adversary" in name else "green"

#         ax.scatter(pos[0], pos[1], c=color, s=200)
#         ax.text(
#             pos[0],
#             pos[1] + 0.05,
#             label,
#             ha="center",
#             va="bottom",
#             fontsize=9,
#             color="white",
#             bbox=dict(facecolor="black", alpha=0.6, pad=1),
#         )

#     # Scoreboard
#     score_text = "\n".join(
#         f"{group_labels[a]}: {scores[a]:.2f}"
#         for a in scores
#     )

#     ax.text(
#         -0.95, 0.95,
#         score_text,
#         ha="left",
#         va="top",
#         fontsize=10,
#         bbox=dict(facecolor="black", alpha=0.7),
#         color="white"
#     )


# def render_tag(num_timesteps, env, scores, group_names, policies, save_path="tag_match.mp4"):

#     fig, ax = plt.subplots(figsize=(6, 6))

#     obs = env.reset()
#     scores = {agent: 0.0 for agent in env.agents}

#     frames = []

#     for t in range(num_timesteps):
#         actions = {}
#         for agent, ob in obs.items():
#             with torch.no_grad():
#                 actions[agent] = policies[agent](
#                     torch.tensor(ob, dtype=torch.float32)
#                 ).numpy()

#         obs, reward, term, trunc, infos = env.step(actions)

#         if all(term.values()) or all(trunc.values()):
#             obs = env.reset()

#         for agent in reward:
#             scores[agent] += reward[agent]

#         draw_frame(ax, env, scores, group_names)

#         # 🔑 FORCE DRAW
#         fig.canvas.draw()
#         fig.canvas.flush_events()

#         # capture frame
#         fig.canvas.draw()
#         w, h = fig.canvas.get_width_height()
#         buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#         buf = buf.reshape((h, w, 4))      # RGBA
#         image = buf[:, :, :3]             # drop alpha channel
#         frames.append(image)

#         # fig.canvas.draw()
#         # image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
#         # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         # frames.append(image)

#     plt.close(fig)

#     # Save video
#     import imageio
#     imageio.mimsave(save_path, frames, fps=20)

#     print("Final scores:")
#     for agent, score in scores.items():
#         print(f"{group_names[agent]} ({agent}): {score:.2f}")

#     print(f"Saved video to: {save_path}")
#     return scores


def render_with_labels(env, scores, group_labels):
    world = env.unwrapped.world

    plt.clf()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect("equal")
    plt.axis("off")

    # Draw landmarks
    for landmark in world.landmarks:
        pos = landmark.state.p_pos
        plt.scatter(pos[0], pos[1], c="gray", s=2300)#, marker="s")

    # Draw agents
    for agent in world.agents:
        pos = agent.state.p_pos
        name = agent.name
        label = group_labels[name] #.get(name, name)

        color = "red" if "adversary" in name else "green"

        plt.scatter(pos[0], pos[1], c=color, s=200)
        plt.text(
            pos[0],
            pos[1] + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    # Scoreboard
    score_text = "\n".join(
        f"{group_labels.get(a, a)}: {scores[a]:.2f}"
        for a in scores
    )

    plt.text(
        -0.95, 0.95,
        score_text,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.7),
        color="white"
    )

    plt.pause(0.05)


def render_tag(num_timesteps, env, scores, group_names, policies):

    plt.figure(figsize=(6, 6))

    obs = env.reset()
    scores = {agent: 0.0 for agent in env.agents}

    for t in range(num_timesteps):
        actions = {}
        for agent, ob in obs.items():
            with torch.no_grad():
                actions[agent] = policies[agent](
                    torch.tensor(ob, dtype=torch.float32))

        obs, reward, term, trunc, infos = env.step(actions)

        if all(term.values()) or all(trunc.values()):
            obs = env.reset()

        for agent in reward:
            scores[agent] += reward[agent]

        render_with_labels(env, scores, group_names)
    
    plt.pause(1.0)    

    plt.close()

    print('Final scores:')
    for agent, score in scores.items():
        print(f"{group_names[agent]} ({agent}): {score:.2f}")

    return scores


def plot_learning_curve(episode_rewards):
    plt.figure()
    plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'))
    plt.xlabel('Episode')
    plt.ylabel('Return (smoothed)')
    plt.title('Training Curve')
    plt.show()


if __name__ == "__main__":
    plt.figure(figsize=(6, 6))
        
    #################################################
    # Define environment
    #################################################

    env = make_env(TIMESTEPS_PER_EPISODE, num_predators=len(predator_group_names), num_preys=len(prey_group_names))
    obs = env.reset()
    print('Environment created with', len(env.agents), 'agents.')


    ###############################################
    # Initialize policies for each agent
    ###############################################
    policies, group_names = load_all_policies(env, prey_group_names, predator_group_names)
    print('group_names:', group_names)


    render_tag(num_timesteps=500, env=env, scores={agent: 0.0 for agent in env.agents}, group_names=group_names, policies=policies)



