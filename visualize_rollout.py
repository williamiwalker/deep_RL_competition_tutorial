#!/usr/bin/env python3
"""
Script to visualize trained agent rollouts in the Tag environment using pygame.

Usage:
    python visualize_rollout.py --model-path <path_to_model.pt> [--role prey|predator]

Example:
    python visualize_rollout.py --model-path winners_prey_prey.pt --role prey
"""

import argparse
import torch
import pygame
import numpy as np
from pettingzoo_wrapper import make_env
from tournament_loader import load_default_policies


def visualize_rollout(
    model_path,
    agent_role="prey",
    policy_module="winners_prey_policy",
    num_prey=1,
    num_predators=2,
    num_timesteps=500,
    episode_length=300
):
    """
    Visualize a trained agent playing in the Tag environment using pygame.
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine agent ID
    if agent_role == "prey":
        agent_id = "agent_0"
        print("Controlling: agent_0 (prey)")
    else:
        agent_id = "adversary_0"
        print("Controlling: adversary_0 (predator)")

    # Create environment
    env = make_env(episode_length, num_predators=num_predators, num_preys=num_prey)
    obs_dim = env.observation_space(agent_id)
    act_dim = env.action_space(agent_id)

    print(f"Environment: {num_predators} predators, {num_prey} prey")
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    # Load the policy network
    try:
        policy_module_obj = __import__(policy_module)

        # Check if it's an actor-critic model or regular policy
        if hasattr(policy_module_obj, 'ActorCriticNet') and 'actor_critic' in model_path:
            print("Loading Actor-Critic model...")
            PolicyNet = policy_module_obj.ActorCriticNet
            is_actor_critic = True
        else:
            print("Loading standard policy model...")
            PolicyNet = policy_module_obj.PolicyNet
            is_actor_critic = False

        policy = PolicyNet(obs_dim, act_dim).to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval()
        print(f"Loaded model from: {model_path}")

    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    # Create policy wrapper for actor-critic
    if is_actor_critic:
        original_policy = policy
        def policy_wrapper(obs):
            with torch.no_grad():
                action_probs, _ = original_policy(obs)
            return action_probs
        policy = policy_wrapper

    # Load default policies for other agents
    def random_policy(obs):
        return torch.rand(act_dim).to(device)

    default_policies = load_default_policies(
        env, num_prey=num_prey, num_predators=num_predators, random_policy=random_policy
    )

    # Initialize pygame
    pygame.init()
    screen_size = 800
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Tag Environment - Agent Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 50, 50)
    GREEN = (50, 255, 50)
    GRAY = (150, 150, 150)
    DARK_RED = (150, 0, 0)
    DARK_GREEN = (0, 150, 0)

    obs = env.reset()
    scores = {agent: 0.0 for agent in env.agents}
    group_names = {
        agent: "YOUR AGENT" if agent == agent_id else "CPU"
        for agent in env.agents
    }

    print("\n" + "="*60)
    print("Starting visualization (close window to exit)...")
    print("="*60 + "\n")

    running = True
    timestep = 0

    while running and timestep < num_timesteps:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get actions from policies
        actions = {}
        for agent, ob in obs.items():
            o = torch.tensor(ob, dtype=torch.float32, device=device)

            with torch.no_grad():
                if agent == agent_id:
                    action_probs = policy(o)
                else:
                    action_probs = default_policies[agent](o)

                actions[agent] = action_probs.cpu().numpy()

        # Step environment
        obs, reward, term, trunc, infos = env.step(actions)

        # Update scores
        for agent in reward:
            scores[agent] += reward[agent]

        # Reset if episode ends
        if all(term.values()) or all(trunc.values()):
            print(f"Episode ended at timestep {timestep}. Resetting...")
            obs = env.reset()

        # Render
        render_pygame(screen, env, scores, group_names, timestep, font, small_font,
                     BLACK, WHITE, RED, GREEN, GRAY, DARK_RED, DARK_GREEN, screen_size)

        pygame.display.flip()
        clock.tick(20)  # 20 FPS

        timestep += 1

    # Print final scores
    print("\n" + "="*60)
    print("FINAL SCORES:")
    print("="*60)
    for agent in sorted(scores.keys()):
        name = group_names[agent]
        score = scores[agent]
        bar = "█" * max(1, int(abs(score) / 5))
        print(f"{name:15s} ({agent}): {score:8.2f} {bar}")
    print("="*60)

    pygame.quit()
    return scores


def draw_fly(screen, pos, color, edge_color, radius):
    """Draw a fly-like shape at the given position."""
    x, y = pos

    # Draw wings (two ovals on the sides)
    wing_width = int(radius * 1.5)
    wing_height = int(radius * 0.8)

    # Left wing
    left_wing_rect = pygame.Rect(x - radius - wing_width // 2, y - wing_height // 2, wing_width, wing_height)
    pygame.draw.ellipse(screen, (200, 200, 255, 128), left_wing_rect)
    pygame.draw.ellipse(screen, edge_color, left_wing_rect, 1)

    # Right wing
    right_wing_rect = pygame.Rect(x + radius - wing_width // 2, y - wing_height // 2, wing_width, wing_height)
    pygame.draw.ellipse(screen, (200, 200, 255, 128), right_wing_rect)
    pygame.draw.ellipse(screen, edge_color, right_wing_rect, 1)

    # Draw body (elongated oval)
    body_width = int(radius * 1.2)
    body_height = int(radius * 2)
    body_rect = pygame.Rect(x - body_width // 2, y - body_height // 2, body_width, body_height)
    pygame.draw.ellipse(screen, color, body_rect)
    pygame.draw.ellipse(screen, edge_color, body_rect, 2)

    # Draw head (circle at top)
    head_radius = int(radius * 0.6)
    pygame.draw.circle(screen, color, (x, y - body_height // 2 - head_radius // 2), head_radius)
    pygame.draw.circle(screen, edge_color, (x, y - body_height // 2 - head_radius // 2), head_radius, 2)

    # Draw eyes (two small white circles)
    eye_radius = int(head_radius * 0.4)
    eye_offset = int(head_radius * 0.4)
    pygame.draw.circle(screen, (255, 255, 255), (x - eye_offset, y - body_height // 2 - head_radius // 2), eye_radius)
    pygame.draw.circle(screen, (255, 255, 255), (x + eye_offset, y - body_height // 2 - head_radius // 2), eye_radius)
    pygame.draw.circle(screen, (0, 0, 0), (x - eye_offset, y - body_height // 2 - head_radius // 2), eye_radius // 2)
    pygame.draw.circle(screen, (0, 0, 0), (x + eye_offset, y - body_height // 2 - head_radius // 2), eye_radius // 2)

    # Draw legs (six lines)
    leg_length = int(radius * 0.8)
    leg_positions = [
        (y - body_height // 4, -1),  # Front left
        (y - body_height // 4, 1),   # Front right
        (y, -1),                      # Middle left
        (y, 1),                       # Middle right
        (y + body_height // 4, -1),  # Back left
        (y + body_height // 4, 1)    # Back right
    ]

    for leg_y, side in leg_positions:
        start_x = x + side * body_width // 3
        end_x = start_x + side * leg_length
        end_y = leg_y + leg_length // 2
        pygame.draw.line(screen, edge_color, (start_x, leg_y), (end_x, end_y), 2)


def render_pygame(screen, env, scores, group_names, timestep, font, small_font,
                  BLACK, WHITE, RED, GREEN, GRAY, DARK_RED, DARK_GREEN, screen_size):
    """Render the environment using pygame."""

    screen.fill(WHITE)
    world = env.unwrapped.world

    # Coordinate transformation: environment uses [-1, 1], screen uses [0, screen_size]
    def world_to_screen(pos):
        x = int((pos[0] + 1) * screen_size / 2)
        y = int((1 - pos[1]) * screen_size / 2)  # Flip y-axis
        return (x, y)

    # Draw obstacles (landmarks)
    for landmark in world.landmarks:
        pos = landmark.state.p_pos
        screen_pos = world_to_screen(pos)
        radius = int(0.2 * screen_size / 2)
        pygame.draw.circle(screen, GRAY, screen_pos, radius)

    # Draw agents as flies
    for agent in world.agents:
        pos = agent.state.p_pos
        name = agent.name
        label = group_names[name]
        screen_pos = world_to_screen(pos)

        # Color: red for predators, green for prey
        if "adversary" in name:
            color = RED
            edge_color = DARK_RED
        else:
            color = GREEN
            edge_color = DARK_GREEN

        # Draw agent as a fly
        radius = int(0.075 * screen_size / 2)
        draw_fly(screen, screen_pos, color, edge_color, radius)

        # Draw label
        label_surface = small_font.render(label, True, WHITE)
        label_rect = label_surface.get_rect()
        label_rect.center = (screen_pos[0], screen_pos[1] - radius * 2 - 15)  # Adjusted for fly height

        # Background for label
        bg_rect = label_rect.inflate(10, 4)
        pygame.draw.rect(screen, BLACK, bg_rect, border_radius=3)
        pygame.draw.rect(screen, WHITE, bg_rect, 1, border_radius=3)

        screen.blit(label_surface, label_rect)

    # Draw scoreboard
    scoreboard_x = 10
    scoreboard_y = 10

    # Title
    title = font.render("Scores:", True, BLACK)
    screen.blit(title, (scoreboard_x, scoreboard_y))
    scoreboard_y += 30

    # Scores
    for agent in sorted(scores.keys()):
        name = group_names[agent]
        score = scores[agent]
        score_text = small_font.render(f"{name}: {score:.1f}", True, BLACK)
        screen.blit(score_text, (scoreboard_x, scoreboard_y))
        scoreboard_y += 25

    # Draw timestep counter
    timestep_text = font.render(f"Timestep: {timestep}", True, BLACK)
    screen.blit(timestep_text, (screen_size - 200, 10))

    # Draw legend
    legend_y = screen_size - 100
    legend_x = 10

    legend_title = small_font.render("Legend:", True, BLACK)
    screen.blit(legend_title, (legend_x, legend_y))
    legend_y += 25

    # Predator
    pygame.draw.circle(screen, RED, (legend_x + 10, legend_y + 8), 8)
    pygame.draw.circle(screen, DARK_RED, (legend_x + 10, legend_y + 8), 8, 2)
    pred_text = small_font.render("Predator", True, BLACK)
    screen.blit(pred_text, (legend_x + 25, legend_y))
    legend_y += 20

    # Prey
    pygame.draw.circle(screen, GREEN, (legend_x + 10, legend_y + 8), 8)
    pygame.draw.circle(screen, DARK_GREEN, (legend_x + 10, legend_y + 8), 8, 2)
    prey_text = small_font.render("Prey", True, BLACK)
    screen.blit(prey_text, (legend_x + 25, legend_y))
    legend_y += 20

    # Obstacle
    pygame.draw.circle(screen, GRAY, (legend_x + 10, legend_y + 8), 8)
    obs_text = small_font.render("Obstacle", True, BLACK)
    screen.blit(obs_text, (legend_x + 25, legend_y))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained agent rollouts in Tag environment"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model weights (.pt file)"
    )
    parser.add_argument(
        "--role",
        type=str,
        choices=["prey", "predator"],
        default="prey",
        help="Agent role: 'prey' or 'predator' (default: prey)"
    )
    parser.add_argument(
        "--policy-module",
        type=str,
        default="winners_prey_policy",
        help="Name of policy module to import (default: winners_prey_policy)"
    )
    parser.add_argument(
        "--num-prey",
        type=int,
        default=1,
        help="Number of prey agents (default: 1)"
    )
    parser.add_argument(
        "--num-predators",
        type=int,
        default=2,
        help="Number of predator agents (default: 2)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500,
        help="Total timesteps to visualize (default: 500)"
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=300,
        help="Max timesteps per episode (default: 300)"
    )

    args = parser.parse_args()

    visualize_rollout(
        model_path=args.model_path,
        agent_role=args.role,
        policy_module=args.policy_module,
        num_prey=args.num_prey,
        num_predators=args.num_predators,
        num_timesteps=args.timesteps,
        episode_length=args.episode_length
    )


if __name__ == "__main__":
    main()
