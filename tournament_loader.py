import importlib.util
import torch
from pettingzoo.mpe import simple_tag_v3
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


def number_of_submissions(role):
    files = glob.glob("submissions/*_"+role+"_policy.py")
    return len(files)

def load_all_group_names(role, group_names):
    files = []
    for name in group_names:
        path = f"submissions/{name}_{role}_policy.py"
        if os.path.exists(path):
            files.append(path)
        else:
            print(f"Warning: no policy file found for group '{name}' with role '{role}'")
    return files

def load_student_policy(pyfile, weightfile, agent_name, env):
    spec = importlib.util.spec_from_file_location("student_policy", pyfile)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    obs_dim = env.observation_space(agent_name)#.shape[0]
    act_dim = env.action_space(agent_name)#.shape[0]
    print('Loading policy for', agent_name, 'obs_dim:', obs_dim, 'act_dim:', act_dim)

    net = module.PolicyNet(obs_dim, act_dim)
    net.load_state_dict(torch.load(weightfile))
    net.eval()
    return net



# def load_all_policies(env, prey_groups, predator_groups):
#     policies = {}
#     group_names = {}

#     # load all prey policies
#     files = sorted(glob.glob("submissions/*_prey_policy.py"))
#     for i, pyfile in enumerate(files):
#         group = os.path.basename(pyfile).replace("_prey_policy.py", "")
#         weightfile = f"submissions/{group}_prey.pt"

#         agent_name = f"agent_{i}"
#         group_names[agent_name] = group
#         print('group:', group, 'agent_name:', agent_name)
#         policies[agent_name] = load_student_policy(pyfile, weightfile, agent_name, env)
#         print(f"Loaded {group} as {agent_name}")

#     # load all predator policies
#     files = sorted(glob.glob("submissions/*_predator_policy.py"))
#     for i, pyfile in enumerate(files):
#         group = os.path.basename(pyfile).replace("_predator_policy.py", "")
#         weightfile = f"submissions/{group}_predator.pt"

#         agent_name = f"adversary_{i}"
#         group_names[agent_name] = group
#         print('group:', group, 'agent_name:', agent_name)
#         policies[agent_name] = load_student_policy(pyfile, weightfile, agent_name, env)
#         print(f"Loaded {group} as {agent_name}")

#     return policies, group_names


def load_all_policies(env, prey_groups, predator_groups):
    policies = {}
    group_names = {}

    # load all prey policies
    # for i, pyfile in enumerate(files):
    for i, name in enumerate(prey_groups):
        try:
            pyfile = f"submissions/{name}_policy.py"
            group = os.path.basename(pyfile).replace("_policy.py", "")
            weightfile = f"submissions/{group}_prey.pt"
        except Exception as e:
            print(f"Error loading prey policy for group '{name}': {e}")
            return

        agent_name = f"agent_{i}"
        group_names[agent_name] = group
        print('group:', group, 'agent_name:', agent_name)
        policies[agent_name] = load_student_policy(pyfile, weightfile, agent_name, env)
        print(f"Loaded {group} as {agent_name}")

    # load all predator policies
    # files = sorted(glob.glob("submissions/*_predator_policy.py"))
    # for i, pyfile in enumerate(files):
    for i, name in enumerate(predator_groups):
        try:
            pyfile = f"submissions/{name}_policy.py"
            group = os.path.basename(pyfile).replace("_policy.py", "")
            weightfile = f"submissions/{group}_predator.pt"
        except Exception as e:
            print(f"Error loading predator policy for group '{name}': {e}")
            return

        agent_name = f"adversary_{i}"
        group_names[agent_name] = group
        print('group:', group, 'agent_name:', agent_name)
        policies[agent_name] = load_student_policy(pyfile, weightfile, agent_name, env)
        print(f"Loaded {group} as {agent_name}")

    return policies, group_names


def load_default_policies(env, num_prey, num_predators, random_policy):
    policies = {}
    group_names = {}

    # load all prey policies
    files = sorted(glob.glob("default_policy.py"))
    print('Default prey policy files found:', files)
    for i in range(num_prey):
        agent_name = f"agent_{i}"
        weightfile = f"default_prey.pt"
        group_names[agent_name] = 'default'
        if len(sorted(glob.glob("default_prey.pt"))) == 0:
            policies[agent_name] = random_policy
        else:
            pyfile = files[0]
            policies[agent_name] = load_student_policy(pyfile, weightfile, agent_name, env)

    # load all predator policies
    files = sorted(glob.glob("default_policy.py"))
    for i in range(num_predators):
        agent_name = f"adversary_{i}"
        weightfile = f"default_predator.pt"
        group_names[agent_name] = 'default'
        if len(sorted(glob.glob("default_predator.pt"))) == 0:
            policies[agent_name] = random_policy
        else:
            pyfile = files[0]
            policies[agent_name] = load_student_policy(pyfile, weightfile, agent_name, env)

    return policies



