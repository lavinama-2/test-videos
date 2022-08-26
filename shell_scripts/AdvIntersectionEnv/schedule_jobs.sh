#!/bin/bash

# DQN
## baseline
### N = 1
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/DQN/baseline/train_all_agents_one_npc.sh
### N = 3
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/DQN/baseline/train_all_agents_three_npc.sh

## ego_attention
### N = 3
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/DQN/ego_attention/train_all_agents_three_npc.sh

# MADDPG
## baseline
### N = 1: TODO
### N = 3: TODO

## ego_attention
### N = 1
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/MADDPG/ego_attention/train_all_agents_three_npc.sh
### N = 3
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/MADDPG/ego_attention/train_all_agents_three_npc.sh