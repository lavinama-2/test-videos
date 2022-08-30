#!/bin/bash

# DQN
## baseline
### N = 1
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/DQN/baseline/test_all_agents_one_npc.sh
### N = 3
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/DQN/baseline/test_all_agents_three_npc.sh

## ego_attention
### N = 3
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/DQN/ego_attention/test_all_agents_three_npc.sh

# MADDPG
## baseline
### N = 1:
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/MADDPG/one_npc/baseline/test_all_agents_one_npc.sh
### N = 3:
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/MADDPG/three_npc/baseline/test_all_agents_three_npc.sh

## ego_attention
### N = 1
# bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/MADDPG/one_npc/ego_attention/train_all_agents_ego_attention.sh
### N = 3
bash ~/rl-agents/shell_scripts/AdvIntersectionEnv/MADDPG/three_npc/ego_attention/test_all_agents_three_npc.sh