#!/bin/bash
# $1: algorithm MADDPG, DQN
# $2: number_agents one_npc, three_npc
# $3: type_agent baseline, ego_attention
# $4: path_model
# $5: name_trained_agent

$background = "background"

if [ $5 == $background ]; then
    python3 experiments.py evaluate configs/AdvIntersectionEnv/$2/env_multi_agent_dest_background.json \
                                configs/AdvIntersectionEnv/agents/$1/$2/$3.json \
                                --test --recover-from=out/MultiAgentAdvIntersectionEnv/$1/$2/$3/$4/checkpoint-best.tar \
                                --episodes=1000 --name-from-config --no-display 
fi
