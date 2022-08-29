#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/mario/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else 
    if [ -f "/Users/mario/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/mario/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/mario/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
export PATH=/Users/mario/opt/anaconda3/bin:$PATH

conda activate highway_env
# source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/TensorRT-6.0.1.8/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/11.4.120-cudnn8.2.4/targets/x86_64-linux/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/11.4.120-cudnn8.2.4/x86_64-linux-gnu

TERM=vt100 # or TERM=xterm

# For Stable Baselines 3 training
# cd /vol/bitbucket/tg4018/MEng/rl-baselines3-zoo

# python train.py --algo ddpg --env parking-v0 --tensorboard-log ./log/tensorboard/parking/ddpg
# python train.py --algo matd3 --env parking-ma-v0 -n 10000000 --vec-env multiagent --save-freq 25000 --eval-freq 25000 -tb ./log/tensorboard/ma_parking/matd3_1_experimental


# For rl-agents training
cd rl-agents/scripts/

# background
python3 experiments.py evaluate configs/AdvIntersectionEnv/one_npc/env_multi_agent_dest_background.json \
                                configs/AdvIntersectionEnv/agents/MADDPGAgent/one_npc/baseline.json \
                                --test --recover-from=out/MultiAgentAdvIntersectionEnv/MADDPGAgent/one_npc/baseline/one_npc_background_train_baseline_20220827-123347_611434/checkpoint.tar \
                                --episodes=1000 --name-from-config --no-display 

# good_agent
python3 experiments.py evaluate configs/AdvIntersectionEnv/one_npc/env_multi_agent_dest_good_agent.json \
                               configs/AdvIntersectionEnv/agents/MADDPGAgent/one_npc/baseline.json \
                               --test --recover-from=out/MultiAgentAdvIntersectionEnv/MADDPGAgent/one_npc/baseline/one_npc_good_agent_train_baseline_20220827-151845_617419/checkpoint-best.tar \
                               --episodes=1000 --name-from-config --no-display 

# zero_sum
python3 experiments.py evaluate configs/AdvIntersectionEnv/one_npc/env_multi_agent_dest_zero_sum.json \
                               configs/AdvIntersectionEnv/agents/MADDPGAgent/one_npc/baseline.json \
				               --test --recover-from=out/MultiAgentAdvIntersectionEnv/MADDPGAgent/one_npc/baseline/one_npc_zero_sum_train_baseline_20220827-171240_620943/checkpoint-best.tar \
                                --episodes=1000 --name-from-config --no-display

# failmaker
python3 experiments.py evaluate configs/AdvIntersectionEnv/one_npc/env_multi_agent_dest_failmaker.json \
                               configs/AdvIntersectionEnv/agents/MADDPGAgent/one_npc/baseline.json \
				                --test --recover-from=out/MultiAgentAdvIntersectionEnv/MADDPGAgent/one_npc/baseline/one_npc_failmaker_train_baseline_20220827-190825_624432/checkpoint-best.tar \
                                --episodes=1000 --name-from-config --no-display

# rule_break
python3 experiments.py evaluate configs/AdvIntersectionEnv/one_npc/env_multi_agent_dest_rule_break.json \
                               configs/AdvIntersectionEnv/agents/MADDPGAgent/one_npc/baseline.json \
                                --test --recover-from=out/MultiAgentAdvIntersectionEnv/MADDPGAgent/one_npc/baseline/one_npc_rule_break_train_baseline_20220827-203808_627204/checkpoint-best.tar \
                                --episodes=1000 --name-from-config --no-display 
