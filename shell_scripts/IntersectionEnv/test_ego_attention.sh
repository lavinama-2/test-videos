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
python3 experiments.py evaluate configs/IntersectionEnv/env_multi_agent_dest.json \
                               configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json \
                               --test --episodes=5 --name-from-config \
                               --recover-from="out/MultiAgentIntersectionEnv/DQNAgent/ego_attention_2h_20220722-162102_1015501/checkpoint-best.tar"
#                               --train --episodes=25000 --no-display \
#                               --recover-from=../../out_agents/DQN_env_multi_agent_hash_dest_ego_attention_2h/checkpoint-best.tar