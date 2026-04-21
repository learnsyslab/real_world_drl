#/usr/bin/env sh

# ROS_ENV_FILE=".ros_env.sh"

# if [ -f "$ROS_ENV_FILE" ]; then
#     echo "$ROS_ENV_FILE already exists. Sourcing it..."
#     . "$ROS_ENV_FILE"
# else
#     echo "$ROS_ENV_FILE not found. Using default environment variables..."
#     export ROS_DOMAIN_ID=101
#     export ROS_LOCALHOST_ONLY=0
#     # export FASTDDS_BUILTIN_TRANSPORTS=UDPv4


#     export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
#     export CYCLONEDDS_URI=file:///home/linusschwarz/crisp_gym/scripts/cyclone_config.xml
#     export NETWORK_INTERFACE=enp4s0

# fi
# export CRISP_CONFIG_PATH=/home/linusschwarz/crisp_configs

export ROS_DOMAIN_ID=101
export GIT_LFS_SKIP_SMUDGE=1
export CRISP_CONFIG_PATH=/home/gabor/repos/crisp_configs_realWorldDRL
export NETWORK_INTERFACE=enp128s31f6
export ROS_NETWORK_INTERFACE=enp128s31f6
export ROS_STATIC_PEERS='127.0.0.1;10.157.163.155'
# export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/gabor/repos/crisp_gym/scripts/cyclone_config.xml
export TORCH_HOME=/home/gabor/.cache/torch
