#/usr/bin/env sh

ROS_ENV_FILE=".ros_env.sh"

if [ -f "$ROS_ENV_FILE" ]; then
    echo "$ROS_ENV_FILE already exists. Sourcing it..."
    . "$ROS_ENV_FILE"
else
    echo "$ROS_ENV_FILE not found. Using default environment variables..."
    export ROS_DOMAIN_ID=100
    export ROS_LOCALHOST_ONLY=0
    # export FASTDDS_BUILTIN_TRANSPORTS=UDPv4


    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file:///home/danielsanjosepro/repos/crisp_gym/scripts/cyclone_config.xml
    export NETWORK_INTERFACE=enp4s0

fi
export CRISP_CONFIG_PATH=/home/danielsanjosepro/repos/crisp_configs/config

ros2 daemon stop
ros2 daemon start
