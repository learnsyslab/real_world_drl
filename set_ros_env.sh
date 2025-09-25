#/usr/bin/env sh

ROS_ENV_FILE=".ros_env.sh"

if [ -f "$ROS_ENV_FILE" ]; then
    echo "$ROS_ENV_FILE already exists. Sourcing it..."
    . "$ROS_ENV_FILE"
else
    echo "$ROS_ENV_FILE not found. Using degust environment variables..."
    export ROS_DOMAIN_ID=102
    export ROS_LOCALHOST_ONLY=0
    export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
fi
export CRISP_CONFIG_PATH=/home/lukahofer/repos/crisp_config

ros2 daemon stop
ros2 daemon start
