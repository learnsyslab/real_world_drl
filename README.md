**CRISP_DRL** is a Deep Reinforcement Learning (DRL) setup for robot manipulation in the real world. It natively integrates well with the low-level [CRISP Controllers](https://github.com/utiasDSL/crisp_controllers) and [CRISP_GYM](https://github.com/utiasDSL/crisp_gym).



## Features
- 🤖 **Real-World Learning** with python Deep Reinforcement Learning implementations.
- ⚙️ **Parallelized** training through Actor-Learner Multiprocessing architecture.  
- 📡 **ROS2 Communication** is handled through the integration with the **CRISP** software stack.
- 🕹️ **Interactive Training** through the Training Command Line Interface. It allows to set sparse reward signals.


______

## Setup

We use **pixi** as a package manager to handle both the python and ROS depencies. Follow the [instructions](https://pixi.sh/dev/installation/) to install **pixi**.

First, we need to clone both **crisp_gym** and **crisp_drl**. 

```bash
mkdir repos
cd repos
git clone git@github.com:utiasDSL/crisp_gym.git
git clone git@github.com:utiasDSL/crisp_drl.git
cd crisp_drl
```

We now want to configure our **crisp_gym** environments. We can create a `config/` folder in the root folder of our repo and add some yaml configuration files there. 
> Note: These are crisp environment configurations, not configurations for our algorithms.

```bash
mkdir config
touch config/gripper.yaml
mkdir config/envs
touch config/envs/rl_setup.yaml
```

The following shows, how your config files could look like.
<details>
<summary>Click here to show sample environment config</summary>

```bash
control_frequency: 10.0
gripper_enabled: true
gripper_continuous_control: true
max_episode_steps: 100

cartesian_control_param_config: null
joint_control_param_config: null

robot_config:
  robot_type: "franka"
  time_to_home: 3.0
  publish_frequency: 50.0

gripper_config:
  from_yaml: "gripper.yaml"

camera_configs:
  - camera_name: "primary"
    camera_frame: "primary_link"
    resolution: [256, 256]
    camera_color_image_topic: "<primary/camera/image/topic>"
    camera_color_info_topic: "<camera/info/topic>"
  - camera_name: "wrist"
    camera_frame: "wrist_link"
    resolution: [256, 256]
    camera_color_image_topic: "<wrist/camera/image/topic>"
    camera_color_info_topic: "<wrist/camera/info/topic>"
```
</details>


<details>
<summary>Click here to show sample gripper config</summary>

```bash
joint_state_topic: "gripper/gripper_state_broadcaster/joint_states"
command_topic: "gripper/gripper_position_controller/commands"
max_value: -0.07976700087890626
min_value: -1.2977477450683594
```
</details>

---
Next, create a `set_ros_env.sh` file in the root folder of your repository with the following content. Read the **CRISP** documentation to find out how to set this up correctly. The two sections [crisp_gym](https://utiasdsl.github.io/crisp_controllers/getting_started/#4-getting-started-with-crisp_gym) and [Multi-machine setup](https://utiasdsl.github.io/crisp_controllers/misc/multi_machine_setup/) contain the relevant information.
```bash
#/usr/bin/env sh

ROS_ENV_FILE=".ros_env.sh"

if [ -f "$ROS_ENV_FILE" ]; then
    echo "$ROS_ENV_FILE already exists. Sourcing it..."
    . "$ROS_ENV_FILE"
else
    echo "$ROS_ENV_FILE not found. Using degust environment variables..."
    export ROS_DOMAIN_ID=<YourRobotId>
    export ROS_LOCALHOST_ONLY=0
    export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
fi
export CRISP_CONFIG_PATH=</path/to/your/config>

ros2 daemon stop
ros2 daemon start
```


Finally, we will install all the dependencies and activate the pixi environment.
```bash
pixi install
pixi shell -e humble-lerobot
```

## Training an agent
For the RLPD algorithm, we first need to create a Replay Buffer containing prerecorded offline expert demonstrations of the desired task. We can use the **crisp_gym** record script to generate an expert dataset. Later we can convert the generated LeRobot Dataset to our required ReplayBuffer format.
1.  Go to the **crisp_gym** repository and execute the record script. Then just follow the instructions on the command line to record the dataset.
```bash
cd ../crisp_gym
python scripts/record_lerobot_format_leader_follower.py
```
2. We now return back to **crisp_drl**. The script `load_and_save_rb_from_dataset.py` can convert a dataset in the **LeRobot Dataset** format to our ReplayBuffer format. 
```bash
cd ../crisp_drl
python crisp_drl/agents/rlpd/load_and_save_rb_from_dataset.py <lerobot_dataset_id> .
```

3. Once we created the Replay Buffer pickle file, we can pass it to the training script and start training. Optionally, we can run training in the interactive cli mode by adding the  `--cli_training` argument.
```bash
python crisp_drl/agents/rlpd/run_rlpd_agent.py <lerobot_dataset_id>.pkl --cli_training
```

<details>
<summary>Other arguments</summary>

- **--run_name**: Specify the run name of the training run to be used for checkpoints and TensorBoard.
- **--eval**: Only launches the actor node and performs inference on the policy.
- **--load_policy**: This should be used in combination with **--eval**. It loads the policy from a specific checkpoint.
- **--resume_training**: This loads all training parameters saved in the checkpoints. This should be used if the training was temporarily interrupted and you would like to continue training. 
</details>