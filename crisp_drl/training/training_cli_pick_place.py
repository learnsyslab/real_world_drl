from argparse import ArgumentParser
import pickle
import sys
import time
from typing import Any
import gymnasium
import numpy as np
from pynput import keyboard
import os
import threading
import subprocess
import torch.multiprocessing as mp
import crisp_gym # type: ignore
from crisp_gym.manipulator_env import ManipulatorCartesianEnv, make_env_config # type: ignore
from crisp_gym.util.rl_utils import load_actions_safe, custom_reset # type: ignore

# Use current user and home directory instead of hardcoded usernames
USER_AT_FRANKA = f"{os.getenv('USER', 'gabor')}@franka"
USER_HOME = os.path.expanduser("~")


def controller_container_watcher(out_queue):
    # until first line of current.log split at " " changes
    while True:
        t = threading.Timer(1.0, lambda: None)
        t.start()
        error_string = subprocess.run(["ssh", USER_AT_FRANKA, fr"grep -P 'cartesian_reflex|communication_constraints_violation|franka::NetworkException' {USER_HOME}/crisp_controllers_demos/current.log"], capture_output=True, text=True).stdout
        if len(error_string) < 5:
            t.join()
            continue
        
        # look for the time code when the container crashed
        crash_time_str = error_string.split(" ")[0].strip("[]") # Time code in format 2025-10-16_10:45:26.554086 
        t.cancel()
        crash_time_struct = time.strptime(crash_time_str.split(".")[0], "%Y-%m-%d_%H:%M:%S")
        crash_timestamp = time.mktime(crash_time_struct) + float("0." + crash_time_str.split(".")[1])


        last_start_time_containing_string = subprocess.run(["ssh", USER_AT_FRANKA, fr"head -n 1 {USER_HOME}/crisp_controllers_demos/current.log"], capture_output=True, text=True).stdout.split(" ")[0]

        if 'cartesian_reflex' in error_string:
            out_queue.put((crash_timestamp, "E_TORQUE"))
        else:
            out_queue.put((crash_timestamp, "E_CONTROLLER_ISSUE"))
        
        time.sleep(20)

        # wait until new container has launched
        while subprocess.run(["ssh", USER_AT_FRANKA, fr"head -n 1 {USER_HOME}/crisp_controllers_demos/current.log"], capture_output=True, text=True).stdout.split(" ")[0] == last_start_time_containing_string:
            time.sleep(2)
        
        # wait until topics are available
        while "/joint_trajectory_controller/state" not in subprocess.run(["ssh", USER_AT_FRANKA, r"source /opt/ros/humble/setup.bash && ROS_DOMAIN_ID=101 ros2 topic list"], capture_output=True, text=True).stdout:
            time.sleep(5)
        out_queue.put((time.time(), "E_CONTROLLER_READY"))


    

def clear_terminal():
    # Clear visible screen
    os.system('cls' if os.name == 'nt' else 'clear')

KEY_EVENT_MAP = {'r': "E_READY", 'q': "E_QUIT", 'd': "E_DROP", 'p': "E_PUSH_OFF", 'l': "E_LOCKED", 'n': "E_NOT_LOCKED"}

class KeyboardInputHandler:
    def __init__(self, out_queue):
        self.out_queue = out_queue
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        now = time.time()
        try:
            if key.char in "rqdpln":                
                # clear_terminal()
                print(f"[CLI] Recognized {key.char}")
                self.out_queue.put((now, KEY_EVENT_MAP[key.char]))
        except AttributeError:
            pass  # special keys ignored





max_step_velocity = 0.01
max_gripper_step = 0.05

class PlaybackActionSource:
    def __init__(self, trajectory: list[np.ndarray]):
        self.trajectory = trajectory
        self.i = 0
        # self.sub = []
        self.current_action = np.zeros(7)
        pass

    def reset(self):
        self.i = 0
        self.current_action = np.zeros(7)
        # self.sub = []

    def next(self, obs) -> np.ndarray | None:
        clip_act = max_step_velocity
        if np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > max_step_velocity:
            return np.zeros(7)
        if np.any(self.current_action != 0.0):
            next_action = np.clip(self.current_action, np.array([-clip_act, -clip_act, -clip_act, 0.0, 0.0, 0.0, -max_gripper_step]), np.array([clip_act, clip_act, clip_act, 0.0, 0.0, 0.0, max_gripper_step]))
            self.current_action -= next_action
            return next_action

        if self.i >= len(self.trajectory):
            return None
        act = np.copy(self.trajectory[self.i])
        self.i += 1
        self.current_action = act
        next_action = np.clip(self.current_action, np.array([-clip_act, -clip_act, -clip_act, 0.0, 0.0, 0.0, -max_gripper_step]), np.array([clip_act, clip_act, clip_act, 0.0, 0.0, 0.0, max_gripper_step]))
        self.current_action -= next_action
        return next_action



def main():
    config = make_env_config("my_env")
    env = ManipulatorCartesianEnv(config=config)
    print("Env created.")
    env.wait_until_ready()
    print("Env ready.")

    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default=None, help="Set the checkpoint name for the experiment. Per default the timestamp is used.")
    argparse.add_argument("--load_policy", type=str, help="Checkpoint name of policy to be loaded.")
    argparse.add_argument("--resume_training", type=str, help="Checkpoint name to be resumed from. This will load the policy, image encoder and replay buffer.")
    args = argparse.parse_args()

    alg_name = "PreProgrammedPolicy"

    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"{alg_name}__{time.strftime('%Y%m%d-%H%M%S')}"
    
    os.mkdir(f"rollouts/{run_name}")
    run_number = 0

    action_source = PlaybackActionSource(load_actions_safe("v2_crafted_rel_acts_insert_plus32.json"))
    approximate_pick_position = np.array([588.63, -138.39,   41.77]) * 0.001
    approximate_place_position = np.array([593.94, -34.18,  53.97]) * 0.001

    ctx = mp.get_context("spawn")

    keyboard_event_queue = ctx.Queue()
    keyboard_input_handler = KeyboardInputHandler(keyboard_event_queue)

    controller_container_watcher_event_queue = ctx.Queue()
    controller_container_watcher_thread = threading.Thread(target=controller_container_watcher, args=(controller_container_watcher_event_queue,))
    controller_container_watcher_thread.start()

    print(KEY_EVENT_MAP)

    all_events = []
    all_event_strings = []
    while True:
        # Wait for container
        if "E_TORQUE" in all_event_strings or "E_CONTROLLER_ISSUE" in all_event_strings:
            print("Waiting for controller to restart...")
            while True:
                while controller_container_watcher_event_queue.empty():
                    time.sleep(0.3)
                timestamp, event = keyboard_event_queue.get()
                if event == "E_CONTAINER_READY":
                    print(f"[CONTROLLER] Event: {event}")
                    break
                else:
                    print(f"[CONTROLLER] Warning Skipping {event}, should have been container ready")

        obs, _ = custom_reset(env, [0.200, -0.020, -0.200])
  

        while True:
            
            while keyboard_event_queue.empty():
                time.sleep(0.3)
            timestamp, event = keyboard_event_queue.get()
            if  event == "E_QUIT":
                env.close()
                print("Quit.")
                sys.exit(0)
            if event == "E_READY":
                print(f"[CLI] Ready.")
                break
            print(f"[CLI] Skipping {event}, waiting for ready or quit.")

        action_source.reset()


        played_actions = []
        action_timestamps = []
        all_observations = [obs]
        all_events.clear()
        # collect rollout
        while True:
            # step environment
            action = action_source.next(obs)
            if action is None:
                print("Out of actions.")
                break

            played_actions.append(action)
            action_timestamps.append(time.time())
            obs, _reward, terminated, truncated, info = env.step(action)
            obs_timestamp = time.time()
            obs["observation.target.pick_pose"] = np.copy(approximate_pick_position)
            obs["observation.target.place_pose"] = np.copy(approximate_place_position)

            all_observations.append(obs)

            # check for controller container events
            _temp_events = []
            while not controller_container_watcher_event_queue.empty():
                timestamp, event = controller_container_watcher_event_queue.get()
                print(f"[CONTROLLER] Event: {event}")
                _temp_events.append(event)
            for i, (timestamp, event) in enumerate(_temp_events):
                if event in  ["E_TORQUE", "E_CONTROLLER_ISSUE"]:
                    all_events.append((timestamp, event))
                else:
                    print(f"Event {event} does not make sense during env execution.")
                    if i == len(_temp_events) - 1:
                        break
            else:
                if len(_temp_events) > 0:
                    break

            if truncated: 
                break

            # check for state change events
            if len(all_events) == 0:
                gripper_position = obs["observation.state.gripper"]
                if gripper_position < 0.31:
                    delta_to_pick_pose = np.abs(approximate_pick_position - obs["observation.state.cartesian"][:3])
                    # z err < 2mm, xy err <7mm each
                    if delta_to_pick_pose[0] < 0.007 and delta_to_pick_pose[1] < 0.007 and delta_to_pick_pose[2] < 0.002:
                        print(f'SUCCESSFUL GRIP at {obs["observation.state.cartesian"]} (ideally: {approximate_pick_position})')
                        all_events.append((obs_timestamp, "E_SUCCESSFUL_GRIP"))
                    else:
                        print(f'BAD GRIP at {obs["observation.state.cartesian"]} (ideally: {approximate_pick_position})')
                        all_events.append((obs_timestamp, "E_BAD_GRIP"))
                        break
            elif all_events[-1][1] == "E_SUCCESSFUL_GRIP":
                gripper_position = obs["observation.state.gripper"]
                if gripper_position > 0.33:
                    print("Place successful? (l/n)")
                    while True:
                        while keyboard_event_queue.empty():
                            time.sleep(0.3)
                        timestamp, event = keyboard_event_queue.get()
                        if event not in ["E_LOCKED", "E_NOT_LOCKED"]:
                            print(f"[CLI] Skipping {event}, waiting for whether the part was locked.")
                        else:
                            print(f"[CLI] Event: {event}")
                            all_events.append((timestamp, event))
                            break
            
            if len(all_events) > 0 and all_events[-1][1] in ["E_LOCKED", "E_NOT_LOCKED"]:
                break



        all_event_strings = list(map(lambda x: x[1], all_events))
        
        # Save data for complete rollout
        if "E_CONTROLLER_ISSUE" not in all_event_strings:
            with open(f"rollouts/{run_name}/act_{run_number}.pkl", "wb") as f:
                pickle.dump(played_actions, f)
            with open(f"rollouts/{run_name}/tmp_{run_number}.pkl", "wb") as f:
                pickle.dump(action_timestamps, f)
            with open(f"rollouts/{run_name}/obs_{run_number}.pkl", "wb") as f:
                pickle.dump(all_observations, f)
            with open(f"rollouts/{run_name}/evt_{run_number}.pkl", "wb") as f:
                pickle.dump(all_events, f)
            run_number += 1





if __name__ == "__main__":
    main()
