try:
    from pynput import keyboard
except ImportError:
    print("pynput not installed, keyboard interrupt will not be available.")
import os


def clear_terminal():
    # Clear visible screen
    os.system("cls" if os.name == "nt" else "clear")


class TrainingCLI:
    def __init__(self, continue_training_event, reward_queue, config):
        self.continue_training_event = continue_training_event
        self.reward_queue = reward_queue
        self.config = config
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        self.current_epsisode = 0

    def menu(self):
        self.continue_training_event.clear()
        print("====================================================")
        print("[Training CLI]")
        print("----------------------------------------------------")
        print("1. Press 'ENTER' to start a new episode")
        print("2. Press 'Ctrl + C' to quit training and save models")
        print("====================================================")
        while True:
            feedback = input().strip().lower()
            if feedback == "s" or feedback == "f" or feedback == "t" or feedback == "":
                clear_terminal()
                print(f"[FEEDBACK] ENTER pressed.")
                print("----------------------------------------------------")
                print(f"Starting episode [{self.current_epsisode}].")
                print("")
                print("====================================================")
                print("Press s for SUCCESS, f for FAILURE, t to TERMINATE")
                print("====================================================")
                self.current_epsisode += 1
                self.continue_training_event.set()
                break
            else:
                print("[FEEDBACK] Invalid input, please press 'ENTER' or 'Ctrl + C'")

    def _on_press(self, key):
        try:
            if key.char == "s":  # success
                self.continue_training_event.clear()
                clear_terminal()
                print("[FEEDBACK] Marked SUCCESS")
                print("\n")
                self.reward_queue.put(self.config.success_reward)
                self.menu()
            elif key.char == "f":  # failure
                self.continue_training_event.clear()
                clear_terminal()
                print("[FEEDBACK] Marked FAILURE")
                print("\n")
                self.reward_queue.put(self.config.failure_reward)
                self.menu()
            elif key.char == "t":  # terminate episode
                clear_terminal()
                print("[FEEDBACK] Episode termination requested.")
                print("\n")
                self.reward_queue.put(0.0)
                self.menu()
        except AttributeError:
            pass  # special keys ignored

    def get_reward(self):
        r, self.reward = self.reward, None
        return r
