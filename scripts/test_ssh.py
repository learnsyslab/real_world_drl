import multiprocessing as mp
import subprocess, os
import time






def my_fn():
    # Copy your env, but clear out library paths
    # env = os.environ.copy()
    # env.pop("LD_LIBRARY_PATH", None)
    # env.pop("DYLD_LIBRARY_PATH", None)  # For macOS
    # env.pop("SSL_CERT_FILE", None)
    # env.pop("OPENSSL_CONF", None)
    error_out = subprocess.run(["ssh", "linusschwarz@franka", r"grep -P 'cartesian_reflex|communication_constraints_violation|franka::NetworkException' /home/linusschwarz/crisp_controllers_demos/current.log"], capture_output=True, text=True) #, env=env)

    print("Errors and Warnings from robot log:")
    print(error_out.stdout)
    if error_out.stderr:
        print("Error while fetching robot log:")
        print(error_out.stderr)

def my_outer_fn():
    time.sleep(1)
    controller_container_watcher_thread = mp.Process(target=my_fn)
    controller_container_watcher_thread.daemon = True # such that it is automatically stopped when the main thread exits
    controller_container_watcher_thread.start()
    time.sleep(100)


def main():
    ctx = mp.get_context("spawn")
    actor_process = ctx.Process(target=my_outer_fn)
    actor_process.start()

    time.sleep(100)  # Let it run for a while


if __name__ == "__main__":
    main()
