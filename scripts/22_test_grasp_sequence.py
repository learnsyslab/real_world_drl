import time
from crisp_drl.agents.shared.config import Config
from crisp_drl.envs.make_env import create_real_env_v3

env = create_real_env_v3(config=Config())

i = 0
try:
    while True:
        env.reset()
        time.sleep(1.0)
        print(i)
        i += 1
except KeyboardInterrupt:
    print("Interrupted, closing env...")
    env.close()
