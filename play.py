import torch
from torch import device
from mujoco_py.generated import const
from mujoco_py import GlfwContext
import numpy as np

GlfwContext(offscreen=True)


class Play:
    def __init__(self, env, agent, state_rms, max_episode=4):
        self.env = env
        # self.env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.state_rms = state_rms
        self.device = device("cpu")

    def evaluate(self):

        for _ in range(self.max_episode):
            s = self.env.reset()
            done = False
            episode_reward = 0
            # x = input("Push any button to proceed...")
            for _ in range(self.env._max_episode_steps):
                s = np.clip((s - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5.0, 5.0)
                dist = self.agent.choose_dist(s)
                action = dist.sample().cpu().numpy()[0]
                s_, r, done, _ = self.env.step(action)
                episode_reward += r
                if done:
                    break
                s = s_
                self.env.render(mode="human")
                self.env.viewer.cam.type = const.CAMERA_FIXED
                self.env.viewer.cam.fixedcamid = 0
                # time.sleep(0.03)
            print(f"episode reward:{episode_reward:3.3f}")
