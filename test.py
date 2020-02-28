import cv2
from collections import deque
import numpy as np

stacked_states = deque([np.zeros((84, 84)) for _ in range(4)], maxlen=4)


def evaluate_model(agent, env):
    total_rewards = 0
    s = env.reset()
    done = False
    # s = stack_state(s.copy(), True)
    while not done :
        action, _ = agent.choose_action(s)
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        # s = stack_state(next_state, False)
        s = next_state
        # env.render()
    # env.close()
    return total_rewards


def stack_state(s, new_episode=False):
    global stacked_states
    s = pre_process(s)

    if new_episode:
        stacked_states = deque([np.zeros((84, 84)) for _ in range(4)], maxlen=4)
        stacked_states.append(s)
        stacked_states.append(s)
        stacked_states.append(s)
        stacked_states.append(s)
    else:
        stacked_states.append(s)

    return np.stack(stacked_states, axis=2)


def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (84, 84))

    return img / 255.0
