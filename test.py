import numpy as np


def evaluate_model(agent, env, state_rms):
    total_rewards = 0
    s = env.reset()
    done = False
    for _ in range(env._max_episode_steps):
        # s = np.clip((s - state_rms.mean) / state_rms.var ** 0.5, -5.0, 5.0)
        action = agent.choose_action(s)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        total_rewards += reward
        if done:
            break
        s = next_state
    env.close()
    return total_rewards
