import numpy as np


def evaluate_model(agent, env, state_rms):
    total_rewards = 0
    s = env.reset()
    done = False
    while not done:
        s = np.clip((s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5.0, 5.0)
        dist = agent.choose_dist(s)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        # env.render()
        s = next_state
        total_rewards += reward
    # env.close()
    return total_rewards
