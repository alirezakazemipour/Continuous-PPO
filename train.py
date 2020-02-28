from copy import deepcopy
import torch
import numpy as np
from torch import multiprocessing as mp
from test import evaluate_model


class Train:
    def __init__(self, env, agent, max_steps_per_episode, max_iter, epochs, mini_batch_size, epsilon, horizon):
        self.env = env
        self.agent = agent
        self.max_steps_per_episode = max_steps_per_episode
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.horizon = horizon
        self.episode_counter = 0
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        mp.set_start_method('spawn')
        mp.set_sharing_strategy('file_system')

        self.global_running_r = []

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            idxes = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[idxes], actions[idxes], returns[idxes], advs[idxes]

#region train
    def train(self, trajectories):
        print(len(trajectories))

        self.agent.set_to_train_mode()
        returns = self.get_gae(rewards, deepcopy(values), next_value, dones)

        advs = returns - np.vstack(values).reshape((sum([len(values[i]) for i in range(self.n_workers)]), 1))
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        states = np.vstack(states)
        actions = np.squeeze(np.vstack(actions), axis=-1)

        for epoch in range(self.epochs):
            for state, action, q_value, adv in self.choose_mini_batch(self.mini_batch_size,
                                                                      states, actions, returns, advs):
                state = torch.Tensor(state).to(self.agent.device)
                action = torch.Tensor(action).to(self.agent.device)
                adv = torch.Tensor(adv).to(self.agent.device)
                q_value = torch.Tensor(q_value).to(self.agent.device).view((self.mini_batch_size, 1))

                dist, value = self.agent.new_policy(state)
                entropy = dist.entropy().mean()
                new_log_prob = self.calculate_log_probs(self.agent.new_policy, state, action)
                old_log_prob = self.calculate_log_probs(self.agent.old_policy, state, action)
                # ratio = torch.exp(new_log_prob) / (torch.exp(old_log_prob) + 1e-8)
                ratio = (new_log_prob - old_log_prob).exp()

                actor_loss = self.compute_ac_loss(ratio, adv)
                # crtitic_loss = self.agent.critic_loss(q_value, value)
                critic_loss = 0.5 * (q_value - value).pow(2).mean()

                total_loss = critic_loss + actor_loss - 0.01 * entropy
                self.agent.optimize(total_loss)

                return total_loss, entropy, rewards
#endregion

    def equalize_policies(self):
        self.agent.set_weights()

    def step(self):
        for iter in range(self.max_iter):

            self.agent.set_to_train_mode()
            states = []
            actions = []
            rewards = []
            dones = []
            values = []
            next_values = []
            trajectories = []
            horizon = 0

            while horizon < self.horizon:
                episode_reward = 0
                state = self.env.reset()
                for _ in range(self.max_steps_per_episode):
                    horizon += 1
                    action = self.agent.choose_action(state)
                    value = self.agent.get_value(state)
                    next_state, reward, done, _ = self.env.step(action)

                    episode_reward += reward
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(reward)
                    values.append(value)

                    if done:
                        next_value = 0
                        next_values.append(next_value)
                        trajectories.append([states, actions, rewards, dones, values, next_values])
                        break
                    else:
                        next_value = self.agent.get_value(next_state)
                        next_values.append(next_value)
                        state = next_state

                    if horizon == self.horizon:
                        trajectories.append([states, actions, rewards, dones, values, next_values])
                        break



            self.train(trajectories)


    def get_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        returns = [[] for _ in range(self.n_workers)]

        for worker in range(self.n_workers):
            values[worker] = values[worker] + [next_values[worker]]
            gae = 0
            for step in reversed(range(len(rewards[worker]))):
                delta = rewards[worker][step] + gamma * (values[worker][step + 1]) * (1 - dones[worker][step]) - \
                        values[worker][step]
                gae = delta + gamma * lam * (1 - dones[worker][step]) * gae
                returns[worker].insert(0, gae + values[worker][step])

        return np.vstack(returns).reshape((sum([len(returns[i]) for i in range(self.n_workers)]), 1))

    def calculate_ratio(self, states, actions):
        new_policy_log = self.calculate_log_probs(self.agent.new_policy, states, actions)
        old_policy_log = self.calculate_log_probs(self.agent.old_policy, states, actions)
        # ratio = torch.exp(new_policy_log) / (torch.exp(old_policy_log) + 1e-8)
        ratio = torch.exp(old_policy_log - new_policy_log)
        return ratio

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution, _ = model(states)
        return policy_distribution.log_prob(actions)

    def compute_ac_loss(self, ratio, adv):
        r_new = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        loss = torch.min(r_new, clamped_r)
        loss = - loss.mean()
        return loss

    def print_logs(self, total_loss, entropy, rewards):
        if self.episode_counter == 150:
            self.global_running_r.append(rewards.item())
        else:
            self.global_running_r.append(self.global_running_r[-1] * 0.99 + rewards.item() * 0.01)

        print(f"Ep:{self.episode_counter}| "
              f"Ep_Reward:{rewards.item():3.3f}| "
              f"Running_reward:{self.global_running_r[-1]:3.3f}| "
              f"Total_loss:{total_loss.item():3.3f}| "
              f"Entropy:{entropy.item():3.3f}| ")
