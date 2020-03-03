from copy import deepcopy
import torch
import numpy as np
import time
from running_mean_std import RunningMeanStd
from test import evaluate_model


class Train:
    def __init__(self, env, agent, max_iter, epochs, mini_batch_size, epsilon, horizon):
        self.env = env
        self.agent = agent
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.horizon = horizon
        self.iteration_counter = 0
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        self.start_time = 0
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))

        self.global_running_r = []

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            idxes = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[idxes], actions[idxes], returns[idxes], advs[idxes]

    # region train
    def train(self, states, actions, rewards, dones, values):
        self.agent.set_to_train_mode()
        returns = self.get_gae(rewards, deepcopy(values), dones)

        advs = returns - np.vstack(values[:-1])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        states = np.vstack(states)
        actions = np.vstack(actions)

        for epoch in range(self.epochs):
            for state, action, q_value, adv in self.choose_mini_batch(self.mini_batch_size,
                                                                      states, actions, returns, advs):

                self.state_rms.update(state)
                state = np.clip((state - self.state_rms.mean) / self.state_rms.var, -5.0, 5.0)
                state = torch.Tensor(state).to(self.agent.device)
                action = torch.Tensor(action).to(self.agent.device)
                adv = torch.Tensor(adv).to(self.agent.device)
                q_value = torch.Tensor(q_value).to(self.agent.device).view((self.mini_batch_size, 1))

                dist = self.agent.new_policy_actor(state)
                value = self.agent.critic(state)
                entropy = dist.entropy().mean()
                new_log_prob = self.calculate_log_probs(self.agent.new_policy_actor, state, action)
                with torch.no_grad():
                    old_log_prob = self.calculate_log_probs(self.agent.old_policy_actor, state, action)
                ratio = (new_log_prob - old_log_prob).exp()
                # print(ratio.mean())

                actor_loss = self.compute_ac_loss(ratio, adv)
                critic_loss = 0.5 * (q_value - value).pow(2).mean()

                total_loss = 1 * critic_loss + actor_loss - 0.0 * entropy

                self.agent.optimize(actor_loss, critic_loss)
                # self.agent.optimize(total_loss)

            return total_loss, entropy, rewards

    # endregion

    # region equalize_policies
    def equalize_policies(self):
        self.agent.set_weights()
    #  endregion

    #  region step
    def step(self):
        for iter in range(self.max_iter):
            self.start_time = time.time()

            self.agent.set_to_train_mode()
            states = []
            actions = []
            rewards = []
            dones = []
            values = []
            log_probs = []

            step_counter = 0
            iteration_reward = 0
            state = self.env.reset()
            self.iteration_counter += 1
            while True:
                step_counter += 1

                # state = np.clip((state - self.state_rms.mean) / self.state_rms.var, -5.0, 5.0)
                action = self.agent.choose_action(state)
                value = self.agent.get_value(state)
                next_state, reward, done, _ = self.env.step(action)

                iteration_reward += reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

                if done:
                    state = self.env.reset()
                else:
                    state = next_state

                if step_counter > 0 and step_counter % self.horizon == 0:
                    if done:
                        next_value = 0
                        values.append(next_value)
                    else:
                        next_value = self.agent.get_value(next_state)
                        values.append(next_value)
                    break
            total_loss, entropy, rewards = self.train(states, actions, rewards, dones, values)
            evaluation_rewards = evaluate_model(self.agent, deepcopy(self.env), deepcopy(self.state_rms))
            self.print_logs(total_loss, entropy, evaluation_rewards)
            self.agent.schedule_lr()
        self.agent.save_weights()
    #  endregion

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):

        returns = []
        gae = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])

        return np.vstack(returns)

    # def calculate_ratio(self, states, actions):
    #     new_policy_log = self.calculate_log_probs(self.agent.new_policy, states, actions)
    #     old_policy_log = self.calculate_log_probs(self.agent.old_policy, states, actions)
    #     # ratio = torch.exp(new_policy_log) / (torch.exp(old_policy_log) + 1e-8)
    #     ratio = torch.exp(old_policy_log - new_policy_log)
    #     return ratio

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution = model(states)
        return policy_distribution.log_prob(actions)

    def compute_ac_loss(self, ratio, adv):
        r_new = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        loss = torch.min(r_new, clamped_r)
        loss = - loss.mean()
        return loss

    def print_logs(self, total_loss, entropy, rewards):
        if self.iteration_counter == 1:
            self.global_running_r.append(rewards)
        else:
            self.global_running_r.append(self.global_running_r[-1] * 0.99 + rewards * 0.01)

        if self.iteration_counter % 100 == 0:
            print(f"Iter:{self.iteration_counter}| "
                  f"Ep_Reward:{rewards:3.3f}| "
                  f"Running_reward:{self.global_running_r[-1]:3.3f}| "
                  f"Total_loss:{total_loss.item():3.3f}| "
                  f"Entropy:{entropy.item():3.3f}| "
                  f"Iter_duration:{time.time() - self.start_time:3.3f}| "
                  f"lr:{self.agent.actor_scheduler.get_last_lr()}")
