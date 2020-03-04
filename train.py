from copy import deepcopy
import torch
import numpy as np
import time
from running_mean_std import RunningMeanStd
from test import evaluate_model


class Train:
    def __init__(self, env, agent, epochs, mini_batch_size, epsilon, horizon):
        self.env = env
        self.test_env = deepcopy(env)
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.time_step = 0
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        self.start_time = 0
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))

        self.global_running_r = []

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[indices], actions[indices], returns[indices], advs[indices]

    # region train
    def train(self, states, actions, rewards, dones, values):
        self.agent.set_to_train_mode()
        returns = self.get_gae(rewards, deepcopy(values), dones)

        advs = returns - np.vstack(values[:-1])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        states = np.vstack(states)
        self.state_rms.update(states)
        actions = np.vstack(actions)

        for epoch in range(self.epochs):
            # print(f"----------Epoch:{epoch}-------------")
            # i = 0
            for state, action, return_, adv in self.choose_mini_batch(self.mini_batch_size,
                                                                      states, actions, returns, advs):
                # i += 1
                # print(f"----------Batch idx:{i}-------------")

                # state = np.clip((state - self.state_rms.mean) / self.state_rms.var, -5.0, 5.0)
                state = torch.Tensor(state).to(self.agent.device)
                action = torch.Tensor(action).to(self.agent.device)
                adv = torch.Tensor(adv).to(self.agent.device)
                return_ = torch.Tensor(return_).to(self.agent.device).view((self.mini_batch_size, 1))

                # dist = self.agent.new_policy_actor(state)
                value = self.agent.critic(state)
                # with torch.no_grad():
                # entropy = dist.entropy().mean()
                entropy = 0
                new_log_prob = self.calculate_log_probs(self.agent.new_policy_actor, state, action)
                with torch.no_grad():
                    old_log_prob = self.calculate_log_probs(self.agent.old_policy_actor, state, action)
                # ratio = (new_log_prob - old_log_prob).exp()
                ratio = torch.exp(new_log_prob) / (torch.exp(old_log_prob) + 1e-8)

                actor_loss = self.compute_actor_loss(ratio, adv)
                critic_loss = 0.5 * (return_ - value).pow(2).mean()

                total_loss = 1 * critic_loss + actor_loss - 0.0 * entropy

                self.agent.optimize(actor_loss, critic_loss)
                # self.agent.optimize(total_loss)

        return total_loss, entropy

    # endregion

    # region equalize_policies
    def equalize_policies(self):
        self.agent.set_weights()

    #  endregion

    #  region step
    def step(self):
        state = self.env.reset()
        states = []
        actions = []
        rewards = []
        dones = []
        values = []

        while True:
            self.start_time = time.time()

            action = self.agent.choose_action(state)
            value = self.agent.get_value(state)
            next_state, reward, done, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            if done:
                state = self.env.reset()

            if self.time_step > 0 and self.time_step % (self.horizon - 1) == 0:
                if done:
                    next_value = 0
                    values.append(next_value)
                else:
                    next_value = self.agent.get_value(next_state)
                    values.append(next_value)

                self.equalize_policies()
                loss, entropy = self.train(states, actions, rewards, dones, values)
                eval_rewards = evaluate_model(deepcopy(self.agent), self.test_env, deepcopy(self.state_rms))
                self.print_logs(loss, entropy, eval_rewards)

                states = []
                actions = []
                rewards = []
                dones = []
                values = []

            self.time_step += 1

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

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution = model(states)
        return policy_distribution.log_prob(actions)

    def compute_actor_loss(self, ratio, adv):
        r_new = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        loss = torch.min(r_new, clamped_r)
        loss = - loss.mean()
        return loss

    def print_logs(self, total_loss, entropy, rewards):
        if self.time_step == self.horizon - 1:
            self.global_running_r.append(rewards)
        else:
            self.global_running_r.append(self.global_running_r[-1] * 0.99 + rewards * 0.01)

        if self.time_step % 100 == 0:
            print(f"Iter:{self.time_step // 100}| "
                  f"Ep_Reward:{rewards:3.3f}| "
                  f"Running_reward:{self.global_running_r[-1]:3.3f}| "
                  f"Total_loss:{total_loss.item():3.3f}| "
                  # f"Entropy:{entropy.item():3.3f}| "
                  f"Iter_duration:{time.time() - self.start_time:3.3f}| "
                  f"lr:{self.agent.actor_scheduler.get_last_lr()}")
