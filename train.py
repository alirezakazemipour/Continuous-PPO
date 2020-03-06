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
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs, probs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[indices], actions[indices], returns[indices], advs[indices], probs[indices]

    # region train
    def train(self, states, actions, rewards, dones, values, old_log_probs):
        self.agent.set_to_train_mode()

        advs = self.get_gae(rewards, deepcopy(values), dones)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        returns = self.get_returns(rewards, dones)
        # returns = advs + np.vstack(values[:-1])

        states = np.vstack(states)
        # self.state_rms.update(states)
        actions = np.vstack(actions)
        old_log_probs = np.vstack(old_log_probs)

        for state, action, return_, adv, old_log_prob in self.choose_mini_batch(self.mini_batch_size,
                                                                                states, actions, returns, advs,
                                                                                old_log_probs):
            # state = np.clip((state - self.state_rms.mean) / self.state_rms.var, -5.0, 5.0)
            state = torch.Tensor(state).to(self.agent.device)
            action = torch.Tensor(action).to(self.agent.device)
            adv = torch.Tensor(adv).to(self.agent.device)
            old_log_prob = torch.Tensor(old_log_prob).to(self.agent.device)  # .sum(dim=-1)
            return_ = torch.Tensor(return_).to(self.agent.device).view((self.mini_batch_size, 1))

            for epoch in range(self.epochs):
                value = self.agent.critic(state)
                new_log_prob = self.calculate_log_probs(self.agent.new_policy_actor, state, action)
                ratio = torch.exp(new_log_prob) / (torch.exp(old_log_prob) + 1e-8)

                actor_loss = self.compute_actor_loss(ratio, adv)
                critic_loss = 0.5 * (return_ - value).pow(2).mean()

                total_loss = critic_loss + actor_loss

                self.agent.optimize(actor_loss, critic_loss)
                # self.agent.optimize(total_loss)

        return total_loss

    # endregion

    #  endregion

    #  region step
    def step(self):
        state = self.env.reset()
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        while True:
            self.start_time = time.time()

            # state = np.clip((state - self.state_rms.mean) / self.state_rms.var, -5.0, 5.0)
            action = self.agent.choose_action(state)
            value = self.agent.get_value(state)[0]
            next_state, reward, done, _ = self.env.step(action)
            with torch.no_grad():
                s = torch.Tensor(state).to(self.agent.device).unsqueeze(dim=0)
                a = torch.Tensor(action).to(self.agent.device)
                old_log_prob = self.calculate_log_probs(self.agent.new_policy_actor, s, a).detach().cpu().numpy()[0]

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            log_probs.append(old_log_prob)

            if done:
                state = self.env.reset()
            else:
                state = next_state

            if self.time_step > 0 and self.time_step % (self.horizon - 1) == 0:
                if done:
                    next_value = 0
                    values.append(next_value)
                else:
                    next_value = self.agent.get_value(next_state)
                    values.append(next_value)

                loss = self.train(states, actions, rewards, dones, values, log_probs)
                eval_rewards = evaluate_model(deepcopy(self.agent), self.test_env, deepcopy(self.state_rms))
                self.print_logs(loss, eval_rewards)

                states = []
                actions = []
                rewards = []
                dones = []
                values = []
                log_probs = []
                self.agent.set_to_eval_mode()
                # self.agent.schedule_lr()

            self.time_step += 1

    #  endregion

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):

        advs = []
        gae = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        return np.vstack(advs)

    @staticmethod
    def get_returns(rewards, dones, gamma=0.99):

        returns = []
        running_returns = 0
        for step in reversed(range(len(rewards))):
            running_returns = rewards[step] + gamma * running_returns * (1 - dones[step])
            returns.insert(0, running_returns)

        return np.vstack(returns)

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution = model(states)
        return policy_distribution.log_prob(actions)  # .sum(axis=-1)

    def compute_actor_loss(self, ratio, adv):
        r_new = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        loss = torch.min(r_new, clamped_r)
        loss = - loss.mean()
        return loss

    def print_logs(self, total_loss, rewards):
        if self.time_step == self.horizon - 1:
            self.global_running_r.append(rewards)
        else:
            self.global_running_r.append(self.global_running_r[-1] * 0.99 + rewards * 0.01)

        if self.time_step % ((self.horizon - 1) * 100) == 0:
            print(f"Iter:{self.time_step // self.horizon - 1}| "
                  f"Ep_Reward:{rewards:3.3f}| "
                  f"Running_reward:{self.global_running_r[-1]:3.3f}| "
                  f"Total_loss:{total_loss.item():3.3f}| "
                  f"Iter_duration:{time.time() - self.start_time:3.3f}| "
                  f"lr:{self.agent.actor_scheduler.get_last_lr()}")
