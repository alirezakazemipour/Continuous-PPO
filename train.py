from copy import deepcopy
import torch
import numpy as np
import time
from running_mean_std import RunningMeanStd
from test import evaluate_model
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, env, n_iterations, agent, epochs, mini_batch_size, epsilon, horizon):
        self.env = env
        self.test_env = deepcopy(env)
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations

        self.start_time = 0
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))

        self.global_running_r = []

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs, values):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[indices], actions[indices], returns[indices], advs[indices], values[indices]

    # region train
    def train(self, states, actions, returns, advs, values):

        # returns = self.get_returns(rewards, dones)
        returns = advs + np.vstack(values[:-1])
        # advs = returns - np.vstack(values[:-1])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        states = np.vstack(states)
        # self.state_rms.update(states)
        actions = np.vstack(actions)
        values = np.vstack(values[:-1])
        for epoch in range(self.epochs):

            for state, action, return_, adv, old_value in self.choose_mini_batch(self.mini_batch_size,
                                                                                 states, actions, returns, advs,
                                                                                 values):
                state = torch.Tensor(state).to(self.agent.device)
                action = torch.Tensor(action).to(self.agent.device)
                return_ = torch.Tensor(return_).to(self.agent.device)
                adv = torch.Tensor(adv).to(self.agent.device)
                old_value = torch.Tensor(old_value).to(self.agent.device)

                dist = self.agent.new_policy_actor(state)
                entropy = dist.entropy().mean()
                value = self.agent.critic(state)
                clipped_value = old_value + torch.clamp(value - old_value, -self.epsilon, self.epsilon)
                clipped_v_loss = (clipped_value - return_).pow(2)
                unclipped_v_loss = (value - return_).pow(2)
                critic_loss = 0.5 * torch.max(clipped_v_loss, unclipped_v_loss).mean()
                # critic_loss = self.agent.critic_loss(value, return_)

                new_log_prob = self.calculate_log_probs(self.agent.new_policy_actor, state, action)
                with torch.no_grad():
                    old_log_prob = self.calculate_log_probs(self.agent.old_policy_actor, state, action)

                ratio = torch.exp(new_log_prob) / (torch.exp(old_log_prob) + 1e-8)
                actor_loss = self.compute_actor_loss(ratio, adv)

                total_loss = actor_loss + critic_loss  # - 0.0 * entropy

                # self.agent.optimize(total_loss)
                self.agent.optimize(actor_loss, critic_loss)

        return total_loss, actor_loss, critic_loss

    # endregion

    #  endregion

    #  region step
    def step(self):
        for iteration in range(self.n_iterations):
            state = self.env.reset()
            states = []
            actions = []
            rewards = []
            values = []
            dones = []
            self.start_time = time.time()
            self.agent.set_to_eval_mode()
            for t in range(self.horizon):
                # state = np.clip((state - self.state_rms.mean) / self.state_rms.var ** 0.5 + 1e-8, -5, 5)
                action = self.agent.choose_action(state)
                value = self.agent.get_value(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = next_state

            if done:
                next_value = 0
                values.append(next_value)
            else:
                next_value = self.agent.get_value(next_state)
                values.append(next_value)

            advs = self.get_gae(rewards, values, dones)
            returns = self.get_returns(rewards, dones)
            self.agent.set_to_train_mode()
            total_loss, actor_loss, critic_loss = self.train(states, actions, returns, advs, values)
            self.agent.set_weights()
            self.agent.schedule_lr()
            self.agent.set_to_eval_mode()
            eval_rewards = evaluate_model(self.agent, self.test_env, self.state_rms)
            self.print_logs(iteration, total_loss, actor_loss, critic_loss, eval_rewards)

    #  endregion

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):

        advs = []
        gae = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        advs.reverse()
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
        return policy_distribution.log_prob(actions).sum(-1, keepdim=True)

    def compute_actor_loss(self, ratio, adv):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()
        return loss

    def print_logs(self, iteration, total_loss, actor_loss, critic_loss, eval_rewards):
        if iteration == 0:
            self.global_running_r.append(eval_rewards)
        else:
            self.global_running_r.append(self.global_running_r[-1] * 0.99 + eval_rewards * 0.01)

        if iteration % 100 == 0:
            print(f"Iter:{iteration}| "
                  f"Ep_Reward:{eval_rewards:3.3f}| "
                  f"Running_reward:{self.global_running_r[-1]:3.3f}| "
                  f"Total_loss:{total_loss.item():3.3f}| "
                  f"Actor_Loss:{actor_loss:3.3f}| "
                  f"Critic_Loss:{critic_loss:3.3f}| "
                  f"Iter_duration:{time.time() - self.start_time:3.3f}| "
                  f"lr:{self.agent.actor_scheduler.get_last_lr()}")
            self.agent.save_weights()

        with SummaryWriter("./logs") as writer:
            writer.add_scalar("Episode running reward", self.global_running_r[-1], iteration)
            writer.add_scalar("Episode reward", eval_rewards, iteration)
            writer.add_scalar("Actor loss", actor_loss, iteration)
            writer.add_scalar("Critic loss", critic_loss, iteration)
