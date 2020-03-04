from model import Actor, Critic
from torch.optim import Adam
from torch import from_numpy
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy


class Agent:
    def __init__(self, n_states, action_bounds, n_actions, actor_lr, critic_lr):
        self.action_bounds = action_bounds
        self.n_actions = n_actions
        self.n_states = n_states
        self.device = "cpu"

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.new_policy_actor = Actor(n_states=self.n_states,
                                      n_actions=self.n_actions).to(self.device)

        self.old_policy_actor = Actor(n_states=self.n_states,
                                      n_actions=self.n_actions).to(self.device)

        self.critic = Critic(n_states=self.n_states).to(self.device)

        self.old_policy_actor.load_state_dict(deepcopy(self.new_policy_actor.state_dict()))
        # self.old_policy_actor.eval()

        self.actor_optimizer = Adam(self.new_policy_actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)

        self.scheduler = lambda step: max(1.0 - float(step * 2048 / 3e+6), 0)

        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=self.scheduler)

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)

    def choose_action(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        dist = self.new_policy_actor(state)
        action = dist.sample().detach().cpu().numpy()
        action = np.squeeze(action, axis=0)
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

        return action

    def get_value(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        value = self.critic(state)

        return value.detach().cpu().numpy()

    def optimize(self, actor_loss, critic_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(self.new_policy_actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

    def schedule_lr(self):
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def set_weights(self):
        for old_params, new_params in zip(self.old_policy_actor.parameters(), self.new_policy_actor.parameters()):
            old_params.data.copy_(new_params.data)

    def save_weights(self):
        torch.save(self.new_policy_actor.state_dict(), "./weights.pth")

    def load_weights(self):
        self.new_policy_actor.load_state_dict(torch.load("./weights.pth"))

    def set_to_eval_mode(self):
        self.new_policy_actor.eval()
        self.critic.eval()

    def set_to_train_mode(self):
        self.new_policy_actor.train()
        self.critic.train()
