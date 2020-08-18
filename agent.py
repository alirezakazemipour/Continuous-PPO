from model import Actor, Critic
from torch.optim import Adam
from torch import from_numpy
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


class Agent:
    def __init__(self, env_name, n_iter, n_states, action_bounds, n_actions, lr):
        self.env_name = env_name
        self.n_iter = n_iter
        self.action_bounds = action_bounds
        self.n_actions = n_actions
        self.n_states = n_states
        self.device = torch.device("cpu")

        self.lr = lr

        self.current_policy = Actor(n_states=self.n_states,
                                    n_actions=self.n_actions).to(self.device)

        self.critic = Critic(n_states=self.n_states).to(self.device)

        self.optimizer = Adam(list(self.current_policy.parameters()) + list(self.critic.parameters()), lr=self.lr,
                              eps=1e-5)

        self.critic_loss = torch.nn.MSELoss()

        self.scheduler = lambda step: max(1.0 - float(step / self.n_iter), 0)

        self.total_scheduler = LambdaLR(self.optimizer, lr_lambda=self.scheduler)

    def choose_dist(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            dist = self.current_policy(state)

        # action *= self.action_bounds[1]
        # action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

        return dist

    def get_value(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            value = self.critic(state)

        return value.detach().cpu().numpy()

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

    def schedule_lr(self):
        self.total_scheduler.step()

    def save_weights(self):
        torch.save(self.current_policy.state_dict(), self.env_name + "_weights.pth")

    def load_weights(self):
        self.current_policy.load_state_dict(torch.load(self.env_name + "_weights.pth"))

    def set_to_eval_mode(self):
        self.current_policy.eval()
        self.critic.eval()

    def set_to_train_mode(self):
        self.current_policy.train()
        self.critic.train()
