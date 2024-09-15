# -*- coding: utf-8 -*-


import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, optim
from typing import List
from torch.distributions import Categorical, Normal


class Model(nn.Module):

    def __init__(self, n_act, n_obs, d_model, noise=3.0):
        nn.Module.__init__(self)
        self.n_act = n_act

        self.n_obs = n_obs
        self.d_model = d_model
        self.noise = noise

        self.p = nn.Parameter(torch.randn((n_act + 1, d_model, d_model)), requires_grad=True)
        self.init = nn.Parameter(torch.full((d_model, ), 5.0), requires_grad=True)
    
    def sigmoid_and(self, x: Tensor, y: Tensor):
        z = - torch.logaddexp(-x, -y)
        return z
    
    def sigmoid_or(self, x: Tensor, y: Tensor):
        z = torch.logaddexp(x, y)
        return z
    
    def sigmoid_or_reduce(self, x: Tensor):
        return torch.logsumexp(x, dim=-1)

    def matrix_mul(self, m1: Tensor, m2: Tensor):
        B, D1h, D1w = m1.shape
        B, D2h, D2w = m2.shape
        assert D1w == D2h
        m1 = m1.view(B, D1h, 1, D1w).expand(B, D1h, D2w, D1w)
        m2 = m2.permute(0, 2, 1).view(B, 1, D2w, D2h).expand(B, D1h, D2w, D2h)
        m3 = self.sigmoid_and(m1, m2)
        m3 = self.sigmoid_or_reduce(m3.view(-1, D1w))
        m3 = m3.view(B, D1h, D2w)
        return m3
    
    def matrix_cum_mul(self, m: Tensor):
        B, L, D, _ = m.shape
        step = 1
        while step < L:
            m1 = m[:, 0:L-step, :, :].clone()
            m2 = m[:, step:L, :, :].clone()
            m3 = m[:, :step, :, :]
            m4 = self.matrix_mul(m1.view(-1, D, D), m2.view(-1, D, D))
            m4 = m4.view(B, -1, D, D)
            m = torch.cat([m3, m4], dim=1)
            step *= 2
        return m
    
    def state(self, act: Tensor):
        B, L = act.shape
        D = self.d_model
        A = self.n_act + 1

        p = self.p.view(1, A, D, D).expand(B, A, D, D)
        p = p + self.noise * torch.randn_like(p)
        p = p.gather(1, (act + 1).view(B, L, 1, 1).expand(B, L, D, D))

        p = self.matrix_cum_mul(p)
        x = self.init.view(1, 1, D).expand(B * L, 1, D)
        x = self.matrix_mul(x, p.view(-1, D, D))
        x = x.view(B, L, D)

        return x
    
    def obs(self, state: Tensor):
        shape = state.shape
        state = state.view(-1, self.d_model)
        B, D = state.shape

        p = self.p[0].view(1, D, D).expand(B, D, D)
        p = p + self.noise * torch.randn_like(p)

        y = self.matrix_mul(state.view(B, 1, D), p)

        return y.view(shape)
    
    def loss(self, act: Tensor, obs: Tensor):
        B, L = act.shape
        act = act.view(B, L)
        obs = obs.view(B, L)

        x = self.state(act)
        x = self.obs(x)

        x = Categorical(logits=x)
        acc = (x.sample() == obs)
        ix = acc.cummin(dim=-1).values.long().sum(-1)

        loss: Tensor = - x.log_prob(obs)
        loss = loss.gather(-1, (ix % L).view(B, 1))

        acc = acc.all(dim=-1).float().mean().item()

        return loss.mean(), acc


class Task:

    def __init__(self):
        self.length = 16
        self.n_act = 1
        self.n_obs = 1

    def sample(self, n, device):
        act = torch.zeros((n, self.length), device=device, dtype=torch.long)
        obs = torch.zeros((n, self.length), device=device, dtype=torch.long)
        return act, obs
    

class Trainer:

    def __init__(self, task: Task):
        self.task = task
        self.device = torch.device("cuda")
        self.d_model = 64
        self.noise = 3.0

        self.batch_size = 64
        self.n_batch = 4
        self.n_step = int(1e6)
        self.print_freq = 100
        # self.lr = 1e-1
        # self.lr = 1.0
        self.lr = 0.5
    
    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g['lr'] = lr
    
    def train(self):
        n_act = self.task.n_act
        n_obs = self.task.n_obs
        model = Model(n_act, n_obs, self.d_model, self.noise).to(self.device)
        opt = optim.Adam(model.parameters(), lr=self.lr)

        self.model = model
        self.opt = opt

        for step in range(self.n_step):

            opt.zero_grad()
            for i in range(self.n_batch):
                act, obs = self.task.sample(self.batch_size, self.device)
                loss, acc = model.loss(act, obs)
                loss.backward()
            opt.step()

            loss = loss.item()
            print(f"step: {step}, loss: {loss}, acc: {acc}")
            if acc >= 1.0:
                break

            if step % self.print_freq == 0:
                print("sample")
                print(act[0])
                print(obs[0])