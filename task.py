# -*- coding: utf-8 -*-

import hmm
import torch


class Remember(hmm.Task):

    def __init__(self, n_card=2, length=16):
        self.length = length

        self.n_card = n_card
        self.wait = n_card
        self.ask = self.wait + 1
        self.end = self.ask + 1

        self.n_act = self.end + 1
        self.n_obs = self.end + 1

    def sample(self, n, device):
        act = torch.zeros((n, self.length), dtype=torch.long, device=device)
        obs = torch.zeros((n, self.length), dtype=torch.long, device=device)

        _l = torch.randint(2, self.length + 1, size=(n, 1), device=device)
        ix = torch.arange(self.length, device=device).view(1, -1)
        act[ix < _l] = self.wait
        act[ix >= _l] = self.end
        act[ix == _l - 1] = self.ask
        act[:, 0] = torch.randint(self.n_card, size=(n, ), device=device)

        obs[ix < _l] = self.wait
        obs[ix >= _l] = self.end
        obs[act == self.ask] = act[:, 0]

        return act, obs


class Count(hmm.Task):

    def __init__(self, n_max=6, length=16):
        assert n_max < length

        self.length = length
        self.n_max = n_max
        self.zero = 0
        self.one = 1
        self.wait = n_max + 1
        self.ask = self.wait + 1
        self.end = self.ask + 1

        self.n_act = self.end + 1
        self.n_obs = self.end + 1
    
    def sample(self, n, device):
        ones = torch.randint(self.n_max + 1, size=(n, 1), device=device)
        zeros = torch.rand((n, 1), device=device)
        zeros = (self.length - ones) * zeros
        zeros = zeros.long()

        order = torch.rand((n, self.length), device=device)
        ix = torch.arange(self.length, device=device).view(1, -1)
        order[ix >= (ones + zeros)] = 2
        order = order.argsort(dim=-1)

        act = torch.zeros((n, self.length), dtype=torch.long, device=device)
        obs = torch.zeros((n, self.length), dtype=torch.long, device=device)

        act[ix < (ones + zeros)] = self.zero
        act[order < ones] = self.one
        act[ix > (ones + zeros)] = self.end
        act[ix == (ones + zeros)] = self.ask

        obs[ix < (ones + zeros)] = self.wait
        obs[ix > (ones + zeros)] = self.end
        obs[act == self.ask] = ones.flatten()

        return act, obs


class Repeat(hmm.Task):

    def __init__(self, n_card=2, length=16, gap=4):
        self.n_card = n_card
        self.length = length
        self.gap = gap

        self.wait = n_card
        self.ask = self.wait + 1
        self.end = self.ask + 1
        self.n_act = self.end + 1
        self.n_obs = self.end + 1

    def sample(self, n, device):
        act = torch.randint(self.n_card, size=(n, self.length), device=device)
        obs = torch.full((n, self.length), self.wait, dtype=torch.long, device=device)

        _l = torch.randint(self.gap + 1, self.length + 1, size=(n, 1), device=device)
        ix = torch.arange(self.length, device=device).view(1, -1)
        act[ix >= _l] = self.end
        act[ix == _l - 1] = self.ask
        obs[ix < _l - 1] = self.wait
        obs[ix >= _l] = self.end
        obs[act == self.ask] = act[ix == _l - 1 - self.gap]

        return act, obs


if __name__ == "__main__":

    # trainer = hmm.Trainer(Remember())
    # trainer = hmm.Trainer(Count())
    trainer = hmm.Trainer(Repeat())
    trainer.train()