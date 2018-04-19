import torch
from torch.optim.optimizer import Optimizer, required

import numpy as np


def kth(arr, topk, sample_rate=1):
    # to numpy array
    arr = arr.numpy().ravel()

    if sample_rate < 1:
        arr = np.random.choice(arr, int(arr.size * sample_rate), replace=False)

    arr = np.abs(arr)
    num = arr.size

    k = max(1, topk * num // 100)
    ids = np.argpartition(arr, -k)[-k:]
    thr = float(np.min(arr[ids]))

    return thr


class DGC(Optimizer):
    r"""Implement Deep Gradient Compression for momentum SGD.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        topk: keep topk percent gradients
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        max_val (float, optinal): clip graidient if abs is greater than max_val

    Example:
        >>> optimizer = DGC(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0, topk=1,
                 weight_decay=0, nesterov=False, 
                 max_val=None, sample_rate=0.1):
        defaults = dict(lr=lr, momentum=momentum, topk=topk,
                        weight_decay=weight_decay, nesterov=nesterov,
                        max_val=max_val, sample_rate=sample_rate)
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        super(DGC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DGC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            topk = group['topk']
            max_val = group['max_val']
            sample_rate = group['sample_rate']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # clip gradient
                if max_val is not None and max_val > 0:
                    d_p.clamp_(-max_val, max_val)

                if momentum != 0:
                    param_state = self.state[p]

                    if 'u_buffer' not in param_state:
                        param_state['u_buffer'] = d_p.clone()
                    u = param_state['u_buffer']

                    if 'v_buffer' not in param_state:
                        param_state['v_buffer'] = d_p.clone()
                    v = param_state['v_buffer']

                    if nesterov:
                        u.add_(d_p).mul_(momentum)
                        v.add_(u + d_p)
                    else:
                        u.mul_(momentum).add_(d_p)
                        v.add_(u)

                    # threshold
                    thr = kth(v, topk, sample_rate=sample_rate)

                    mask = (v.abs() >= thr).type(d_p.type())
                    nmask = (v.abs() < thr).type(d_p.type())

                    torch.mul(v, mask, out=d_p)

                    torch.mul(v, nmask, out=v)
                    torch.mul(u, nmask, out=u)
                else:  # SGD
                    param_state = self.state[p]
                    if 'g_buffer' not in param_state:
                        param_state['g_buffer'] = d_p.clone()
                    g = param_state['g_buffer']
                    g.add_(d_p)

                    # threshold
                    thr = kth(g, topk, sample_rate=sample_rate)

                    mask = (g.abs() >= thr).type(d_p.type())
                    nmask = (g.abs() < thr).type(d_p.type())

                    torch.mul(g, mask, out=d_p)
                    torch.mul(g, nmask, out=g)

                p.data.add_(-group['lr'], d_p)

        return loss
