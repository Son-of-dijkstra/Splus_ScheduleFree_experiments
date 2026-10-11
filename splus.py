import torch
from torch.optim.optimizer import Optimizer
from typing import Any

class SPlus(Optimizer):

    def __init__(
           self,
            params: Any,
            lr: float = 1e-1,
            b1: float = 0.9,
            b2: float = 0.999,
            ema_rate: float = 0.999,
            inverse_every: int = 100,
            eps: float = 1e-30,
            weight_decay: float = 1e-2,
            max_dim: int = 10000,
            nonstandard_constant: float = 0.001,
        ):
        defaults = dict(
            lr=lr,
            b1=b1,
            b2=b2,
            ema_rate=ema_rate,
            inverse_every=inverse_every,
            weight_decay=weight_decay,
            eps=eps,
            max_dim=max_dim,
            nonstandard_constant=nonstandard_constant,
        )
        super(SPlus, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for p in group["params"]:
                p_state = self.state.get(p, [])
                step_val = float(p_state["step"])
                p_state["step"] = torch.tensor(step_val)

    @torch.no_grad()
    def eval(self): # Sets parameters to EMA values for evaluation.
        for group in self.param_groups:
            if 'train_mode' in group:
                train_mode = group['train_mode']
                ema_rate = group['ema_rate']
                if train_mode:
                    for p in group['params']:
                        state = self.state[p]
                        state['param_buffer'] = p.clone()
                        p.lerp_(state['ema'], 1)
                        p.mul_(1 / (1 - ema_rate ** state['step']))
                    group['train_mode'] = False

    @torch.no_grad()
    def train(self): # Resets parameters back from buffer.
        for group in self.param_groups:
            if 'train_mode' in group:
                train_mode = group['train_mode']
                if not train_mode:
                    for p in group['params']:
                        state = self.state[p]
                        if 'param_buffer' in state:
                            p.lerp_(state['param_buffer'], 1) # p.copy_(state['param_buffer'])
                            del state['param_buffer']
                    group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                state = self.state[param]
                if len(state) == 0: #Initialization
                    state["step"] = torch.tensor(0.0)
                    state['momentum'] = torch.zeros_like(param)
                    if group['ema_rate'] is not None:
                        state['ema'] = torch.zeros_like(param)
                    group['train_mode'] = True
                    if len(param.shape) == 2:
                        state['sides'] = [torch.zeros((d, d), device=param.device) if d < group['max_dim'] else None for d in param.shape]
                        state['q_sides'] = [torch.eye(d, device=param.device) if d < group['max_dim'] else None for d in param.shape]


                # Scaling dependent on shape
                if len(param.shape) != 2 or param.shape[0] > group['max_dim'] or param.shape[1] > group['max_dim']:
                    scaled_lr = group['lr'] * group['nonstandard_constant']
                else:
                    scaled_lr = group['lr'] * (2 / (param.shape[0] + param.shape[1]))

                #Main update
                state['step'] +=1
                m = state['momentum']
                m.lerp_(grad, 1-group["b1"])
                if len(param.shape) == 2:
                    m = state['q_sides'][0].T @ m if state['q_sides'][0] is not None else m
                    m = m @ state['q_sides'][1] if state['q_sides'][1] is not None else m

                    state['sides'][0] = torch.lerp(state['sides'][0], grad @ grad.T, 1 - group['b2']) if state['sides'][0] is not None else None
                    state['sides'][1] = torch.lerp(state['sides'][1], grad.T @ grad, 1 - group['b2']) if state['sides'][1] is not None else None

                    u = torch.sign(m)
                    u = state['q_sides'][0] @ u if state['q_sides'][0] is not None else u
                    u = u @ state['q_sides'][1].T if state['q_sides'][1] is not None else u


                    #Eigendecomposition every `inverse_every` steps
                    if state['step'] == 1 or state['step'] % group['inverse_every'] == 0:
                        if state['sides'][0] is not None:
                            _, eigvecs = torch.linalg.eigh(state['sides'][0] + group['eps'] * torch.eye(state['sides'][0].shape[0], device=param.device))
                            state['q_sides'][0] = eigvecs
                        if state['sides'][1] is not None:
                            _, eigvecs = torch.linalg.eigh(state['sides'][1] + group['eps'] * torch.eye(state['sides'][1].shape[0], device=param.device))
                            state['q_sides'][1] = eigvecs
                else:
                    u = torch.sign(m)

                param.add_(u, alpha=-scaled_lr)
                if 'ema' in state:
                    state['ema'].lerp_(param, 1 - group['ema_rate'])
                param.mul_(1 - scaled_lr * group["weight_decay"])

        return loss