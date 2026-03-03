# type: ignore  # This file is adapted from external source and kept untyped
"""
Adapted from: https://github.com/Liuhong99/Sophia - Sophia: A Scalable
    Stochastic Second-order Optimizer for Language Model Pre-training
Original Author: Liuhong99
Notes: This version may include modifications.

License: MIT License

Copyright (c) 2023 Hong Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class SophiaG(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=1e-1,
        *,
        maximize: bool = False,
        capturable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter at index 1: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "rho": rho,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "capturable": capturable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            _beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError("Hero does not support sparse gradients")
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])
                hessian.append(state["hessian"])

                if self.defaults["capturable"]:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(
                params_with_grad,
                grads,
                exp_avgs,
                hessian,
                state_steps,
                bs=bs,
                beta1=beta1,
                beta2=beta2,
                rho=group["rho"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                capturable=group["capturable"],
            )

        return loss


def sophiag(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    hessian: list[Tensor],
    state_steps: list[Tensor],
    capturable: bool = False,
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument "
            "must contain a list of singleton tensors"
        )

    func = _single_tensor_sophiag

    func(
        params,
        grads,
        exp_avgs,
        hessian,
        state_steps,
        bs=bs,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        capturable=capturable,
    )


def _single_tensor_sophiag(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    hessian: list[Tensor],
    state_steps: list[Tensor],
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    capturable: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            if not param.is_cuda and step_t.is_cuda and bs.is_cuda:
                raise RuntimeError("Capturable tensors must be on the same device")

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if capturable:
            step = step_t  # type: ignore
            step_size = lr
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step = step_t.item()  # type: ignore  # noqa: F841
            step_size_neg = -lr

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
