"""Learning-rate schedulers and scheduler factory."""

import math
from typing import Any, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing with warmup and optional cycle restarts."""

    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        if warmup_steps >= first_cycle_steps:
            raise ValueError("warmup_steps must be less than first_cycle_steps")

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs

        if self.step_in_cycle < self.warmup_steps:
            if self.warmup_steps == 0:
                return [self.max_lr for _ in self.base_lrs]
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]

        progress = (self.step_in_cycle - self.warmup_steps) / (
            self.cur_cycle_steps - self.warmup_steps
        )
        return [
            base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1

            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
                self.max_lr = self.max_lr * self.gamma
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1,
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult ** n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class OneCycleLRWithWarmup(_LRScheduler):
    """One-cycle schedule with a separate linear warmup phase."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-7,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.initial_lr = max_lr / div_factor
        self.min_post_warmup_lr = max_lr / final_div_factor

        super().__init__(optimizer, last_epoch)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return [self.max_lr for _ in self.optimizer.param_groups]
            progress = step / self.warmup_steps
            return [
                self.initial_lr + (self.max_lr - self.initial_lr) * progress
                for _ in self.optimizer.param_groups
            ]

        post_warmup_steps = self.total_steps - self.warmup_steps
        post_warmup_current_step = step - self.warmup_steps
        progress = post_warmup_current_step / post_warmup_steps

        if self.anneal_strategy == "cos":
            return [
                self.min_post_warmup_lr
                + (self.max_lr - self.min_post_warmup_lr)
                * (1 + math.cos(math.pi * progress))
                / 2
                for _ in self.optimizer.param_groups
            ]

        return [
            self.max_lr + (self.min_post_warmup_lr - self.max_lr) * progress
            for _ in self.optimizer.param_groups
        ]


class LRSchedulerFactory:
    """Create PyTorch or Lightning-compatible scheduler configs."""

    @staticmethod
    def create(
        scheduler_type: str,
        optimizer: Optimizer,
        num_epochs: int,
        warmup_epochs: int = 5,
        eta_min: float = 1e-6,
        **kwargs,
    ) -> Union[_LRScheduler, Dict[str, Any]]:
        if scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "max"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 3),
                min_lr=eta_min,
                verbose=kwargs.get("verbose", True),
            )
            return {
                "scheduler": scheduler,
                "monitor": kwargs.get("monitor", "val/iou_epoch"),
                "interval": "epoch",
                "frequency": 1,
            }

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=eta_min,
            )
            return {"scheduler": scheduler, "interval": "epoch"}

        if scheduler_type == "cosine_warmup":
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=num_epochs,
                cycle_mult=kwargs.get("cycle_mult", 1.0),
                max_lr=kwargs.get("max_lr", optimizer.defaults["lr"]),
                min_lr=eta_min,
                warmup_steps=warmup_epochs,
                gamma=kwargs.get("gamma", 1.0),
            )
            return {"scheduler": scheduler, "interval": "epoch"}

        if scheduler_type == "one_cycle":
            steps_per_epoch = kwargs.get("steps_per_epoch", 100)
            scheduler = OneCycleLRWithWarmup(
                optimizer,
                max_lr=kwargs.get("max_lr", optimizer.defaults["lr"]),
                total_steps=num_epochs * steps_per_epoch,
                warmup_steps=warmup_epochs * steps_per_epoch,
                min_lr=eta_min,
                pct_start=kwargs.get("pct_start", 0.3),
            )
            return {"scheduler": scheduler, "interval": "step"}

        if scheduler_type == "polynomial":
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=num_epochs,
                power=kwargs.get("power", 0.9),
            )
            return {"scheduler": scheduler, "interval": "epoch"}

        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", num_epochs // 3),
                gamma=kwargs.get("gamma", 0.1),
            )
            return {"scheduler": scheduler, "interval": "epoch"}

        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
