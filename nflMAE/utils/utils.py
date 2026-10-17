from pathlib import Path
import hashlib

import torch
import numpy as np

ARTIFACT_DIR = Path("/Users/henrykraessig/code/route2vec/data")


class Artifact:

    def __init__(self, name: str, cfg: dict, out_path: Path):
        self.name = name
        self.cfg = cfg
        self.out_path = out_path

        self.artifact_path = ARTIFACT_DIR / name
        if not self.artifact_path.exists(): 
            Path.mkdir(self.artifact_path)


    def __enter__(self): 
        m = hashlib.sha256()

        hash_path = self.artifact_path / "name-cfg.sha256"
        if not hash_path.exists(): 
            with hash_path.open('wb') as hash_file:
                m.update(str(self.cfg))
                hash_file.write(m.digest())


    def __exit__(self): 
        ...


class CosineLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_value,
        final_value,
        total_iters,
        start_value=0,
        warmup_iters=0,
        freeze_value=None,
        freeze_iters=0,
    ):
        self.iter = 0
        super().__init__(optimizer) # TODO: checkpointing

        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters

        warmup_schedule = np.linspace(start_value, base_value, warmup_iters)

        if freeze_value is None:
            freeze_value = final_value
        freeze_schedule = np.ones(freeze_iters) * freeze_value

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((warmup_schedule, schedule, freeze_schedule))


    # TODO: define param groups
    def get_lr(self): 
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(self.get(self.iter))
        

    def get(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

    def step(self):        
        self.iter += 1


    def reset(self):
        self.iter = 0

    def __getitem__(self, it):
        return self.get(it)