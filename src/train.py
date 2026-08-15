from pathlib import Path
import json
import pandas as pd
import logging
import torch
from PIL import Image
import io
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
import time
import numpy as np

import bdb2025
import bdb2024
import datasets
import src.backbones as backbones
import utils
import hooks

logger = logging.getLogger(__name__)
device = torch.device('mps')


def parse_config():
    with open("./config.json", "r") as configfile:
        config = json.load(configfile)
    return config


def make_experiment_folder(train_config: dict): 
    # save train run in experiment subfolder
    now = datetime.now().strftime("%m-%d-%y--%H:%M:%S")
    experiment_folder = Path(train_config["target_folder"]) / ("experiment" + now)
    Path.mkdir(experiment_folder)
    return experiment_folder


def train(
        model: torch.nn.Module, 
        train_set: torch.utils.data.Dataset, 
        val_set: torch.utils.data.Dataset, 
        test_set: torch.utils.data.Dataset, 
        train_config: dict,
        experiment_folder: Path
):

    # start tensorboard
    writer = SummaryWriter(experiment_folder)

    writer.add_custom_scalars({
        'Activations': {'block_activations': ['Multiline', ['Blocks/Act/norm_0', 'Blocks/Act/norm_1', 'Blocks/Act/norm_2', 'Blocks/Act/norm_3']]},
        'Gradients': {'block_gradients': ['Multiline', ['Blocks/Grad/norm_0', 'Blocks/Grad/norm_1', 'Blocks/Grad/norm_2', 'Blocks/Grad/norm_3']]}
    })

    train_loader = torch.utils.data.DataLoader(train_set, train_config["batch_size"], collate_fn=datasets.Frame.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, train_config["batch_size"], collate_fn=datasets.Frame.collate_fn, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, train_config["batch_size"], collate_fn=datasets.Frame.collate_fn)
    
    optim = torch.optim.AdamW(
        params=model.parameters(), 
        lr=train_config["lr"],

    )

    loss_fn = torch.nn.MSELoss(reduction='mean')

    try:
        for epoch in range(train_config["epochs"]):
            grad_norms = {}
            act_norms = {}
            handles = []
            total_grad_norm = None
            if epoch % train_config['log_every'] == 0: 
                grad_norms, act_norms, handles = utils.register_hooks(model)

                total_grad_norm = utils.total_grad_norm(model)
            
            train_loss = train_epoch(train_loader, model, optim, loss_fn)
            writer.add_scalar('Loss/train', train_loss, epoch)

            for i, norm in enumerate(grad_norms.values()): 
                writer.add_scalar(f'Blocks/Grad/norm_{i}', norm, epoch)

            for i, norm in enumerate(act_norms.values()): 
                writer.add_scalar(f'Blocks/Act/norm_{i}', norm, epoch)

            if total_grad_norm is not None: 
                writer.add_scalar(f'Grad/norm', total_grad_norm, epoch)


            for handle in handles: 
                handle.remove()
            
            val_loss = val_epoch(val_loader, model, loss_fn)
            writer.add_scalar('Loss/val', val_loss, epoch)

            if train_config['log_every'] < 1: 
                continue    

            if epoch % train_config['log_every'] == 0: 
                with torch.no_grad(): 
                    idx = np.random.randint(0, len(val_set) - 1)
                    input_sample = val_set[idx].feats.cpu()
                    target_sample = val_set[idx].target.cpu()
                    
                    pred_sample = datasets.Frame(val_set[idx].feats.unsqueeze(0).to(device), val_set[idx].coords.unsqueeze(0).to(device))
                    preds = model(pred_sample)[0].cpu()
                    fig = utils.plot_frame_tensor(input_sample, target_sample, preds)
                    img_bytes = fig.to_image(format='jpg')
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    writer.add_image(f"sample/{epoch}", np.array(img).transpose([2, 0, 1]))

                


            
    except KeyboardInterrupt: 
        print()
        logger.warning(f"Training interrupted after {epoch} steps")


    # test model
    # norms, stds = utils.register_hooks(model)
    test_ade = test_model(test_loader, model)

    if False:
        fig = utils.plot_activation_stats(norms, stds)
        img_bytes = fig.to_image(format='jpg')
        img = Image.open(io.BytesIO(img_bytes))
        img.save(Path(experiment_folder, 'layers.jpg'))

    writer.add_hparams({'lr': train_config['lr'],
                        'bsize': train_config['batch_size']},
                        {'ade': test_ade},
                        run_name=experiment_folder)

    writer.flush()
    writer.close()

    return test_ade





def train_epoch(
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        loss_fn: torch.nn.Module
) -> float: 
    
    model.train()
    train_loss = 0
    c = 0

    for frame in loader:
        optim.zero_grad()
        frame.to(device)

        out = model(frame)

        loss = loss_fn(out, frame.target)

        if not torch.isfinite(loss): 
            logger.warning("Infinte loss, skipping batch")
            continue

        train_loss += loss.item()
        c += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optim.step()

    return train_loss / c
        

def val_epoch(
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module
) -> float: 
    model.eval()
    val_loss = 0
    de = torch.empty(0)
    c = 0
    with torch.no_grad():

        for frame in loader:
            frame.to(device)

            out = model(frame)

            loss = loss_fn(out, frame.target)
            val_loss += loss.item()
            c += 1

            de = torch.cat((calculate_de(frame.target, out), de))
        ade = de.mean()            
        logger.info(f"Val ADE: {ade:.2f}")

        return val_loss / c
    


def test_model(
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
) -> float: 
    model.eval()

    de = torch.empty(0)
    
    with torch.no_grad():

        for frame in loader: 

            frame.to(device)

            out = model(frame)

            de = torch.cat((calculate_de(frame.target, out), de))

        ade = de.mean()

        print(f"Test ADE: {ade:.2f}")

        return ade.item()
    

def calculate_de(target: torch.Tensor, pred: torch.Tensor): 
    de = (((target - pred) ** 2).sum(axis=1) ** .5)


    return de.cpu()






class CosineScheduler(torch.optim.lr_scheduler.LRScheduler):
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
        if optimizer: 
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
        self.iter = 0


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





class SSLTrainer: 

    def __init__(
            self,
            cfg: dict,
            model: torch.nn.Module,
            hooks: hooks.Hooks,
            dataset: datasets.FrameDataset,


    ): 
        self.cfg = cfg
        
        self.model = model

        # build dataloaders
        self.train_dataloader = self.build_dataloader(dataset) # TODO new collate function
        # build optimizer
        self.optim = self.build_optimizer()

        self.num_epochs = cfg["n_epochs"]
        self.total_steps = len(self.train_dataloader) * self.num_epochs

        # build schedulers
        self.lr_scheduler = CosineScheduler(self.optim,) # TODO: give config[lr_schedule] = {...}
        # TODO: weight decay schedule
        self.momentum_scheduler = CosineScheduler()
        self.temperature_scheduler = CosineScheduler()
        self.mask_scheduler = CosineScheduler()
        # TODO: mask size vs mask ratio vs both?
        
        # register hooks
        self.hooks = hooks


        


    def build_dataloader(self, dataset: torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"]
            
        )
        return dataloader


    def train(
            self, 
    ): 
        self.hooks.before_train()

        # TODO: support resume training
        for epoch in range(self.num_epochs):

            self.hooks.before_epoch(epoch)
            self.train_epoch()

            self.hooks.after_epoch(epoch)

        self.hooks.after_train()



    def train_epoch(self): 
        self.model.train()
        for step, minibatch in enumerate(self.train_dataloader): 
            self.hooks.before_step(step)
            momentum = self.momentum_scheduler.get()
            temperature = self.temperature_scheduler.get()
            mask_ratio = self.mask_scheduler.get()

            step_dict = {"momentum": momentum, "temperature": temperature, "mask_ratio": mask_ratio}

            self.optim.zero_grad()
            losses = self.model(minibatch, step_dict)

            self.model.student_backprop(losses)
            torch.nn.utils.clip_grad_norm_() # just student parameters
            self.optim.step()

            self.model.update_teacher()

            self.momentum_scheduler.step()
            self.temperature_scheduler.step()
            self.mask_scheduler.step()
            self.lr_scheduler.step()

            self.hooks.after_step(step)





    def build_optimizer(self, model) -> torch.optim.Optimizer: 

        optim = torch.optim.AdamW(
            self.model.parameters() # TODO: param groups (last layer lr)
        )
        return optim











