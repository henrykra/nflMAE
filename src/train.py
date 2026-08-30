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
import time

import bdb2025
import bdb2024
import datasets
import backbones
import utils
import hooks

logger = logging.getLogger(__name__)
device = torch.device('mps')


def make_experiment_folder(train_config: dict): 
    # save train run in experiment subfolder
    now = datetime.now().strftime("%m-%d-%y--%H:%M:%S")
    experiment_folder = Path(train_config["target_folder"]) / ("experiment" + now)
    Path.mkdir(experiment_folder)
    return experiment_folder


def dict_to_device(data_dict): 
    device_dict = {}
    for key, val in data_dict.items():
        device_dict[key] = val.to(device)

    return device_dict



class MAETrainer: 

    def __init__(
            self,
            cfg: dict,
            model: torch.nn.Module,
            train_dataset: datasets.FrameDataset,
            val_dataset: datasets.FrameDataset,

    ): 
        self.cfg = cfg
        
        self.model = model

        # make experiment folder
        self.experiment_folder = make_experiment_folder(cfg)
        self.writer = SummaryWriter(self.experiment_folder)

        # build dataloaders
        self.train_dataloader = self.build_dataloader(train_dataset) # TODO new collate function
        self.val_dataloader = self.build_dataloader(val_dataset)
        # build optimizer
        self.optim = self.build_optimizer(self.model)

        self.loss_fn = torch.nn.MSELoss()

        self.num_epochs = cfg["epochs"]
        self.epoch = 0
        self.total_steps = len(self.train_dataloader) * self.num_epochs

        # build schedulers
        self.lr_scheduler = utils.CosineLRScheduler(self.optim, cfg['lr_base'], cfg['lr_final'], self.total_steps, warmup_iters=cfg['lr_warmup']) # TODO: give config[lr_schedule] = {...}
        # TODO: weight decay schedule
        # TODO: mask schedule


    def build_dataloader(self, dataset: torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True
        )
        return dataloader


    def train(
            self, 
    ): 
        self.run_hooks('before_train')

        # TODO: support resume training
        try: 
            while self.epoch < self.num_epochs: 

                self.run_hooks('before_epoch')
                self.train_epoch()

                self.eval_epoch()
                self.run_hooks('after_epoch')

                self.epoch += 1

        except KeyboardInterrupt: 
            print()
            logger.warning(f"Training interrupted after {self.epoch} steps")

        self.run_hooks('after_train')



    def train_epoch(self):
        total_loss = 0
        num_batches = len(self.train_dataloader)

        self.model.train()
        for step, minibatch in enumerate(self.train_dataloader): 
            self.run_hooks('before_train_step')
            
            self.optim.zero_grad()
            minibatch = dict_to_device(minibatch)

            self.model.generate_mask(minibatch)
            pred = self.model(minibatch)

            # calculate loss
            # loss for only masked tokens!!!
            loss = self.loss_fn(pred[minibatch['mask']], minibatch['feat'][minibatch['mask']])

            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) 

            self.optim.step()
            self.lr_scheduler.step()

            self.run_hooks('after_train_step')


        self.mean_train_loss = total_loss / num_batches
        return


    def eval_epoch(self): 
        total_loss = 0
        num_batches = len(self.val_dataloader)

        self.model.eval()
        with torch.no_grad():

            for step, val_minibatch in enumerate(self.val_dataloader):
                self.run_hooks('before_val_step')
                val_minibatch = dict_to_device(val_minibatch)
                self.model.generate_mask(val_minibatch)
                pred = self.model(val_minibatch)

                loss = self.loss_fn(pred[val_minibatch['mask']], val_minibatch['feat'][val_minibatch['mask']])
                total_loss += loss.item()
                self.run_hooks('after_val_step')

            self.val_sample = val_minibatch['feat'][0] 
            self.val_prediction = pred[0]
            self.val_mask = val_minibatch['mask'][0]


        self.mean_val_loss = total_loss / num_batches
        return
    
        

        

    def build_optimizer(self, model) -> torch.optim.Optimizer: 

        optim = torch.optim.AdamW(
            self.model.parameters() # TODO: param groups (last layer lr)
        )
        return optim


    def register_hooks(self, hooks: list[hooks.Hook]): 
        hook_names = [
            'before_train', 'before_epoch',
            'before_train_step', 'after_train_step', 
            'before_val_step', 'after_val_step',
            'after_epoch', 'after_train'
        ]
        for name in hook_names: 
            setattr(self, name, [])


        for h in hooks: 
            for name in hook_names: 
                if hasattr(h, name): 
                    getattr(self, name).append(getattr(h, name))

        return 


    def run_hooks(self, type: str):
        hook_fns = getattr(self, type)

        for fn in hook_fns: 
            fn(self)
        return



