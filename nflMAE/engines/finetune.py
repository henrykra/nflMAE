from datetime import datetime
from pathlib import Path
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from nflMAE.models import models
from nflMAE.models import backbones
from nflMAE.utils import hooks

logger = logging.getLogger(__name__)
device = torch.device('mps')


def make_experiment_folder(train_config: dict, folder_name: str=None): 
    # save train run in experiment subfolder
    if folder_name is None:
        now = datetime.now().strftime("%m-%d-%y--%H:%M:%S")
        experiment_folder = Path(train_config['target_folder'], ("finetune" + now))
    else: 
        experiment_folder = Path(train_config['target_folder'], folder_name)
    Path.mkdir(experiment_folder)
    return experiment_folder


def dict_to_device(data_dict: dict[str, torch.Tensor]): 
    device_dict = {}
    for key, val in data_dict.items():
        device_dict[key] = val.to(device)

    return device_dict


class Finetuner: 
    def __init__(
        self,
        cfg: dict,
        backbone: models.NflMaeModel,
        probe: torch.nn.Module,
        train_dataset,
        val_dataset,
        test_dataset,
    ):
        self.cfg = cfg
        self.model = backbone
        self.probe = probe
        self.dataset_type = type(train_dataset)

        # make experiment folder
        self.experiment_folder = make_experiment_folder(cfg)
        self.writer = SummaryWriter(self.experiment_folder)

        # build dataloaders
        self.train_dataloader = self.build_dataloader(train_dataset)
        self.val_dataloader = self.build_dataloader(val_dataset)
        self.test_dataloader = self.build_dataloader(test_dataset)
        # build optimizer
        self.optim = self.build_optimizer()

        self.loss_fn = train_dataset.get_loss_fn()

        self.patience = cfg['patience']
        self.loss_queue = []

        self.epoch = 0
        self.num_epochs = cfg['epochs']

        return 

    def build_dataloader(self, dataset: torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True
        )
        return dataloader

    def build_optimizer(self) -> torch.optim.Optimizer: 
        optim = torch.optim.AdamW([
            {"params": self.model.parameters(), "lr": self.cfg['lr']},
            {"params": self.probe.parameters(), "lr": self.cfg['lr']},
        ])
        return optim

    def train(
                self, 
        ): 
        self.run_hooks("before_train")
        # TODO: support resume training
        try: 
            while self.epoch < self.num_epochs: 

                self.run_hooks("before_epoch")
                self.train_epoch()

                stop = self.eval_epoch()
                self.run_hooks("after_epoch")

                self.epoch += 1

                if stop: 
                    break

        except KeyboardInterrupt: 
            print()
            logger.warning(f"Training interrupted after {self.epoch} steps")

        self.run_hooks("after_train")

    def train_epoch(self):
        total_loss = 0
        num_batches = len(self.train_dataloader)
        self.model.train()
        self.probe.train()
        for step, minibatch in enumerate(self.train_dataloader):  
            self.run_hooks("before_train_step")

            self.optim.zero_grad()
            # get embedding from cls token as linear probing input
            minibatch = dict_to_device(minibatch)
            probe_in = self.model.encode(minibatch, self.cfg['use_cls_token'], token_mask=minibatch.get('target_mask'))
            pred = self.probe(probe_in)

            # calculate loss
            loss = self.loss_fn(pred, minibatch)
            loss.backward()
            total_loss += loss.item()

            self.optim.step()

            self.run_hooks("after_train_step")


        self.mean_train_loss = total_loss / num_batches
        return


    def eval_epoch(self): 
        total_loss = 0
        evals = torch.empty(0)
        num_batches = len(self.val_dataloader)
        self.model.eval()
        self.probe.eval()
        with torch.no_grad():

            for step, val_minibatch in enumerate(self.val_dataloader):
                self.run_hooks("before_val_step")
                val_minibatch = dict_to_device(val_minibatch)
                probe_in = self.model.encode(val_minibatch, cls_only=self.cfg['use_cls_token'], token_mask=val_minibatch.get('target_mask'))
                pred = self.probe(probe_in)

                # calculate loss
                loss = self.loss_fn(pred, val_minibatch)
                total_loss += loss.item()

                stat = self.calculate_eval_statistic(pred, val_minibatch)
                evals = torch.cat( (evals, stat) )

                self.run_hooks("after_val_step")

            stat = evals.mean()
            self.stat = stat
        
            self.save_sample(val_minibatch, pred)

        if self.check_early_stopping(total_loss): 
            return True
        
        self.mean_val_loss = total_loss / num_batches
        return False


    def test(self): 
        evals = torch.empty(0)
        self.model.eval()
        self.probe.eval()
        with torch.no_grad(): 
            for step, test_minibatch in enumerate(self.test_dataloader): 
                test_minibatch = dict_to_device(test_minibatch)
                probe_in = self.model.encode(test_minibatch, self.cfg['use_cls_token'], token_mask=test_minibatch.get('target_mask'))
                pred = self.probe(probe_in)

                stat = self.calculate_eval_statistic(pred.detach().cpu(), test_minibatch)
                evals = torch.cat( (evals, stat) )


            stat = evals.mean()

        return stat.item()



    def register_hooks(self, hooks: list[hooks.Hook]): 
        hook_names = [
            "before_train", "before_epoch",
            "before_train_step", "after_train_step", 
            "before_val_step", "after_val_step",
            "after_epoch", "after_train"
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


    def calculate_eval_statistic(self, pred: torch.Tensor, minibatch):
        return self.dataset_type.eval_statistic(pred, minibatch)


    def check_early_stopping(self, total_val_loss: float) -> bool:
        self.loss_queue.append(total_val_loss)
        if len(self.loss_queue) < self.patience: 
            return False

        if self.loss_queue.pop(0) <= min(self.loss_queue):
            return True
        else: 
            return False


    def save_sample(self, minibatch: dict, pred):
        self.val_sample = minibatch.get('feat')[0]
        if "bc" in minibatch:
            self.val_bc = minibatch.get('bc')[0]
        self.val_prediction = pred[0]
        if "target" in minibatch:
            self.val_target = minibatch.get('target')[0]
        return 

    def clear(self):
        self.num_epochs = self.epoch + self.cfg['epochs']
        self.optim = self.build_optimizer()
        self.loss_queue = []


def build_linear_probe(backbone, target_features):
    probe = torch.nn.Linear(backbone.dimension(), target_features, device=device)
    return probe


def build_attn_probe(cfg, backbone):
    d = backbone.dimension()
    h = backbone.cfg['encoder']['heads']

    probe = torch.nn.Sequential(
        backbones.AttnPoolingLayer(d, h),
        torch.nn.LayerNorm(d),
        torch.nn.Linear(d, d),
        torch.nn.ReLU(),
        torch.nn.Dropout(.3),
        torch.nn.Linear(d, d // 4),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(d // 4),
        torch.nn.Linear(d // 4, cfg['target_features'])
    )

    return probe


def build_pooling_probe(cfg, backbone):
    d = backbone.dimension()
    h = backbone.cfg['encoder']['heads']

    probe = torch.nn.Sequential(
        backbones.MaxPoolingLayer(d),
        torch.nn.LayerNorm(d),
        torch.nn.Linear(d, d),
        torch.nn.ReLU(),
        torch.nn.Dropout(.3),
        torch.nn.Linear(d, d // 4),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(d // 4),
        torch.nn.Linear(d // 4, cfg['target_features'])
    )
    return probe
