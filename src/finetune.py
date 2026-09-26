import torch
from datetime import datetime
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter

import models
import datasets
import utils
import backbones
import hooks

logger = logging.getLogger(__name__)
device = torch.device('mps')


def make_experiment_folder(train_config: dict): 
    # save train run in experiment subfolder
    now = datetime.now().strftime("%m-%d-%y--%H:%M:%S")
    experiment_folder = Path(train_config["target_folder"]) / ("linprobe" + now)
    Path.mkdir(experiment_folder)
    return experiment_folder


def dict_to_device(data_dict): 
    device_dict = {}
    for key, val in data_dict.items():
        device_dict[key] = val.to(device)

    return device_dict




class LinProber: 
    def __init__(
            self,
            cfg: dict,
            rest: models.REST,
            train_dataset: datasets.TacklePosDataset,
            val_dataset: datasets.TacklePosDataset,
            test_dataset: datasets.TacklePosDataset
    ): 
        self.cfg = cfg

        self.rest = rest
        self.rest.eval()
        embed_dimension = rest.cfg['encoder']['model_dimension']

        self.probe = torch.nn.Linear(embed_dimension, cfg['target_features'], device=device)

        # make experiment folder
        self.experiment_folder = make_experiment_folder(cfg)
        self.writer = SummaryWriter(self.experiment_folder)

        # build dataloaders
        self.train_dataloader = self.build_dataloader(train_dataset)
        self.val_dataloader = self.build_dataloader(val_dataset)
        self.test_dataloader = self.build_dataloader(test_dataset)
        # build optimizer
        self.optim = self.build_optimizer()

        self.loss_fn = torch.nn.SmoothL1Loss()

        self.num_epochs = cfg["epochs"]
        self.epoch = 0
        self.total_steps = len(self.train_dataloader) * self.num_epochs

        self.patience = cfg['patience']
        self.loss_queue = []

        # build schedulers
        # self.lr_scheduler = utils.CosineLRScheduler(self.optim, cfg['lr_base'], cfg['lr_final'], self.total_steps, warmup_iters=cfg['lr_warmup']) # TODO: give config[lr_schedule] = {...}
        self.lr = cfg['lr']
        return 
    
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

                stop = self.eval_epoch()
                self.run_hooks('after_epoch')

                self.epoch += 1

                if stop: 
                    break

        except KeyboardInterrupt: 
            print()
            logger.warning(f"Training interrupted after {self.epoch} steps")

        self.run_hooks('after_train')


    def train_epoch(self):
        total_loss = 0
        num_batches = len(self.train_dataloader)

        for step, minibatch in enumerate(self.train_dataloader):  
            self.run_hooks('before_train_step')

            self.optim.zero_grad()
            # get embedding from cls token as linear probing input
            minibatch = dict_to_device(minibatch)
            probe_in = self.rest.encode(minibatch, self.cfg['use_cls_token'], token_mask=minibatch['bc_mask'])
            pred = self.probe(probe_in)

            # calculate loss
            loss = self.loss_fn(pred, minibatch['target'])
            loss.backward()
            total_loss += loss.item()

            self.optim.step()

            self.run_hooks('after_train_step')


        self.mean_train_loss = total_loss / num_batches
        return


    def eval_epoch(self): 
        total_loss = 0
        num_batches = len(self.val_dataloader)
        de = torch.empty(0)

        with torch.no_grad():

            for step, val_minibatch in enumerate(self.val_dataloader):
                self.run_hooks('before_val_step')
                val_minibatch = dict_to_device(val_minibatch)
                probe_in = self.rest.encode(val_minibatch, cls_only=self.cfg['use_cls_token'], token_mask=val_minibatch['bc_mask'])
                pred = self.probe(probe_in)

                # calculate loss
                loss = self.loss_fn(pred, val_minibatch['target'])
                total_loss += loss.item()

                de = torch.cat((self.calculate_de(pred.detach().cpu(), val_minibatch['target'].detach().cpu()), de))

                self.run_hooks('after_val_step')

            ade = de.mean()
            print(f'Val ADE: {ade:.4f}')
            self.val_sample = val_minibatch['feat'][0]
            self.val_bc = val_minibatch['bc'][0]
            self.val_prediction = pred[0]
            self.val_target = val_minibatch['target'][0]

        if self.check_early_stopping(total_loss): 
            return True

        self.mean_val_loss = total_loss / num_batches
        return
    

    def test(self): 
        de = torch.empty(0)
        with torch.no_grad(): 
            for step, test_minibatch in enumerate(self.test_dataloader): 
                test_minibatch = dict_to_device(test_minibatch)
                probe_in = self.rest.encode(test_minibatch, self.cfg['use_cls_token'], token_mask=test_minibatch['bc_mask'])
                pred = self.probe(probe_in)

                de = torch.cat((self.calculate_de(pred.detach().cpu(), test_minibatch['target'].detach().cpu()), de))

            ade = de.mean()
            print(f"Test ADE: {ade}")

        return ade.item()


    @staticmethod
    def calculate_de(pred: torch.Tensor, target: torch.Tensor): 
        de = (((target - pred) ** 2).sum(axis=1) ** .5)
        return de
        

    def build_optimizer(self) -> torch.optim.Optimizer: 

        optim = torch.optim.AdamW(
            self.probe.parameters() # TODO: param groups (last layer lr)
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


    def check_early_stopping(self, total_val_loss: float) -> bool:
        self.loss_queue.append(total_val_loss)
        if len(self.loss_queue) < self.patience: 
            return False

        if self.loss_queue.pop(0) <= min(self.loss_queue):
            return True
        else: 
            return False






class AttnProbe(torch.nn.Module): 

    def __init__(
            self,
            model_dimension: int,
            out_dimension: int,
            heads: int,
            drop: float=.2
            
    ): 
        super().__init__()

        self.block = backbones.MyBlock(model_dimension, model_dimension*4, heads, drop, drop, drop)
        self.norm = torch.nn.LayerNorm(model_dimension)
        self.pooling = torch.nn.AdaptiveMaxPool1d(1)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(model_dimension, model_dimension),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop),
            torch.nn.Linear(model_dimension, model_dimension // 4),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(model_dimension // 4),
            torch.nn.Linear(model_dimension // 4, out_dimension)
        )

    def forward(self, coord, feat):
        feat = self.block(feat, coord)
        feat = self.norm(feat)

        pooled = torch.squeeze(self.pooling(feat.permute(0, 2, 1)), -1)
        out = self.decoder(pooled)

        return out





class AttnProber:
    def __init__(
            self,
            cfg: dict,
            rest: models.REST,
            train_dataset: datasets.TacklePosDataset,
            val_dataset: datasets.TacklePosDataset,
            test_dataset: datasets.TacklePosDataset,
    ): 
        self.cfg = cfg

        self.rest = rest
        self.rest.requires_grad_(False)
        self.rest.eval()
        embed_dimension = rest.cfg['encoder']['model_dimension']
        heads = rest.cfg['encoder']['heads']

        self.probe = AttnProbe(
            embed_dimension, cfg['target_features'], heads
        ).to(device)

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

        self.num_epochs = cfg["epochs"]
        self.epoch = 0
        self.total_steps = len(self.train_dataloader) * self.num_epochs

        # build schedulers
        # self.lr_scheduler = utils.CosineLRScheduler(self.optim, cfg['lr_base'], cfg['lr_final'], self.total_steps, warmup_iters=cfg['lr_warmup']) # TODO: give config[lr_schedule] = {...}
        self.lr = cfg['lr']
        return 
    
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
        self.probe.train()
        for step, minibatch in enumerate(self.train_dataloader):  
            self.run_hooks('before_train_step')

            self.optim.zero_grad()
            # get embedding from cls token as linear probing input
            minibatch = dict_to_device(minibatch)
            embeds = self.rest.encode(minibatch, cls_only=False)
            B = minibatch['feat'].shape[0]
            # ballcarrier_mask = torch.cat((torch.zeros(B, 1, 1), minibatch['bc_mask']), dim=1)
            # probe_in = torch.cat((embeds, ballcarrier_mask.to(device)), dim=-1)
            pred = self.probe(minibatch['coord'], embeds)

            # calculate loss
            loss = self.loss_fn(pred, minibatch['target'])
            loss.backward()
            total_loss += loss.item()

            self.optim.step()

            self.run_hooks('after_train_step')


        self.mean_train_loss = total_loss / num_batches
        return


    def eval_epoch(self): 
        total_loss = 0
        num_batches = len(self.val_dataloader)
        de = torch.empty(0)
        self.probe.eval()
        with torch.no_grad():

            for step, val_minibatch in enumerate(self.val_dataloader):
                self.run_hooks('before_val_step')
                val_minibatch = dict_to_device(val_minibatch)
                embeds = self.rest.encode(val_minibatch, cls_only=False)
                B = val_minibatch['feat'].shape[0]
                # ballcarrier_mask = torch.cat((torch.zeros(B, 1, 1), val_minibatch['bc_mask']), dim=1)
                # probe_in = torch.cat((embeds, ballcarrier_mask.to(device)), dim=-1)
                pred = self.probe(val_minibatch['coord'], embeds)

                # calculate loss
                loss = self.loss_fn(pred, val_minibatch['target'])
                total_loss += loss.item()

                de = torch.cat((self.calculate_de(pred.detach().cpu(), val_minibatch['target'].detach().cpu()), de))

                self.run_hooks('after_val_step')

            ade = de.mean()
            print(f'Val ADE: {ade:.4f}')
            self.val_sample = val_minibatch['feat'][0]
            self.val_bc = val_minibatch['bc'][0]
            self.val_prediction = pred[0]
            self.val_target = val_minibatch['target'][0]

        self.mean_val_loss = total_loss / num_batches
        return
    

    def test(self): 
        de = torch.empty(0)
        self.probe.eval()
        with torch.no_grad(): 
            for step, test_minibatch in enumerate(self.test_dataloader): 
                test_minibatch = dict_to_device(test_minibatch)
                embeds = self.rest.encode(test_minibatch, cls_only=False)
                B = test_minibatch['feat'].shape[0]
                # ballcarrier_mask = torch.cat((torch.zeros(B, 1, 1), test_minibatch['bc_mask']), dim=1)
                # probe_in = torch.cat((embeds, ballcarrier_mask.to(device)), dim=-1)
                pred = self.probe(test_minibatch['coord'], embeds)

                de = torch.cat((self.calculate_de(pred.detach().cpu(), test_minibatch['target'].detach().cpu()), de))

            ade = de.mean()
            print(f"Test ADE: {ade}")

        return ade.item()


    @staticmethod
    def calculate_de(pred: torch.Tensor, target: torch.Tensor): 
        de = (((target - pred) ** 2).sum(axis=1) ** .5)
        return de
        

    def build_optimizer(self) -> torch.optim.Optimizer: 
        optim = torch.optim.AdamW(
            self.probe.parameters() # TODO: param groups (last layer lr)
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

