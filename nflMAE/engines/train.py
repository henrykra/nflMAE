import logging
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from nflMAE.data import datasets
from nflMAE.utils import utils
from nflMAE.utils import hooks

logger = logging.getLogger(__name__)
device = torch.device('mps')


def make_experiment_folder(train_config: dict, folder_name: str=None): 
    # save train run in experiment subfolder
    if folder_name is None:
        now = datetime.now().strftime("%m-%d-%y--%H:%M:%S")
        experiment_folder = Path(train_config['target_folder'], ("pretrain" + now))
    else: 
        experiment_folder = Path(train_config['target_folder'], folder_name)
    Path.mkdir(experiment_folder, exist_ok=True)
    return experiment_folder


def dict_to_device(data_dict): 
    device_dict = {}
    for key, val in data_dict.items():
        device_dict[key] = val.to(device)

    return device_dict



class MaskedNflLoss(torch.nn.Module):
    """Weighted loss for pretraining MAE task"""
    def __init__(self): 
        super().__init__()
        self.position_idx = slice(0, 2)
        self.position_weight = 1

        self.orientation_idx = slice(2, 4)
        self.orientation_weight = .5

        self.velocity_idx = slice(4, 7)
        self.velocity_weight = .5

        self.side_idx = slice(7, 9)
        self.side_weight = .5

        self.metrics_idx = slice(9, 11)
        self.metrics_weight = .1

        self.mse_loss = torch.nn.SmoothL1Loss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    
    def forward(self, pred, target): 

        total_loss = (
            self.position_weight * self.mse_loss(pred[..., self.position_idx].contiguous(), target[..., self.position_idx].contiguous()) + 
            self.orientation_weight * self.mse_loss(pred[..., self.orientation_idx].contiguous(), target[..., self.orientation_idx].contiguous()) + 
            self.velocity_weight * self.mse_loss(pred[..., self.velocity_idx].contiguous(), target[..., self.velocity_idx].contiguous()) + 
            self.side_weight * self.bce_loss(pred[..., self.side_idx].contiguous(), target[..., self.side_idx].contiguous()) + 
            self.metrics_weight * self.mse_loss(pred[..., self.metrics_idx].contiguous(), target[..., self.metrics_idx].contiguous())
        )

        return total_loss
    


class MAETrainer: 

    def __init__(
            self,
            cfg: dict,
            model: torch.nn.Module,
            train_dataset: datasets.FrameDataset,
            val_dataset: datasets.FrameDataset,
            folder_name: str=None,

    ): 
        self.cfg = cfg
        
        self.model = model

        # make experiment folder
        self.experiment_folder = make_experiment_folder(cfg, folder_name)
        self.writer = SummaryWriter(self.experiment_folder)

        # build dataloaders
        self.train_dataloader = self.build_dataloader(train_dataset) # TODO new collate function
        self.val_dataloader = self.build_dataloader(val_dataset)
        # build optimizer
        self.optim = self.build_optimizer(self.model)

        self.loss_fn = MaskedNflLoss()

        self.num_epochs = cfg['epochs']
        self.epoch = 0
        self.total_steps = len(self.train_dataloader) * self.num_epochs

        # build schedulers
        self.lr_scheduler = utils.CosineLRScheduler(self.optim, cfg['lr_base'], cfg['lr_final'], self.total_steps, warmup_iters=cfg['lr_warmup']) # TODO: give config[lr_schedule] = {...}
        # TODO: weight decay schedule

        self.mask_scheduler = utils.MaskRatioScheduler(cfg['mask_ratio_base'], cfg['mask_ratio_final'], self.total_steps)
        return

    
    @torch.no_grad()
    def generate_mask(self, data_dict: dict):
        """Create mask for MAE"""

        # if coordinates are present, determine grid centroid for noisy masking
        if "coord" in data_dict:
            min_coord, _ = data_dict['coord'].min(dim=1)
            # find patch num for each dimension
            grid_coord = ((data_dict['coord'] - min_coord.unsqueeze(1)) // self.cfg['grid_size']).int()
            # grab all points in the same patch
            unique, cluster = torch.unique(grid_coord.cpu(), dim=1, sorted=True, return_inverse=True)
            cluster = cluster.to("mps")
            data_dict['centroid'] = grid_coord * self.cfg['grid_size'] + self.cfg['grid_size'] / 2

        #assert not (self.cfg['mask_regions'] and not self.cfg['encode_masks'])# not ready yet, can't handle uneven masking

        if self.cfg['mask_players']:
            N, P, D = data_dict['feat'].shape  
            n_maskable = 22
            n_maskable = min(n_maskable, P)
            # assert P == 23 -> noise everything but the football

            len_keep = int(n_maskable * (1 - self.mask_scheduler.get()))
            
            noise = torch.rand(N, n_maskable, device='mps')  # noise in [0, 1], 
            
            # sort noise for each sample
            # = random order of indices - truncate and set those be non-masks
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]

            mask = torch.ones([N, P], device='mps')
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            # mask.scatter_(1, ids_keep, 0)

            # only optionally mask the last n_maskable players
            if P - n_maskable > 0:
                mask = torch.cat( (torch.zeros(N, P-n_maskable, device='mps'), mask), dim=-1)

            data_dict['mask'] = mask.bool()


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
        self.run_hooks("before_train")

        # TODO: support resume training
        try: 
            while self.epoch < self.num_epochs: 

                self.run_hooks("before_epoch")
                self.train_epoch()

                self.eval_epoch()
                self.run_hooks("after_epoch")

                self.epoch += 1

        except KeyboardInterrupt: 
            print()
            logger.warning(f"Training interrupted after {self.epoch} steps")

        self.run_hooks("after_train")


    def train_epoch(self):
        total_loss = 0
        num_batches = len(self.train_dataloader)

        self.model.train()
        for step, minibatch in enumerate(self.train_dataloader): 
            self.run_hooks("before_train_step")
            bs, n, c = minibatch['feat'].shape
            
            self.optim.zero_grad()
            minibatch = dict_to_device(minibatch)

            self.generate_mask(minibatch)
            self.mask_scheduler.step()

            pred = self.model(minibatch)

            # calculate loss
            # loss for only masked tokens!!!
            loss = self.loss_fn(pred, minibatch['feat'][minibatch['mask']].view(bs, -1, c))

            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) 

            self.optim.step()
            self.lr_scheduler.step()

            self.run_hooks("after_train_step")


        self.mean_train_loss = total_loss / num_batches
        return


    def eval_epoch(self): 
        total_loss = 0
        num_batches = len(self.val_dataloader)

        self.model.eval()
        with torch.no_grad():

            for step, val_minibatch in enumerate(self.val_dataloader):
                self.run_hooks("before_val_step")
                bs, n, c = val_minibatch['feat'].shape
                val_minibatch = dict_to_device(val_minibatch)
                self.generate_mask(val_minibatch)
                pred = self.model(val_minibatch)

                loss = self.loss_fn(pred, val_minibatch['feat'][val_minibatch['mask']].view(bs, -1, c))
                total_loss += loss.item()
                self.run_hooks("after_val_step")

            self.val_sample = val_minibatch['feat'][0] 
            self.val_prediction = pred[0]
            self.val_mask = val_minibatch['mask'][0]
            self.reference_frame = val_minibatch.get('reference_feat')


        self.mean_val_loss = total_loss / num_batches
        return
    

    def build_optimizer(self, model) -> torch.optim.Optimizer: 
        sid_params = []
        params = []

        for block in self.model.encoder.blocks:
            sid_params.append(block.attn_layer.sid_value)

        params = [p for p in self.model.parameters() if not any(p is s for s in sid_params)]


        optim = torch.optim.AdamW(
            [
                {"params": sid_params, "weight_decay": 0.02},
                {"params": params, "weight_decay": 0.02}
            ]
        )
        return optim


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
