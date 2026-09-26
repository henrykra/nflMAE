import matplotlib.pyplot as plt
import logging
import torch
import numpy as np
from pathlib import Path
import yaml
import time
import torch

logger = logging.getLogger(__name__)


class Hook: 
    def __init__(self, every: int = 1): 
        if every < 1: 
            logger.warning(f"Invalid step value: {every}, logging every epoch")
            every = 1
        self.every = every
        return


class PlotFrame(Hook):
    color_map = {0: "#7f7f7f", 1: "#1f77b4", 2: "#d62728", 3: "#7f7f7f"}
    label_map = {0: "Ball", 1: "Offense", 2: "Defense"}

    def after_epoch(self, trainer):
        # plot a frame from a validation sample for the MAE training task
        # add the frame to tensorboard images
        assert hasattr(trainer, 'val_sample') and hasattr(trainer, 'val_prediction') and hasattr(trainer, 'val_mask')

        if trainer.epoch % self.every != 0: 
            return
        
        unmask_input = self._to_numpy(trainer.val_sample)

        mask = self._to_numpy(trainer.val_mask)
        pred = self._to_numpy(trainer.val_prediction)

        x, y = unmask_input[:, 0], unmask_input[:, 1]
        ox, oy = unmask_input[:, 2], unmask_input[:, 3]
        offense, defense = unmask_input[:, 7], unmask_input[:, 8]
        club = (defense * 2 + offense).astype(int)

        fig, ax = plt.subplots(figsize=(6, 5))


        for c in np.unique(club): 
            club_mask = club == c 
            ax.scatter(
                x[club_mask], y[club_mask],
                c=self.color_map[c], 
                label=self.label_map[c],
                alpha=~mask[club_mask] * .7 + .3,
                s=60, edgecolors='black', linewidths=.5, zorder=3,
            )

        x_pred, y_pred = pred[:, 0], pred[:, 1]
        ax.scatter(
            x_pred, y_pred,
            c="#10ab0e", 
            label="prediction",
            alpha=.7,
            s=60, edgecolors='black', linewidths=.5, zorder=3,
        )


        # arrows
        for xi, yi, oxi, oyi, ci in zip(x, y, ox, oy, club):
            if ci == 0: # skip ball
                continue
            ax.annotate(
                '', xy=(xi + 1.5 + oxi, yi + 1.5 * oyi), xytext=(xi, yi),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                zorder=2
            )

        # keep aspect ratio fixed and field bounds constant
        ax.set_xlim(-10, 40)
        ax.set_ylim(-53.3 / 2, 53.3 / 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)

        fig.tight_layout()

        # save figure to tensorboard
        trainer.writer.add_figure('Samples', fig, trainer.epoch)
        trainer.writer.flush()
        return 
        

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

# fix now that reference frame is separate
class Plot2Frames(PlotFrame):
    def after_epoch(self, trainer): 
        fig, axes = plt.subplots(figsize=(10, 5), ncols=2, sharex=True, sharey=True)
        n_players = 22 + 1


        for i in range(2):
            # plot a frame from a validation sample for the MAE training task
            # add the frame to tensorboard images
            assert hasattr(trainer, 'val_sample') and hasattr(trainer, 'val_prediction') and hasattr(trainer, 'val_mask')
    
            if trainer.epoch % self.every != 0: 
                return
            
            unmask_input = self._to_numpy(trainer.val_sample)
    
            frame_start = i * n_players
            frame_end = (i+1) * n_players

            mask = self._to_numpy(trainer.val_mask)[frame_start:frame_end]
            pred = self._to_numpy(trainer.val_prediction)[frame_start:frame_end]
    
            x, y = unmask_input[frame_start:frame_end, 0], unmask_input[frame_start:frame_end, 1]
            ox, oy = unmask_input[frame_start:frame_end, 2], unmask_input[frame_start:frame_end, 3]
            offense, defense = unmask_input[frame_start:frame_end, 7], unmask_input[frame_start:frame_end, 8]
            club = (defense * 2 + offense).astype(int)
    
    
    
            for c in np.unique(club): 
                club_mask = (club == c)
                axes[i].scatter(
                    x[club_mask], y[club_mask],
                    c=self.color_map[c], 
                    label=self.label_map[c],
                    alpha=~mask[club_mask] * .7 + .3,
                    s=60, edgecolors='black', linewidths=.5, zorder=3,
                )
    
            x_pred, y_pred = pred[:, 0], pred[:, 1]
            axes[i].scatter(
                x_pred, y_pred,
                c="#10ab0e", 
                label="prediction",
                alpha=.7,
                s=60, edgecolors='black', linewidths=.5, zorder=3,
            )
    
    
            # arrows
            for xi, yi, oxi, oyi, ci in zip(x, y, ox, oy, club):
                if ci == 0: # skip ball
                    continue
                axes[i].annotate(
                    '', xy=(xi + 1.5 + oxi, yi + 1.5 * oyi), xytext=(xi, yi),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                    zorder=2
                )
    
            # keep aspect ratio fixed and field bounds constant
            axes[i].set_xlim(-10, 40)
            axes[i].set_ylim(-53.3 / 2, 53.3 / 2)
            axes[i].set_aspect("equal", adjustable="box")
            axes[i].set_xlabel("x")
            axes[i].set_ylabel("y")
            axes[i].legend(loc="upper right", fontsize=8)
            axes[i].grid(True)
    
        fig.tight_layout()
        # save figure to tensorboard
        trainer.writer.add_figure('Samples', fig, trainer.epoch)
        trainer.writer.flush()
        return 


class PlotTackle(PlotFrame): 
    
    def after_epoch(self, trainer):
        # plot a frame from a validation sample for the MAE training task
        # add the frame to tensorboard images
        assert hasattr(trainer, 'val_sample') and hasattr(trainer, 'val_prediction') and hasattr(trainer, 'val_target') and hasattr(trainer, 'val_bc')

        if trainer.epoch % self.every != 0: 
            return
        
        frame_input = self._to_numpy(trainer.val_sample)
        target = self._to_numpy(trainer.val_target)

        pred = self._to_numpy(trainer.val_prediction)

        x, y = frame_input[:, 0], frame_input[:, 1]
        ox, oy = frame_input[:, 2], frame_input[:, 3]
        offense, defense = frame_input[:, 7], frame_input[:, 8]
        club = (defense * 2 + offense).astype(int)

        fig, ax = plt.subplots(figsize=(8, 5))


        for c in np.unique(club): 
            club_mask = club == c 
            ax.scatter(
                x[club_mask], y[club_mask],
                c=self.color_map[c], 
                label=self.label_map[c],
                alpha=.7,
                s=60, edgecolors='black', linewidths=.5, zorder=3,
            )

        # arrows
        for xi, yi, oxi, oyi, ci in zip(x, y, ox, oy, club):
            if ci == 0: # skip ball
                continue
            ax.annotate(
                '', xy=(xi + 1.5 + oxi, yi + 1.5 * oyi), xytext=(xi, yi),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                zorder=2
            )

        # tackle location
        ax.scatter(
            x=target[0], y=target[1],
            c="#10AA10",
            marker="X",
            label="Tackle",
            edgecolors='black', linewidths=.5, zorder=4
        )

        # prediction
        ax.scatter(
            x=pred[0], y=pred[1],
            c="#ECF541",
            marker="X",
            label="Tackle",
            edgecolors='black', linewidths=.5, zorder=4
        )


        # keep aspect ratio fixed and field bounds constant
        ax.set_xlim(-10, 30)
        ax.set_ylim(-53.3 / 2, 53.3 / 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout()

        # save figure to tensorboard
        trainer.writer.add_figure('Samples', fig, trainer.epoch)
        return 




class AddTrainLoss(Hook): 


    def after_epoch(self, trainer): 
        # add the average train loss for the epoch to tensorboard
        if not hasattr(trainer, 'mean_train_loss') or not hasattr(trainer, 'epoch'): 
            logger.error(f'Trainer loss not available for logging')
            return 
        trainer.writer.add_scalar('Loss/train', trainer.mean_train_loss, trainer.epoch)
        return


class AddValLoss(Hook): 

    def after_epoch(self, trainer): 
        # add the average val loss for the epoch to tensorboard
        if not hasattr(trainer, 'mean_val_loss') or not hasattr(trainer, 'epoch'): 
            logger.error(f'Trainer loss not available for logging')
            return 
        trainer.writer.add_scalar('Loss/val', trainer.mean_val_loss, trainer.epoch)
        return



class EpochPBar(Hook): 
    def before_epoch(self, trainer): 
        self.steps = 0
        self.n_hashes = 0
        self.t = time.time()
        print(f'Train Epoch {trainer.epoch}\t[', end='', flush=True)

    def after_train_step(self, trainer):
        self.steps += 1
        # incrament a progress bar for the training epoch
        n_steps = len(trainer.train_dataloader)
        pct_done = self.steps / n_steps
        n_hashes = int(pct_done * 10)
        if n_hashes > self.n_hashes: 
            self.n_hashes = n_hashes
            print('#', end='', flush=True)


    def after_epoch(self, trainer):
        print(f']\tEpoch Time: {time.time() - self.t:.1f}s')


class PlotLayerGrads(Hook): 

    def before_train(self, trainer): 
        trainer.writer.add_custom_scalars({
            'Gradients': {'block_gradients': ['Multiline', ['Blocks/Grad/norm_0', 'Blocks/Grad/norm_1', 'Blocks/Grad/norm_2', 'Blocks/Grad/norm_3']]}
        })

    def before_epoch(self, trainer): 
        if trainer.epoch % self.every != 0: 
            return
        
        self.run_hook = True
        # register pytorch hooks
        self.handles = []
        self.grad_norms = {}
        def make_block_grad_norm_hook(name): 
            def hook(module, grad_input, grad_output): 
                grads = [g for g in grad_output if g is not None]
                if not grads: 
                    return
                norm = torch.cat([g.detach().flatten() for g in grads]).norm().item()
                self.grad_norms[name] = norm

            return hook

        # my model encoder backbone only
        if hasattr(trainer.model.encoder, 'blocks'): 
            for i, block in enumerate(trainer.model.encoder.blocks): 
                handle = block.register_full_backward_hook(make_block_grad_norm_hook(f'block_{i}'))
                self.handles.append(handle)


    def after_train_step(self, trainer): 
        # logs block gradients for the first training batch
        if self.run_hook: 
            for i, norm in enumerate(self.grad_norms.values()): 
                trainer.writer.add_scalar(f'Blocks/Grad/norm_{i}', norm, trainer.epoch)

            for handle in self.handles: 
                handle.remove()
        
            self.run_hook = False
        return


class PlotLayerActNorms(Hook): 

    def before_train(self, trainer): 
        trainer.writer.add_custom_scalars({
            'Activations': {'block_activations': ['Multiline', ['Blocks/Act/norm_0', 'Blocks/Act/norm_1', 'Blocks/Act/norm_2', 'Blocks/Act/norm_3']]}
        })
        return

    def before_epoch(self, trainer): 
        if trainer.epoch % self.every != 0: 
            return
        
        self.run_hook = True
        # register pytorch hooks
        self.handles = []
        self.act_norms = {}
        def make_block_act_norm_hook(name): 
            def hook(module, input, output): 
                act_norm = output.detach().norm().item()
                self.act_norms[name] = act_norm

            return hook

        # my model encoder backbone only
        if hasattr(trainer.model.encoder, 'blocks'): 
            for i, block in enumerate(trainer.model.encoder.blocks): 
                handle = block.register_forward_hook(make_block_act_norm_hook(f'block_{i}'))
                self.handles.append(handle)


    def after_train_step(self, trainer): 
        # logs block activations for the first training batch
        if self.run_hook: 
            for i, norm in enumerate(self.act_norms.values()): 
                trainer.writer.add_scalar(f'Blocks/Act/norm_{i}', norm, trainer.epoch)

            for handle in self.handles: 
                handle.remove()
        
            self.run_hook = False
        return


class PlotLayerSidVals(Hook):
    def before_train(self, trainer): 
        trainer.writer.add_custom_scalars({
            'SID Values': {'sid_vals': ['Multiline', ['Blocks/Sid/sid_0', 'Blocks/Sid/sid_1', 'Blocks/Sid/sid_2', 'Blocks/Sid/sid_3']]}
        })
        return
    
    def before_epoch(self, trainer): 
        if trainer.epoch % self.every != 0: 
            return
        
        self.run_hook = True
        # register pytorch hooks
        self.handles = []
        self.sid_vals = {}
        def make_block_act_norm_hook(name): 
            def hook(module, input, output): 
                sid_val = module.attn_layer.sid_value.detach().sum()
                self.sid_vals[name] = sid_val

            return hook

        # my model encoder backbone only
        if hasattr(trainer.model.encoder, 'blocks'): 
            for i, block in enumerate(trainer.model.encoder.blocks): 
                handle = block.register_forward_hook(make_block_act_norm_hook(f'block_{i}'))
                self.handles.append(handle)


    def after_train_step(self, trainer): 
        # logs block activations for the first training batch
        if self.run_hook: 
            for i, val in enumerate(self.sid_vals.values()): 
                trainer.writer.add_scalar(f'Blocks/Sid/sid_{i}', val, trainer.epoch)

            for handle in self.handles: 
                handle.remove()
        
            self.run_hook = False
        return



class PrintLosses(Hook): 
    ...


class RunLinprobeTest(Hook):
    def after_train(self, trainer): 
        logger.info(f"Starting linear probing test run.")
        
        ...


class LogHParams(Hook): 
    def before_train(self, trainer): 
        with Path(trainer.experiment_folder, 'config.yaml').open('w') as f:
            yaml.dump(trainer.cfg, f)

        return


class SaveModel(Hook): 

    def after_train(self, trainer):
        torch.save(trainer.model, trainer.experiment_folder / "rest.pt")
        