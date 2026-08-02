import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
import plotly.graph_objects as go
import io
from pathlib import Path



class NflTrainer(): 


    def __init__(self, hparams: dict):
        self.hparams = hparams 

        self.patience_counter = 0
        self.best_val_loss = float('inf')

        return
    

    def check_early_stopping(self, loss: float) -> tuple[bool, float, int]:
        if loss < self.best_val_loss:
            self.patience_counter = 0
            self.best_val_loss = loss
            return True # reset no improvement
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.hparams["patience"]:
                return False
            else:
                return True
            

    def start_tensorboard(self, folder: Path) -> SummaryWriter:
        writer = SummaryWriter(folder)
        return writer
    

    def create_dataloaders(self, train_set: torch.utils.data.DataLoader, val_set: torch.utils.data.DataLoader, test_set: torch.utils.data.Dataset):
        bs = self.hparams["batch_size"]

        return (
            torch.utils.data.DataLoader(train_set, bs, True),
            torch.utils.data.DataLoader(val_set, bs, True),
            torch.utils.data.DataLoader(test_set, bs, False)
        )
    


class NflClassiferTrainer(NflTrainer): 

    def __init__(self, hparams, model: torch.nn.Module): 
        super().__init__(hparams)

        self.optim = torch.optim.AdamW(model.parameters(), lr=hparams["lr"])
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.model = model

        return 


    def train(self, train_set: torch.utils.data.Dataset, val_set: torch.utils.data.Dataset, test_set: torch.utils.data.Dataset, experiment_folder: Path):
        writer = self.start_tensorboard(experiment_folder)
 
        train_loader, val_loader, test_loader = self.create_dataloaders(train_set, val_set, test_set)

        for epoch in range(self.hparams["epochs"]):

            c = 0
            self.train_loss = 0
            self.val_loss = 0

            self.model.train()
            for (inp, tar) in train_loader:
                inp, tar = inp.to(self.hparams["device"]), tar.to(self.hparams["device"])
                self.optim.zero_grad()
                x = self.model(inp)

                loss = self.loss_fn(x, tar)
                self.train_loss += loss
                c += 1

                loss.backward()
                
                self.optim.step()
            # log average loss

            writer.add_scalar('Loss/train', self.train_loss / c, epoch)

            c = 0
            self.model.eval()
            with torch.no_grad():
                for (inp, tar) in val_loader:
                    inp, tar = inp.to(self.hparams["device"]), tar.to(self.hparams["device"])
                    x = self.model(inp)
                    

                    loss = self.loss_fn(x, tar)

                    self.val_loss += loss
                    c += 1

            # log average loss
                writer.add_scalar('Loss/val', self.val_loss / c, epoch)

                cont = self.check_early_stopping(self.val_loss / c)
                if not cont: # early stopped
                    print(f"Early stopping at epoch {epoch}\nBest Loss: {self.best_val_loss:.3f}")
                    break

                if epoch % 10 == 0:
                    # log a sample play
                    # pred = model(inp)
                    
                    # fig = plot_eval(tar.cpu().numpy().copy(), pred.cpu().numpy().copy(), mask.cpu().numpy().copy(), 0, 40)
                    # img_bytes = fig.to_image(format='jpg')
                    # img = Image.open(io.BytesIO(img_bytes))
                    
                    # writer.add_image(f"sample/{epoch}", np.array(img).transpose([2, 0, 1]))

                    print(f"Epoch: {epoch}\tVal Loss: {self.val_loss/c:.3f}")
                    writer.flush()
                
        writer.flush()
        writer.close()

        # get test loss
        self.test_loss = 0
        c = 0
        right = 0
        total = 0
        
        with torch.no_grad():
            for (inp, tar) in test_loader:
                inp, tar = inp.to(self.hparams["device"]), tar.to(self.hparams["device"])
                x = self.model(inp)
                right += sum(torch.argmax(x, dim=-1) == torch.argmax(tar, dim=-1))

                loss = self.loss_fn(x, tar)
                self.test_loss += loss
                c += 1
                total += tar.shape[0]

        print(f"Test Loss: {self.test_loss / c:.3f}")
        print(f"Test Accuracy: {right / total:.3f}")
        return self.test_loss / c






# def plot_eval(tar: np.ndarray, pred: np.ndarray, mask: np.ndarray, idx: int, interp: int):
    
#     if mask.shape[-1] == 1:
#         mask = mask.squeeze(axis=-1)

#     # static (not animated) plotly plot to visualize model predictions against groundtruth

#     # plot the first of the entire input array
#     targets = tar.reshape(*tar.shape[:-1], interp, -1)
#     predictions = pred.reshape(*pred.shape[:-1], interp, -1)

#     starts = targets[idx, :, 0, :2] # first frame of x&y for 22 players

#     # get the masked player's movements
#     gt_movement = targets[idx, mask[idx].astype(np.bool)]

#     # get predicted movements
#     pred_movement = predictions[idx, mask[idx].astype(np.bool)]


#     # plot start locations and lines for gt and pred movement
#     fig = go.Figure()

#     fig.add_trace( # start locations
#         go.Scatter(
#             y = starts[..., 0],
#             x = starts[..., 1],
#             mode = 'markers', 
#             showlegend=False
#         )
#     )

#     for i, (player_gt, player_pred) in enumerate(zip(gt_movement, pred_movement)):

#         fig.add_trace(
#             go.Scatter(
#                 y = player_gt[..., 0],
#                 x = player_gt[..., 1],
#                 name = f"groundtruth {i}",
#                 showlegend = False,
#                 mode = 'lines',
#                 marker = dict(
#                     color = 'black'
#                 )
#             )
#         )

#         fig.add_trace(
#             go.Scatter(
#                 y = player_pred[..., 0],
#                 x = player_pred[..., 1],
#                 name = f"predicted {i}",
#                 showlegend = False,
#                 mode = 'lines',
#                 marker = dict(
#                     color = 'red'
#                 )
#             )
#         )
    
#     return fig


