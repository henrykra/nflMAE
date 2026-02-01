import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
import plotly.graph_objects as go
import io
from pathlib import Path


def check_early_stopping(loss: float, best_val_loss: float, no_improvement: int, patience: int) -> tuple[bool, float, int]:
    if loss < best_val_loss:
        return True, loss, 0 # reset no improvement
    else:
        no_improvement += 1
        if no_improvement >= patience:
            return False, best_val_loss, patience
        else:
            return True, best_val_loss, no_improvement
        

def start_tensorboard(folder: Path) -> SummaryWriter:
    writer = SummaryWriter(folder)
    return writer


def plot_eval(tar: np.ndarray, pred: np.ndarray, mask: np.ndarray, idx: int, interp: int):
    
    if mask.shape[-1] == 1:
        mask = mask.squeeze(axis=-1)

    # static (not animated) plotly plot to visualize model predictions against groundtruth

    # plot the first of the entire input array
    targets = tar.reshape(*tar.shape[:-1], interp, -1)
    predictions = pred.reshape(*pred.shape[:-1], interp, -1)

    starts = targets[idx, :, 0, :2] # first frame of x&y for 22 players

    # get the masked player's movements
    gt_movement = targets[idx, mask[idx].astype(np.bool)]

    # get predicted movements
    pred_movement = predictions[idx, mask[idx].astype(np.bool)]


    # plot start locations and lines for gt and pred movement
    fig = go.Figure()

    fig.add_trace( # start locations
        go.Scatter(
            y = starts[..., 0],
            x = starts[..., 1],
            mode = 'markers', 
            showlegend=False
        )
    )

    for i, (player_gt, player_pred) in enumerate(zip(gt_movement, pred_movement)):

        fig.add_trace(
            go.Scatter(
                y = player_gt[..., 0],
                x = player_gt[..., 1],
                name = f"groundtruth {i}",
                showlegend = False,
                mode = 'lines',
                marker = dict(
                    color = 'black'
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                y = player_pred[..., 0],
                x = player_pred[..., 1],
                name = f"predicted {i}",
                showlegend = False,
                mode = 'lines',
                marker = dict(
                    color = 'red'
                )
            )
        )
    
    return fig


def create_optim(model: torch.nn.Module, hparams: dict):
    
    return torch.optim.AdamW(model.parameters(), lr=hparams["lr"])

def create_loss_fn():

    return torch.nn.SmoothL1Loss()

def create_dataloaders(train_set: torch.utils.data.DataLoader, val_set: torch.utils.data.DataLoader, test_set: torch.utils.data.Dataset, hparams):
    bs = hparams["batch_size"]
    return (
        torch.utils.data.DataLoader(train_set, bs, True),
        torch.utils.data.DataLoader(val_set, bs, True),
        torch.utils.data.DataLoader(test_set, bs, False)
    )


def train(hparams: dict, model: torch.nn.Module, train_set: torch.utils.data.Dataset, val_set: torch.utils.data.Dataset, test_set: torch.utils.data.Dataset, experiment_folder: Path):
    writer = start_tensorboard(experiment_folder)
    
    loss_fn = create_loss_fn()
    optim = create_optim(model, hparams)
    train_loader, val_loader, test_loader = create_dataloaders(train_set, val_set, test_set, hparams)

    for epoch in range(hparams["epochs"]):

        c = 0
        train_loss = 0
        val_loss = 0
        best_val_loss = float("inf")
        no_improvement = 0
        model.train()
        for (inp, tar, mask) in train_loader:
            inp, tar, mask = inp.to(hparams["device"]), tar.to(hparams["device"]), mask.to(hparams["device"])
            optim.zero_grad()
            x = model(inp)

            loss = loss_fn(x*mask, tar*mask)
            train_loss += loss
            c += 1
            loss.backward()
            
            optim.step()
        # log average loss
        writer.add_scalar('Loss/train', train_loss / c, epoch)

        c = 0
        model.eval()
        with torch.no_grad():
            for (inp, tar, mask) in val_loader:
                inp, tar, mask = inp.to(hparams["device"]), tar.to(hparams["device"]), mask.to(hparams["device"])
                x = model(inp)

                loss = loss_fn(x*mask, tar*mask)
                val_loss += loss
                c += 1

        # log average loss
            writer.add_scalar('Loss/val', val_loss / c, epoch)

            cont, best_val_loss, no_improvement = check_early_stopping(val_loss, best_val_loss, no_improvement, hparams["patience"])
            if not cont: # early stopped
                print(f"Early stopping at epoch {epoch}\nBest Loss: {best_val_loss}")
                break

            if epoch % 10 == 0:
                # log a sample play
                pred = model(inp)
                
                fig = plot_eval(tar.cpu().numpy().copy(), pred.cpu().numpy().copy(), mask.cpu().numpy().copy(), 0, 40)
                img_bytes = fig.to_image(format='jpg')
                img = Image.open(io.BytesIO(img_bytes))
                
                writer.add_image(f"sample/{epoch}", np.array(img).transpose([2, 0, 1]))

                print(f"Epoch: {epoch}\tVal Loss: {val_loss/c:.3f}")
                writer.flush()
            
    writer.flush()
    writer.close()

    # get test loss
    test_loss = 0
    c = 0
    with torch.no_grad():
        for (inp, tar, mask) in test_loader:
            inp, tar, mask = inp.to(hparams["device"]), tar.to(hparams["device"]), mask.to(hparams["device"])
            x = model(inp)

            loss = loss_fn(x*mask, tar*mask)
            test_loss += loss
            c += 1

    print(f"Test Loss: {test_loss / c:.3f}")
    return test_loss / c
