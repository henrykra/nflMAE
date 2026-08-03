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
import models
import utils

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



def main(config: dict): 
    
    # load bdb data
    data_config = config["data"]
    data_path = Path(
        data_config['data_folder']) / ("bdb2024" + ("_minimal.parquet" if data_config["minimal"] else ".parquet")
    )

    if data_config['force'] or not data_path.exists(): 
        t1 = time.time()
        bdb2024.run_data_pipeline(
            Path(data_config['bdb_folder']), 
            Path(data_config['data_folder']), 
            minimal=data_config['minimal']
        )
        logger.info(f"Data pipeline completed in {time.time() - t1:.0f} seconds")
   
    # set index so that the train/test split keeps frames together
    logger.info(f"reading dataframe from {data_path}")
    df = pd.read_parquet(data_path).set_index(['gameId', 'playId'])
    
    train_idx, val_idx, test_idx = datasets.train_val_test_split_df(
        df, 
        data_config['train'], data_config['val'], data_config['test']
    )

    train_set_path = Path(data_config["data_folder"], "train.pt")
    val_set_path = Path(data_config["data_folder"], "val.pt")
    test_set_path = Path(data_config["data_folder"], "test.pt")

    if data_config['force'] or data_config['minimal'] or not (train_set_path.exists() and val_set_path.exists() and test_set_path.exists()): 
        train_set = datasets.TacklePosDataset(df.loc[train_idx].copy())
        val_set = datasets.TacklePosDataset(df.loc[val_idx].copy())
        test_set = datasets.TacklePosDataset(df.loc[test_idx].copy())

        if not data_config["minimal"]: 
            torch.save(train_set, train_set_path)
            torch.save(val_set, val_set_path)
            torch.save(test_set, test_set_path)
    else: 
        train_set = torch.load(train_set_path, weights_only=False)
        val_set = torch.load(val_set_path, weights_only=False)
        test_set = torch.load(test_set_path, weights_only=False)
        logger.info(f"Reading torch datasets from {data_config['data_folder']}")



    model_config = config["model"]
    if model_config['model'] == 'my': 
        model = models.MyModel(
            feature_channels = train_set.channels(),
            model_dimension=model_config["model_dimension"],
            hidden_dimension=model_config["model_dimension"] * 4,
            heads = model_config["n_heads"],
            attn_drop=.3,
            proj_drop=.3,
            ffn_drop=.3,
            depth=model_config["n_layers"],
            out_channels=train_set.target_channels(),
            add_rpb=model_config['rpb']
        ).to(device)
    elif model_config['model'] == 'stt':
        model = models.SportsTransformer(
            train_set.channels(),
            model_config['model_dimension'],
            model_config['n_layers'],
        ).to(device)
    elif model_config['model'] == 'zoo': 
        pass

    train_config = config["train"]
    print(f'Training model for {train_config["epochs"]} epochs')
    experiment_folder = make_experiment_folder(train_config)
    test_ade = train(model, train_set, val_set, test_set, train_config, experiment_folder)

    out_config = {
        'config': config,
        'metric': test_ade
    }

    with open(Path(experiment_folder, 'result.json'), 'w') as f:
        json.dump(out_config, f, indent=4)

    return 

    

if __name__ == "__main__": 
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger('utils').setLevel(logging.DEBUG)
    logging.getLogger("choreographer").setLevel(logging.WARNING)
    logging.getLogger("kaleido").setLevel(logging.WARNING)

    config = parse_config()

    main(config)
    print('done')