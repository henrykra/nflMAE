from pathlib import Path
import pandas as pd
import logging
import torch
import sys
import time
import yaml

from nflMAE.data import bdb2025, datasets
from nflMAE.models import models
from nflMAE.engines import train
from nflMAE.utils import hooks
from nflMAE import root
from nflMAE.utils import utils
from nflMAE.data.bdb_common import bdb_folder

logger = logging.getLogger(__name__)
device = torch.device("mps")

def parse_config():
    with Path(root.parent, "configs", "pretrain_config.yaml").open("r") as configfile:
        config = yaml.safe_load(configfile)
        logger.info(config)
    return config


def main(config: dict): 
    # load bdb data
    data_config = config['data']

    if data_config['bdb_year'] != 2025:
        raise ValueError(f"run_pretrain currently only supports BDB 2025 data, got bdb_year={data_config['bdb_year']}")

    bdb_2025_folder = bdb_folder(data_config['bdb_year'])
    
    data_path_2025 = Path(
        data_config['data_folder']) / ("bdb2025" + ("_minimal.parquet" if data_config["minimal"] else ".parquet")
    )

    if data_config['force'] or not data_path_2025.exists(): # TODO: with caching
        t1 = time.time()
        bdb2025.run_data_pipeline(
            bdb_2025_folder,
            Path(data_config['data_folder']),
            minimal=data_config['minimal']
        )
        logger.info(f"BDB 2025 Data pipeline completed in {time.time() - t1:.0f} seconds")

    # set index so that the train/test split keeps frames together
    # run dataset pipeline (task)
    logger.info(f"reading dataframe from {data_path_2025}")
    df_2025 = pd.read_parquet(data_path_2025).set_index(['gameId', 'playId'])

    df = df_2025
    
    train_idx, val_idx = datasets.train_val_split_df(
        df, 
        data_config['train'], data_config['val']
    )

    train_set_path = Path(data_config['data_folder'], "train.pt")
    val_set_path = Path(data_config['data_folder'], "val.pt")

    if data_config['force'] or data_config['minimal'] or not (train_set_path.exists() and val_set_path.exists()): 
        train_set = datasets.MultiFrameDataset(df.loc[train_idx].copy())
        val_set = datasets.MultiFrameDataset(df.loc[val_idx].copy())

        if not data_config['minimal']: 
            torch.save(train_set, train_set_path)
            torch.save(val_set, val_set_path)
    else: 
        train_set = torch.load(train_set_path, weights_only=False)
        val_set = torch.load(val_set_path, weights_only=False)
        logger.info(f"Reading torch datasets from {data_config['data_folder']}")

    model_config = config['model']

    model = models.NflMaeModel(model_config, train_set.channels()).to(device)

    train_config = config['train']
    trainer = train.MAETrainer(train_config, model, train_set, val_set, folder_name=train_config['folder_name'])
    # add hooks
    train_hooks = [
        hooks.PlotFrame(1), hooks.AddTrainLoss(1), hooks.AddValLoss(1),
        hooks.PlotLayerGrads(1), hooks.PlotLayerActNorms(1), hooks.SaveModel(),
        hooks.EpochPBar(1), hooks.PlotLayerSidVals(1),
    ]
    trainer.register_hooks(train_hooks)

    logger.info(f"Training model for {train_config['epochs']} epochs")
    logger.info(f"{len(train_set)} training samples")
    trainer.train()

    return 

    

if __name__ == "__main__":  
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger("utils").setLevel(logging.DEBUG)

    config = parse_config()

    main(config)
    print("done")