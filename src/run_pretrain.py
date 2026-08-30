from pathlib import Path
import pandas as pd
import logging
import torch
import sys
import time
import yaml

import bdb2025
import bdb2024 
import datasets 
import models 
import utils 
import train 
import hooks 

logger = logging.getLogger(__name__)
device = torch.device('mps')


def parse_config():
    with open("../config.yaml", "r") as configfile:
        config = yaml.safe_load(configfile)
        print(config)
    return config


def main(config: dict): 
    # load bdb data
    data_config = config["data"]
    data_path = Path(
        data_config['data_folder']) / ("bdb2024" + ("_minimal.parquet" if data_config["minimal"] else ".parquet")
    )
    if data_config['force'] or not data_path.exists(): # TODO: with caching
        t1 = time.time()
        bdb2024.run_data_pipeline(
            Path(data_config['bdb_folder']), 
            Path(data_config['data_folder']), 
            minimal=data_config['minimal']
        )
        logger.info(f"Data pipeline completed in {time.time() - t1:.0f} seconds")
   
    # set index so that the train/test split keeps frames together
    # run dataset pipeline (task)
    logger.info(f"reading dataframe from {data_path}")
    df = pd.read_parquet(data_path).set_index(['gameId', 'playId'])
    
    train_idx, val_idx = datasets.train_val_split_df(
        df, 
        data_config['train'], data_config['val']
    )

    train_set_path = Path(data_config["data_folder"], "train.pt")
    val_set_path = Path(data_config["data_folder"], "val.pt")

    if data_config['force'] or data_config['minimal'] or not (train_set_path.exists() and val_set_path.exists()): 
        train_set = datasets.FrameDataset(df.loc[train_idx].copy())
        val_set = datasets.FrameDataset(df.loc[val_idx].copy())

        if not data_config["minimal"]: 
            torch.save(train_set, train_set_path)
            torch.save(val_set, val_set_path)
    else: 
        train_set = torch.load(train_set_path, weights_only=False)
        val_set = torch.load(val_set_path, weights_only=False)
        logger.info(f"Reading torch datasets from {data_config['data_folder']}")


    model_config = config["model"]

    model = models.REST(model_config, train_set.channels()).to(device)

    train_config = config["train"]
    trainer = train.MAETrainer(train_config, model, train_set, val_set)
    # add hooks
    train_hooks = [
        hooks.PlotFrame(1), hooks.AddTrainLoss(1), hooks.AddValLoss(1),
        hooks.PlotLayerGrads(1), hooks.PlotLayerActNorms(1), hooks.SaveModel(),
        hooks.EpochPBar(1)
    ]
    trainer.register_hooks(train_hooks)

    print(f'Training model for {train_config["epochs"]} epochs')
    print(f'{len(train_set)} training samples')
    trainer.train()

    return 

    

if __name__ == "__main__":  
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger('utils').setLevel(logging.DEBUG)

    config = parse_config()

    main(config)
    print('done')