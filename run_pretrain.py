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

import src.bdb2025 as bdb2025
import src.bdb2024 as bdb2024
import src.datasets as datasets
import src.backbones as backbones
import src.utils as utils
import src.train as train

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
        model = backbones.MyModel(
            feature_channels = train_set.channels(),
            model_dimension=model_config["model_dimension"],
            hidden_dimension=model_config["model_dimension"] * 4,
            heads = model_config["n_heads"],
            attn_drop=.3,
            proj_drop=.3,
            ffn_drop=.3,
            depth=model_config["n_layers"],
            out_channels=train_set.target_channels(),
            add_rpb=model_config["rpb"],
            norm_first=model_config["norm_first"]
        ).to(device)
    elif model_config['model'] == 'stt':
        model = backbones.SportsTransformer(
            train_set.channels(),
            model_config['model_dimension'],
            model_config['n_layers'],
        ).to(device)
    elif model_config['model'] == 'zoo': 
        pass

    train_config = config["train"]
    print(f'Training model for {train_config["epochs"]} epochs')
    experiment_folder = make_experiment_folder(train_config)
    test_ade = train.train(model, train_set, val_set, test_set, train_config, experiment_folder)

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

    config = parse_config()

    main(config)
    print('done')