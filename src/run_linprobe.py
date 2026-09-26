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
import finetune
import backbones
import hooks 

logger = logging.getLogger(__name__)
device = torch.device('mps')


def parse_config():
    with open("../configs/tackle_prediction_config.yaml", "r") as configfile:
        config = yaml.safe_load(configfile)
        print(config)
    return config


def main(config: dict): 
    # load bdb data
    data_config = config["data"]
    data_path = Path(data_config['data_folder']) / ("bdb2024" + "_finetune.parquet")

    ## Train    
    t1 = time.time()
    bdb2024.run_data_pipeline(
        Path(data_config['bdb_folder']), 
        Path(data_config['data_folder']), 
        weeks=[1, 2],
        df_name=data_path.name
    )
    logger.info(f"Data pipeline completed in {time.time() - t1:.0f} seconds")

    # set index so that the train/test split keeps frames together
    # run dataset pipeline (task)
    logger.info(f"reading dataframe from {data_path}")
    df = pd.read_parquet(data_path)

    train_set_path = Path(data_config["data_folder"], "train_finetune.pt")

    train_set = datasets.TacklePosDataset(df)
    torch.save(train_set, train_set_path)


    ## Val
    t1 = time.time()
    bdb2024.run_data_pipeline(
        Path(data_config['bdb_folder']), 
        Path(data_config['data_folder']), 
        weeks=[3],
        df_name=data_path.name
    )
    logger.info(f"Data pipeline completed in {time.time() - t1:.0f} seconds")

    logger.info(f"reading dataframe from {data_path}")
    df = pd.read_parquet(data_path)

    val_set_path = Path(data_config["data_folder"], "val_finetune.pt")

    val_set = datasets.TacklePosDataset(df)
    torch.save(val_set, val_set_path)

    ## Test
    t1 = time.time()
    bdb2024.run_data_pipeline(
        Path(data_config['bdb_folder']), 
        Path(data_config['data_folder']), 
        weeks=[9],
        df_name=data_path.name
    )
    logger.info(f"Data pipeline completed in {time.time() - t1:.0f} seconds")

    logger.info(f"reading dataframe from {data_path}")
    df = pd.read_parquet(data_path)

    test_set_path = Path(data_config["data_folder"], "test_finetune.pt")

    test_set = datasets.TacklePosDataset(df)
    torch.save(test_set, test_set_path)

    model_path = config["model"]

    rest_model = torch.load(model_path, weights_only=False).to(device)
    #for parameter in rest_model.parameters(): 
    #    assert parameter.requires_grad == False

    train_config = config["train"]
    trainer = finetune.LinProber(train_config, rest_model, train_set, val_set, test_set)
    # add hooks
    train_hooks = [
        hooks.PlotTackle(1), hooks.AddTrainLoss(1), hooks.AddValLoss(1), hooks.EpochPBar(1)
    ]
    trainer.register_hooks(train_hooks)

    print(f'Finetuning model for {train_config["epochs"]} epochs')
    print(f'{len(train_set)} training samples')
    trainer.train()
    ade = trainer.test()
    trainer.writer.add_scalar("ADE/test", ade)
    return 

    

if __name__ == "__main__":  
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger('utils').setLevel(logging.DEBUG)

    config = parse_config()

    main(config)
    print('done')