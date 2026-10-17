from pathlib import Path
import logging
import torch
import sys
import time
import yaml

import pandas as pd

from nflMAE.models import models
from nflMAE.data import bdb2024 
from nflMAE.data import datasets 
from nflMAE.engines import finetune
from nflMAE.utils import hooks 
from nflMAE import root
from nflMAE.utils import utils
from nflMAE.data.bdb_common import bdb_folder

logger = logging.getLogger(__name__)
device = torch.device('mps')


def parse_config():
    with Path(root.parent, "configs", "tackle_config.yaml").open("r") as configfile:
        config = yaml.safe_load(configfile)
        logger.info(config)
    return utils.resolve_config_paths(config)


def main(config: dict): 
    # load bdb data
    data_config = config['data']
    train_data_path = Path(data_config['data_folder']) / ("bdb2024" + "_tackle_train.parquet")
    test_data_path = Path(data_config['data_folder']) / ("bdb2024" + "_tackle_test.parquet")

    ## Train / Val
    if data_config['force'] or not train_data_path.exists():
        t1 = time.time()
        bdb2024.run_data_pipeline(
            bdb_folder(data_config['bdb_year']), 
            Path(data_config['data_folder']), 
            minimal=data_config['minimal'],
            weeks=data_config['train_weeks'],
            df_name=train_data_path.name
        )
        logger.info(f"Data pipeline completed in {time.time() - t1:.0f} seconds")

    df = pd.read_parquet(train_data_path).set_index(['gameId', 'playId'])
    train_idx, val_idx = datasets.train_val_split_df(df, data_config['train'], data_config['val'])

    train_set_path = Path(data_config['data_folder'], "tackle_train.pt")
    val_set_path = Path(data_config['data_folder'], "tackle_val.pt")
    if data_config['force'] or not (train_set_path.exists() and val_set_path.exists()): 
        # run dataset pipeline (task)
        logger.info(f"reading dataframe from {train_data_path}")
        train_set = datasets.TacklePosDataset(df.loc[train_idx].copy())
        val_set = datasets.TacklePosDataset(df.loc[val_idx].copy())
        torch.save(train_set, train_set_path)
        torch.save(val_set, val_set_path)
    else: 
        train_set = torch.load(train_set_path, weights_only=False)
        val_set = torch.load(val_set_path, weights_only=False)

    ## Test
    if data_config['force'] or not test_data_path.exists():
        t1 = time.time()
        bdb2024.run_data_pipeline(
            bdb_folder(data_config['bdb_year']), 
            Path(data_config['data_folder']), 
            minimal=data_config['minimal'],
            weeks=data_config['test_weeks'],
            df_name=test_data_path.name
        )
        logger.info(f"Data pipeline completed in {time.time() - t1:.0f} seconds")

    test_df = pd.read_parquet(test_data_path)

    test_set_path = Path(data_config['data_folder'], "tackle_test.pt")
    if data_config['force'] or not test_set_path.exists(): 
        # run dataset pipeline (task)
        logger.info(f"reading dataframe from {test_data_path}")
        test_set = datasets.TacklePosDataset(test_df)
        torch.save(test_set, test_set_path)
    else: 
        test_set = torch.load(test_set_path, weights_only=False)

    model_config = {'encoder': config['model']}
    if isinstance(config['model'], dict):
        # build model from scratch
        logger.info(f"Building model from scratch")
        scratch = True
        model = models.NflMaeModel(model_config, train_set.channels()).to(device)
    elif isinstance(config['model'], str) and Path(config['model']).exists():
        # read model
        logger.info(f"Reading pretrained model")
        scratch = False
        model = torch.load(config['model'], weights_only=False).to(device)
    else: 
        logger.error(f"Unknown model config: {config['model']}")
        return

    train_config = config['train']
    probe = finetune.build_linear_probe(model, train_config['target_features']).to(device)
    trainer = finetune.Finetuner(train_config, model, probe, train_set, val_set, test_set)
    
    # add hooks
    train_hooks = [
        hooks.PlotTackle(1), hooks.AddTrainLoss(1), hooks.AddValLoss(1), hooks.EpochPBar(1), hooks.PrintStat(1, "ADE")
    ]
    trainer.register_hooks(train_hooks)

    logger.info(f"Finetuning model for {train_config['epochs']} epochs")
    logger.info(f"{len(train_set)} training samples")

    if not scratch: 
        model.requires_grad_(False)
        trainer.train()
        ade = trainer.test()
        print(f"Test ADE: {ade:.4f}")
        # trainer.writer.add_scalar("ADE/test", ade) # TODO: log hyperparams

        trainer.clear()
        model.encoder.blocks[-1].requires_grad_(True)
        trainer.train()
        ade = trainer.test()
        print(f"Test ADE: {ade:.4f}")

    trainer.clear()
    model.requires_grad_(True)
    trainer.train()
    ade = trainer.test()
    print(f"Test ADE: {ade:.4f}")
    return 
    

if __name__ == "__main__":  
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger('utils').setLevel(logging.DEBUG)

    config = parse_config()

    main(config)
    print("done")