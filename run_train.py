"""entrypoint for training"""
import logging
from pathlib import Path
from datetime import datetime
import sys
import json
logger = logging.getLogger(__name__)


import datasets
import modeling
import train

def parse_config(): 

    with open('./train_config.json') as f:
        config = json.load(f)
    
    return config


def main(config: dict):
    # create experiment folder
    now = datetime.now().strftime("%m-%d-%y--%H:%M:%S")
    experiment_folder = Path(config["target_folder"]) / ("experiment" + now)
    Path.mkdir(experiment_folder)

    train_dataset = datasets.load_dataset(Path(config["data_folder"]) / "train.pt")
    val_dataset = datasets.load_dataset(Path(config["data_folder"]) / "val.pt")
    test_dataset = datasets.load_dataset(Path(config["data_folder"]) / "test.pt")
    with open(Path(config["data_folder"]) / "data_config.txt", "r") as f:
        data_config = f.read()

    # debug
    # model
    model = modeling.NflClassifier(
        train_dataset.data.shape[-1], # in 
        train_dataset.target.shape[-1], # out
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"], 
        num_heads=config["num_heads"], 
        ffn_size=config["ffn_size"],
    ).to(config["device"])
    # train
    trainer = train.NflClassiferTrainer(config, model)
    test_loss = trainer.train(train_dataset, val_dataset, test_dataset, experiment_folder)
    modeling.save_model(model, experiment_folder / "model.pt")
    # write
    with open(experiment_folder / "out.txt", "a") as outfile:
        outfile.write(data_config)
        outfile.write('\n')
        json.dump(config, outfile)
        outfile.write(f"\nTest Loss: {test_loss}")
    
    return


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger("choreographer").setLevel(logging.WARNING)
    logging.getLogger("kaleido").setLevel(logging.WARNING)
    
    logger.info("starting")
    config = parse_config()

    main(config)