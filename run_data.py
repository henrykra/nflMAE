"""Entrypoint for data pipeline. Outputs experiment folder for gpu training"""
import bdb2025
import datasets
from pathlib import Path
import json

def parse_config():
    with open("./data_config.json", "r") as configfile:
        config = json.load(configfile)
    return config

def write_config(config):
    with open(Path(config["targetfolder"]) / "data_config.json", "w") as f:
        json.dump(config, f)

def main(config: dict):

    if config["minimal"]:
        tracking = bdb2025.load_csv_data([Path("minimal_tracking.csv")])[0]
    else:
        tracking = bdb2025.load_tracking_data(folder=Path(config["folder"]))
    players = bdb2025.load_player_data(folder=Path(config["folder"]))
    plays = bdb2025.load_play_data(folder=Path(config["folder"]))
    # clean
    players = bdb2025.clean_player_data(players)
    tracking = bdb2025.clean_tracking_data(tracking)
    tracking = bdb2025.mirror_tracking_plays(tracking)

    static, dynamic, _ = bdb2025.create_data_arrays(tracking, plays, players, config["interp"])
    # prepare
    # generate mask
    mask = datasets.generate_mask(static, config["mask"])
    n = static.shape[0]
    train_idx, val_idx, test_idx = datasets.train_val_test_split(n, config["train"], config["val"], config["test"])
    train_dataset = datasets.NflDataset(dynamic, static, mask, train_idx, premask=config["premask"], mask_pct=config["mask"])
    val_dataset = datasets.NflDataset(dynamic, static, mask, val_idx)
    test_dataset = datasets.NflDataset(dynamic, static, mask, test_idx)
    # write train in, train out, train mask
    datasets.write_dataset(train_dataset, Path(config["targetfolder"]) / "train.pt")
    datasets.write_dataset(val_dataset, Path(config["targetfolder"]) / "val.pt")
    datasets.write_dataset(test_dataset, Path(config["targetfolder"]) / "test.pt")
    write_config(config)
    print(f"Generated {len(train_idx)} training samples")
    return

if __name__ == "__main__":
    config = parse_config()

    main(config)

