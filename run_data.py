"""Entrypoint for data pipeline. Outputs experiment folder for gpu training"""
import bdb2025
import datasets
from pathlib import Path
import json

def parse_config():
    with open("./data_config.json", "r") as configfile:
        config = json.load(configfile)
    return config


def write_config(config, dataset_name: str):
    with open(Path(config["targetfolder"]) / "data_config.txt", "w") as f:
        json.dump(config, f, indent=2)
        f.write(f'\nDataset: {dataset_name}\n')


def main(config: dict):

    # base load
    if config["minimal"]:
        tracking = bdb2025.load_csv_data([Path("minimal_tracking.csv")])[0]
    else:
        tracking = bdb2025.load_tracking_data(folder=Path(config["folder"]))

    player_plays = bdb2025.load_player_play_data(folder=Path(config["folder"]))
    plays = bdb2025.load_play_data(folder=Path(config["folder"]))
    players = bdb2025.load_player_data(folder=Path(config["folder"]))


    # clean
    players = bdb2025.clean_player_data(players)
    tracking = bdb2025.clean_tracking_angles(tracking)
    tracking = bdb2025.standardize_tracking_plays(tracking)

    
    # filter
    tracking = bdb2025.filter_play_start_frames(tracking)
    tracking = bdb2025.filter_play_end_frames(tracking)


    # transform
    plays, tracking = bdb2025.find_play_invariant_columns(plays, tracking)
    player_plays, tracking = bdb2025.find_player_play_invariant_columns(player_plays, tracking)
    player_plays, plays = bdb2025.clean_player_play_data(player_plays, plays)
    tracking, player_plays = bdb2025.player_relative_positions(tracking, player_plays)

    # mirror
    tracking = bdb2025.mirror_tracking_data(tracking)


    # merge
    merged = bdb2025.merge_dataframes_wide(tracking, player_plays, plays, players,
                                           ['gameId', 'playId', 'nflId', 'mirrored', 'frameId', 'x', 'y', 'vx', 'vy', 'o', 'dir', 'a'],
                                           ['gameId', 'playId', 'nflId', 'is_ball', 'offense', 'defense', 'x_start', 'y_start'],
                                           ['gameId', 'playId', 'x_ball_snap', 'y_ball_snap'],
                                           ['nflId', 'height_z', 'weight_z', 'position'])
    
    merged = merged.set_index(['gameId', 'playId', 'mirrored'])

    train_idx, val_idx, test_idx = bdb2025.train_val_test_split_df(
        merged, config["train"], config["val"], config["test"]
    )

    # create datasets

    train_dataset = datasets.ManZoneDataset(merged.loc[train_idx].copy(), plays)
    val_dataset = datasets.ManZoneDataset(merged.loc[val_idx].copy(), plays)
    test_dataset = datasets.ManZoneDataset(merged.loc[test_idx].copy(), plays)

    # write datasets
    datasets.write_dataset(train_dataset, Path(config["targetfolder"]) / "train.pt")
    datasets.write_dataset(val_dataset, Path(config["targetfolder"]) / "val.pt")
    datasets.write_dataset(test_dataset, Path(config["targetfolder"]) / "test.pt")

    # save the config
    write_config(config, train_dataset.name)

    print(f"Generated {len(train_dataset)} training samples")
    return

if __name__ == "__main__":
    config = parse_config()

    main(config)

