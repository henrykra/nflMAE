"""Functions for perparing datasets for modeling"""
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
import torch
import logging
from pathlib import Path
logger = logging.getLogger(__name__)


def interpolate_movement(tracking_df: pd.DataFrame, interp: int) -> np.ndarray:

    play_index = tracking_df.set_index(['gameId', 'playId', 'mirrored']).sort_index()

    num_obs = interp
    interpolated_columns = ['x', 'y', 'ox', 'oy', 'vx', 'vy', 'a']
    ids = play_index.index.unique()
    num_plays = len(ids)
    routes_per_play = tracking_df.groupby(['gameId', 'playId'])['nflId'].nunique(dropna=True).max()

    print(f"interpolating data {len(interpolated_columns)} columns of data")
    # array for interpolated movement
    movement_arr = np.zeros( (num_plays, routes_per_play, num_obs, len(interpolated_columns)), dtype=np.float32)

    for i, (gameid, playid, mirrored) in enumerate(ids): 
        
        play: pd.DataFrame = play_index.loc[gameid, playid, mirrored]
        
        for ii, nflid in enumerate(play['nflId'].dropna().unique()):

            player: pd.DataFrame = play.loc[play['nflId'] == nflid]
            min_frame = player['frameId'].min()
            frame_range = player['frameId'].max() - min_frame
            t = np.array((player['frameId'] - min_frame) / frame_range)

            player_data = np.array(player[interpolated_columns])

            # smooth the info about the player over time
            spline = CubicSpline(t, player_data, axis=0, extrapolate=False)

            # take n observations
            observation_times = np.arange(num_obs) / num_obs

            # populate an array with smoothed values
            movement_arr[i, ii, ...] = spline(observation_times)


    return movement_arr


# def prepare_data_arrays(static_df: pd.DataFrame, movement_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     logger.info("preparing arrays")
#     play_index = static_df.set_index(['gameId', 'playId', 'mirrored'])
    
#     num_plays = len(play_index.index.unique())
#     routes_per_play = static_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=True).iloc[0]

#     assert (static_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=False) == routes_per_play).all(), \
#     "Different values across player dimension"
    
#     num_columns = len(play_index.columns) - 1 # subtract the nflId column
#     player_info = np.zeros( (num_plays, routes_per_play, num_columns), dtype=np.float32) 

#     for i, (name, group) in enumerate(play_index.groupby(['gameId', 'playId', 'mirrored'])):
#         player_info[i, ...] = group.drop(columns=['nflId'])

#     input_arr = np.concat(
#         [
#             movement_arr.reshape((num_plays, routes_per_play, -1)), # combine final axes 
#             player_info, # concat fixed info
#         ],
#         axis=-1
#     )

#     target_arr = movement_arr[..., :2].reshape((num_plays, routes_per_play, -1))

#     return input_arr, target_arr


def generate_mask(static: np.ndarray, mask_pct: float) -> np.ndarray:
    print("generating mask")

    n_player_vectors = np.prod(static.shape[:-1])
    mask_idx = np.random.choice(
        n_player_vectors,
        size=int(n_player_vectors * mask_pct),
        replace=False
    ) # random choice over the product of the first axes

    mask = np.ones(n_player_vectors)
    mask[mask_idx] = 0

    return mask.reshape((*static.shape[:-1], 1))



def train_val_test_split(n: int, train_pct: float, val_pct: float, test_pct: float):
    print(f"making splits at {train_pct}, {val_pct}, {test_pct}")
    idx = np.arange(n)

    cats = np.random.choice(a=[0, 1, 2], size=n, replace=True, p=[train_pct, val_pct, test_pct])

    split = np.stack([idx, cats], axis=-1)

    train_idx = split[split[:, -1] == 0, 0]
    val_idx = split[split[:, -1] == 1, 0]
    test_idx = split[split[:, -1] == 2, 0]

    return train_idx, val_idx, test_idx


class NflDataset(Dataset):
    
    def __init__(self, dynamic_arr: np.ndarray, static_arr: np.ndarray, mask_arr: np.ndarray, idx: np.ndarray, premask: bool=True, mask_pct: float=.15):

        # make sure there's the right amount of input observatinos and target observations
        assert static_arr.shape[0] == dynamic_arr.shape[0], "# Observations mismatch between input and target"
        self.premask = premask
        self.mask_pct = mask_pct
        n_plays = len(idx)
        routes_per_play =  static_arr.shape[1]

        self.static_t: torch.Tensor = torch.Tensor(static_arr[idx].copy())
        self.dynamic_t: torch.Tensor = torch.Tensor(dynamic_arr[idx].reshape(n_plays, routes_per_play, -1).copy())
        self.target_t: torch.Tensor = torch.Tensor(dynamic_arr[idx, ..., :2].reshape(n_plays, routes_per_play, -1).copy())
        self.mask_t: torch.Tensor = torch.Tensor(mask_arr[idx].copy()).bool()
        return
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        tar = self.target_t[idx]

        # mask dynamic features
        if self.premask:
            inp = torch.cat([self.static_t[idx], (self.dynamic_t[idx] * self.mask_t[idx])], dim=-1)
            mask = ~self.mask_t[idx]
        else:
            # random mask
            mask = (torch.rand(size=(self.dynamic_t.shape[1], 1)) < self.mask_pct).bool()
            inp = torch.cat([self.static_t[idx], (self.dynamic_t[idx] * ~mask)], dim=-1)

        return inp, tar, mask
        

    def __len__(self) -> int:

        return self.static_t.shape[0]
    

def load_dataset(filepath: Path) -> NflDataset:
    dataset = torch.load(filepath, weights_only=False)
    return dataset

def write_dataset(dataset: Dataset, filepath: Path) -> NflDataset:
    torch.save(dataset, filepath)
    
    
    
