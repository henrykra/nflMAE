"""Functions for perparing datasets for modeling"""
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
import torch
import logging
from pathlib import Path
logger = logging.getLogger(__name__)




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



# create a dataset for a certian model/task
from torch.utils.data import Dataset, DataLoader
import torch

class NflDataset(Dataset): 
    """Default dataset"""

    name = "NFLDataset"

    def __init__(
        self, 
        merged_df: pd.DataFrame
    ):
        
        # base dataframe for getting inputs.
        # for each sample, get a B, P, F tensor

        # these are the base columns to be used as an input
        # the point is for these columns to be as general as possible, 
        # to be applied to any potential sports tracking dataset

        # think about solving the player ordering problem!

        id_cols = ['gameId', 'playId', 'mirrored', 'nflId', 'frameId']


        merged_df = merged_df.reset_index().set_index(['gameId', 'playId', 'mirrored']).sort_index()
        play_index = merged_df.index.unique()

        data_cols = list(set(merged_df.columns).difference(id_cols))
        num_data_cols = len(data_cols)
        num_plays = len(play_index)
        routes_per_play = merged_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=False).iloc[0]
        assert (merged_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=False) == routes_per_play).all(), \
        "Different values across player dimension"
        
        self.data = np.zeros((num_plays, routes_per_play, num_data_cols), dtype=np.float32)

        for i, (gameid, playid, mirrored) in enumerate(play_index): 

            data = merged_df.loc[(gameid, playid, mirrored), data_cols].to_numpy(dtype=np.float32)
            self.data[i, ...] = data

        self.torch_data = torch.Tensor(self.data)
        self.index = play_index

        return
    
    

    def __len__(self): 
        return self.data.shape[0]


    def __getitem__(self, idx: int): 
        return self.torch_data[idx]



class ManZoneDataset(NflDataset): 
    name = "ManZoneDataset"

    def __init__(
        self, 
        merged_df: pd.DataFrame,
        play_df: pd.DataFrame
    ):
        super().__init__(merged_df)

        play_df = pd.get_dummies(play_df.set_index(['gameId', 'playId'])['pff_manZone'])
        target = np.zeros( (self.data.shape[0], len(play_df.columns)) )

        for i, (gameid, playid, mirrored) in enumerate(self.index):
            target[i, ...] = play_df.loc[(gameid, playid)].to_numpy()

        self.target = torch.Tensor(target)
        assert self.target.shape[0] == self.data.shape[0]

        
    def __getitem__(self, idx: int):

        return self.data[idx], self.target[idx]




class MaskedNflDataset(NflDataset): 

    def __init__(
        self, 
        tracking_df: pd.DataFrame,
        player_play_df: pd.DataFrame,
        play_df: pd.DataFrame, 
        player_df: pd.DataFrame, 
    ):
        super().__init__(tracking_df, player_play_df, play_df, player_df)        



        return 

    
    

def load_dataset(filepath: Path) -> NflDataset:
    dataset = torch.load(filepath, weights_only=False)
    return dataset


def write_dataset(dataset: Dataset, filepath: Path) -> NflDataset:
    torch.save(dataset, filepath)
    
    