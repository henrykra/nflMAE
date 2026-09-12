"""Functions for perparing datasets for modeling"""
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
import torch
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)
device = torch.device('mps')


def train_val_test_split_df(df: pd.DataFrame, train_pct: float, val_pct: float, test_pct: float) -> tuple[pd.Index, pd.Index, pd.Index]:

    logger.info(f"making splits at {train_pct}, {val_pct}, {test_pct}")

    index = df.index.unique() # split unique df index
    
    cats = np.random.choice(
        a=[0, 1, 2], 
        size=len(index), 
        replace=True, 
        p=[train_pct, val_pct, test_pct]
    )

    split = np.stack([index.to_numpy(), cats], axis=-1)

    train_idx = split[split[:, -1] == 0, 0]
    val_idx = split[split[:, -1] == 1, 0]
    test_idx = split[split[:, -1] == 2, 0]

    return train_idx, val_idx, test_idx


def train_val_split_df(df: pd.DataFrame, train_pct: float, val_pct: float) -> tuple[pd.Index, pd.Index]:

    logger.info(f"making splits at {train_pct}, {val_pct}")

    index = df.index.unique() # split unique df index
    
    cats = np.random.choice(
        a=[0, 1], 
        size=len(index), 
        replace=True, 
        p=[train_pct, val_pct]
    )

    split = np.stack([index.to_numpy(), cats], axis=-1)

    train_idx = split[split[:, -1] == 0, 0]
    val_idx = split[split[:, -1] == 1, 0]

    return train_idx, val_idx




class FrameDataset(torch.utils.data.Dataset):
    """Dataset where each sample represents a frame of an nfl play"""

    def __init__(self, merged_df: pd.DataFrame): 
        
        self.data = {}
        position_cols = [col for col in merged_df.columns if col.startswith('position')]
        self.cols = ['x', 'y', 'ox', 'oy', 'vx', 'vy', 'a', 'offense', 'defense', 'height_z', 'weight_z', 'nflId'] # + position_cols

        isna = merged_df[self.cols].isna()
        null_columns = isna.columns[isna.any()].tolist()
        player_mask = merged_df['nflId'] != -1
        player_null_columns = isna.columns[isna.loc[player_mask].any()].tolist()

        logger.warning(f"Found columns with null values: {null_columns}")
        logger.warning(f"Found non-football columns with null values: {player_null_columns}")
        merged_df = merged_df.fillna(0)

        merged_df = merged_df.sort_values(['gameId', 'playId', 'mirrored', 'frameId', 'nflId'])

        values = merged_df[self.cols].to_numpy(dtype=np.float32)

        groups = merged_df.groupby(['gameId', 'playId', 'mirrored', 'frameId'], sort=False).indices
        # data (gameId, playid, mirrored, frameid): 23x? array
        self.data = {key: values[pos, :-1] for key, pos in groups.items()}
        self.pids = {key: values[pos, -1] for key, pos in groups.items()}

        self.idx = list(self.data.keys())  
        logger.info(f"{len(self.idx)} frames loaded")
        logger.info(f"Loaded {self.channels()} channels")
        return

    
    def __getitem__(self, idx: int): 
        key = self.idx[idx]

        return {
            "feat": torch.from_numpy(self.data[key]), 
            "coord": torch.from_numpy(self.data[key][:, :2]) # x and y only
        }
        
    

    def __len__(self): 
        return len(self.idx)
    

    def channels(self): 
        return len(self.cols) - 1


class MultiFrameDataset(FrameDataset):
    frame_dif = 15

    def __init__(self, merged_df: pd.DataFrame):
        super().__init__(merged_df)
        self.min_frame_ids = merged_df.groupby(['gameId', 'playId', 'mirrored'])['frameId'].min()

    def __getitem__(self, idx: int):
        key = self.idx[idx]
        frameid = key[-1]

        p = self.data[key].shape[0]

        # x frames before, or the first frame
        newkey = (*key[:-1], frameid - self.frame_dif)
        min_frameid = self.min_frame_ids.loc[newkey[:-1]]
        if newkey[-1] < min_frameid:
            newkey = (*key[:-1], min_frameid)

        return {
            "feat": torch.cat( (torch.from_numpy(self.data[newkey]), torch.from_numpy(self.data[key])), dim=0),
            "coord": torch.cat( (torch.from_numpy(self.data[newkey][:, :2]), torch.from_numpy(self.data[key][:, :2])), dim=0), 
            "fids": torch.repeat_interleave(torch.Tensor([newkey[-1], frameid]), repeats=p),
            "pids": torch.cat( (torch.from_numpy(self.pids[newkey]), torch.from_numpy(self.pids[key])), dim=0),
            "weights": torch.Tensor([(frameid - newkey[-1]) / self.frame_dif]).float()
        }




class TacklePosDataset(FrameDataset):

    def __init__(self, merged_df: pd.DataFrame):
        """Update framedataset superclass to remove frames from data/idx that don't have a tackle"""
        
        super().__init__(merged_df)
        tackle_positions = self._get_tackle_positions(merged_df)
        tackle_positions.sort_index(inplace=True)

        assert list(tackle_positions.columns) == ['x', 'y']

        # dict (gameid, playid, mirrored, frameid): 23x2 tensor
        self.target = {}

        frame_idx_df = pd.DataFrame(self.idx, columns=['gameId', 'playId', 'mirrored', 'frameId'])

        # merge the dataframes so that each frame finds the tackle from its play
        merged = (
            frame_idx_df
            .merge(
                tackle_positions.reset_index(),
                how='inner',
                on=['gameId', 'playId', 'mirrored']
            )
            .merge(
                merged_df.loc[merged_df['isBallCarrier'] == 1, ['gameId', 'playId', 'mirrored', 'frameId', 'x', 'y']],
                how='left',
                on=['gameId', 'playId', 'mirrored', 'frameId'],
                suffixes=('', '_bc')
            )
        )

        kept_keys = list(merged[['gameId', 'playId', 'mirrored', 'frameId']].itertuples(index=False, name=None))
        target_values = merged[['x', 'y']].to_numpy(dtype=np.float32)
        bc_locations = merged[['x_bc', 'y_bc']].to_numpy(dtype=np.float32)
        merged['isBallCarrier'] = (merged[['x', 'y']] == merged[['x_bc', 'y_bc']].rename(columns={'x_bc': 'x', 'y_bc': 'y'})).all(axis=1)
        bc_mask = merged[['isBallCarrier']].to_numpy(dtype=np.float32)

        self.target = dict(zip(kept_keys, target_values))
        self.bc = dict(zip(kept_keys, bc_locations))
        self.bc_mask = dict(zip(kept_keys, bc_mask))
        self.data = {k: self.data[k] for k in kept_keys}
        logger.info(f"deleting {len(self.idx) - len(kept_keys)} frames from plays without tackles")
        self.idx = kept_keys

        logger.info(f"{len(self.idx)} frames loaded")

        return 
        
        

    def __getitem__(self, idx: int):
        key = self.idx[idx]

        return {
            "feat": torch.from_numpy(self.data[key]), # drop nflId
            "coord": torch.from_numpy(self.data[key][:, :2]), # x and y only,
            "target": torch.from_numpy(self.target[key]), 
            "bc": torch.from_numpy(self.bc[key]), # ball carrier x/y at the given frame
            "bc_mask": torch.from_numpy(self.bc_mask[key]) # bool is ballcarrier for all players
        }



    def _get_tackle_positions(self, merged_df: pd.DataFrame) -> pd.DataFrame: 
        # football location at tackle event for each play

        merged_df = merged_df.reset_index()
        n_plays = merged_df.set_index(['gameId', 'playId', 'mirrored']).index.nunique()

        tackles = (
            merged_df
            .loc[merged_df['event'].isin(['tackle', 'out_of_bounds', 'touchdown', 'qb_slide', 'fumble'])]
            .groupby(['gameId', 'playId', 'mirrored'])
            .tail(1)
            .set_index(['gameId', 'playId', 'mirrored', 'frameId'])
        )
        ball_carrier_locations = merged_df.loc[merged_df['isBallCarrier'] == 1].set_index(['gameId', 'playId', 'mirrored', 'frameId'])

        tackle_locations = ball_carrier_locations.loc[tackles.index.unique()]

        logger.info(f"{len(tackle_locations)} / {n_plays} plays with tackles loaded")

        return tackle_locations.droplevel('frameId')[['x', 'y']].copy()


    def target_channels(self):
        return 2 # x, y tackle position


    def get_loss_fn(self): 

        return torch.nn.MSELoss()






def run_datset_pipeline(): 
    ...