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


@dataclass
class Frame:
    feats: torch.Tensor
    coords: torch.Tensor
    relative_coords: Optional[torch.Tensor]=None
    target: Optional[torch.Tensor]=None


    def to(self, device: torch.device):
        self.feats = self.feats.to(device)
        self.coords = self.coords.to(device)

        if self.relative_coords is not None:
            self.relative_coords = self.relative_coords.to(device)

        if self.target is not None: 
            self.target = self.target.to(device)

        return self
    

    @staticmethod
    def collate_fn(frames: list["Frame"]):

        if not all([frame.relative_coords is not None for frame in frames]): 
            rel_coords = None
        else: 
            rel_coords = torch.stack([frame.relative_coords for frame in frames])

        if not all([frame.target is not None for frame in frames]): 
            targets = None
        else: 
            targets = torch.stack([frame.target for frame in frames])
        

        newframe = Frame(
            torch.stack([frame.feats for frame in frames]),
            torch.stack([frame.coords for frame in frames]),
            rel_coords,
            targets
        )
        return newframe


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



class FrameDataset(torch.utils.data.Dataset):
    """Dataset where each sample represents a frame of an nfl play"""

    def __init__(self, merged_df: pd.DataFrame): 
        
        self.data = {}
        position_cols = [col for col in merged_df.columns if col.startswith('position')]
        self.cols = ['x', 'y', 'ox', 'oy', 'vx', 'vy', 'a', 'offense', 'defense', 'height_z', 'weight_z'] + position_cols

        player_df = merged_df.loc[merged_df['nflId'] != -1]
        null_columns = [col for col in self.cols if merged_df[col].isna().any()]
        player_null_columns = [col for col in self.cols if player_df[col].isna().any()]

        logger.warning(f"Found columns with null values: {null_columns}")
        logger.warning(f"Found non-football columns with null values: {player_null_columns}")
        merged_df = merged_df.fillna(0)

        for group, df in merged_df.groupby(['gameId', 'playId', 'mirrored', 'frameId']): 
            self.data[group] = df[self.cols].to_numpy(dtype=np.float32)

        self.idx = list(self.data.keys())
        logger.info(f"{len(self.idx)} frames loaded")
        logger.info(f"Loaded {self.channels()} channels")
        return

    
    def __getitem__(self, idx: int): 
        key = self.idx[idx]
        
        return Frame(torch.Tensor(self.data[key]), torch.Tensor(self.data[key][:, :2]))
    

    def __len__(self): 
        return len(self.idx)
    

    def channels(self): 
        return len(self.cols)



class TacklePosDataset(FrameDataset):

    def __init__(self, merged_df: pd.DataFrame):
        
        super().__init__(merged_df)
        tackle_positions = self._get_tackle_positions(merged_df)
        tackle_positions.sort_index(inplace=True)


        assert list(tackle_positions.columns) == ['x', 'y']
        self.target = {}

        delete_keys = []
        for gameid, playid, mirrored, frameid in self.data.keys(): 
            if (gameid, playid, mirrored) not in tackle_positions.index: 
                delete_keys.append((gameid, playid, mirrored, frameid))
                continue
            
            position = tackle_positions.loc[(gameid, playid, mirrored)]


            if len(position.shape) > 1 and position.shape[0] > 1: 
                logger.warning("play found with multiple tackle positions")
                logger.warning(f"{len(position)}")
                position = position.iloc[0].copy()


            self.target[(gameid, playid, mirrored, frameid)] = position.to_numpy().squeeze()

        
        # remove frames w/ no tackle, can't modify dict during iteration
        logging.info(f"deleting {len(delete_keys)} frames from plays without tackles")
        for key in delete_keys: 
            del self.data[key]

        self.idx = list(self.data.keys())
        logger.info(f"{len(self.idx)} frames loaded")

        return 
        
        

    def __getitem__(self, idx: int):
        key = self.idx[idx]
        frame = Frame(
            torch.Tensor(self.data[key]), 
            torch.Tensor(self.data[key][:, :2]), 
            target=torch.Tensor(self.target[key])
        )

        return frame



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
        ball_locations = merged_df.loc[~(merged_df['offense'] | merged_df['defense'])].set_index(['gameId', 'playId', 'mirrored', 'frameId'])

        tackle_locations = ball_locations.loc[tackles.index.unique()]

        logger.info(f"{len(tackle_locations)} / {n_plays} plays with tackles loaded")

        return tackle_locations.droplevel('frameId')[['x', 'y']].copy()


    def target_channels(self):
        return 2 # x, y tackle position

























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


        merged_df = merged_df.reset_index().sort_values(['gameId', 'playId', 'mirrored', 'nflId'])

        self.index = pd.MultiIndex.from_frame(merged_df[['gameId', 'playId', 'mirrored']].drop_duplicates())

        data_cols = list(set(merged_df.columns).difference(id_cols))
        num_data_cols = len(data_cols)
        num_plays = len(self.index)
        routes_per_play = merged_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=False).iloc[0]
        assert (merged_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=False) == routes_per_play).all(), \
        "Different values across player dimension"
        
        self.torch_data = torch.tensor(
            merged_df[data_cols].to_numpy(dtype=np.float32)
            .reshape(num_plays, routes_per_play, num_data_cols),
            dtype=torch.float32
        )

        return
    
    

    def __len__(self): 
        return self.torch_data.shape[0]


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
        target = np.zeros( (self.torch_data.shape[0], len(play_df.columns)) )

        for i, (gameid, playid, mirrored) in enumerate(self.index):
            target[i, ...] = play_df.loc[(gameid, playid)].to_numpy()

        self.target = torch.Tensor(target)
        assert self.target.shape[0] == self.torch_data.shape[0]

        
    def __getitem__(self, idx: int):

        return self.torch_data[idx], self.target[idx]




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
    
    