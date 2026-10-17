"""Functions for perparing datasets for modeling"""
import logging

import pandas as pd
import numpy as np
import torch


logger = logging.getLogger(__name__)
device = torch.device("mps")


def train_val_test_split_df(df: pd.DataFrame, train_pct: float, val_pct: float, test_pct: float) -> tuple[pd.Index, pd.Index, pd.Index]:

    logger.info(f"making splits at {train_pct}, {val_pct}, {test_pct}")

    index = df.index.unique() # split unique df index

    # assign train/val/test label to each index
    cats = np.random.choice(
        a=[0, 1, 2], 
        size=len(index), 
        replace=True, 
        p=[train_pct, val_pct, test_pct]
    )

    # get label from index
    split = np.stack([index.to_numpy(), cats], axis=-1)

    # prepare indexes as filters
    train_idx = split[split[:, -1] == 0, 0]
    val_idx = split[split[:, -1] == 1, 0]
    test_idx = split[split[:, -1] == 2, 0]

    return train_idx, val_idx, test_idx


def train_val_split_df(df: pd.DataFrame, train_pct: float, val_pct: float) -> tuple[pd.Index, pd.Index]:

    logger.info(f"making splits at {train_pct}, {val_pct}")

    index = df.index.unique() # split unique df index

    # assign train/val/test label to each index 
    cats = np.random.choice(
        a=[0, 1], 
        size=len(index), 
        replace=True, 
        p=[train_pct, val_pct]
    )

    # get label from index
    split = np.stack([index.to_numpy(), cats], axis=-1)

    # prepare indexes as filters
    train_idx = split[split[:, -1] == 0, 0]
    val_idx = split[split[:, -1] == 1, 0]

    return train_idx, val_idx


class FrameDataset(torch.utils.data.Dataset):
    """Dataset where each sample represents a frame of an nfl play"""

    def __init__(self, merged_df: pd.DataFrame, extra_cols: dict[str, list[str]]={}):
        position_cols = [col for col in merged_df.columns if col.startswith('position')]
        id_cols = ['gameId', 'playId', 'mirrored', 'frameId']
        self.standard_cols = ['x', 'y', 'ox', 'oy', 'vx', 'vy', 'a', 'offense', 'defense', 'height_z', 'weight_z', 'nflId']
        # specify columns for training input

        self.cols = self.standard_cols + [col for name, cols in extra_cols.items() for col in cols]

        # check for NAs before fillna
        isna = merged_df[self.cols].isna()
        null_columns = isna.columns[isna.any()].tolist()
        player_mask = merged_df['nflId'] != -1
        player_null_columns = isna.columns[isna.loc[player_mask].any()].tolist()

        logger.warning(f"Found columns with null values: {null_columns}")
        logger.warning(f"Found non-football columns with null values: {player_null_columns}")
        merged_df = merged_df.fillna(0)

        merged_df = merged_df.sort_values(id_cols + ['nflId'])

        values = merged_df[self.cols].to_numpy(dtype=np.float32)

        groups = merged_df.groupby(id_cols, sort=False).indices
        # data (gameId, playid, mirrored, frameid): 23x? array
        self.data = {key: values[pos, :len(self.standard_cols) - 1] for key, pos in groups.items()}
        self.pids = {key: values[pos, len(self.standard_cols) - 1] for key, pos in groups.items()}
        offset = len(self.standard_cols)
        for col_group, cols in extra_cols.items():
            n_cols = len(cols)
            setattr(self, col_group, {key: values[pos, offset:offset+n_cols] for key, pos in groups.items()})
            offset += n_cols

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
        return len(self.standard_cols) - 1


class MultiFrameDataset(FrameDataset):
    """Dataset representing a frame from and NFL play, and a reference frame before it"""
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
        if newkey[-1] < min_frameid or newkey not in self.data:
            newkey = (*key[:-1], min_frameid)

        return {
            "feat": torch.from_numpy(self.data[key]),
            "reference_feat": torch.from_numpy(self.data[newkey]),
            "coord": torch.from_numpy(self.data[key][:, :2]),
            "reference_coord": torch.from_numpy(self.data[newkey][:, :2]),
            "fids": torch.repeat_interleave(torch.Tensor([newkey[-1], frameid]), repeats=p),
            "pids": torch.cat( (torch.from_numpy(self.pids[newkey]), torch.from_numpy(self.pids[key])), dim=0),
            "weights": torch.Tensor([(frameid - newkey[-1]) / self.frame_dif]).float()
        }




class TacklePosDataset(FrameDataset):
    """Frame dataset with target location of ballcarrier during a tackle event"""
    # TODO: fix __init__

    def __init__(self, merged_df: pd.DataFrame):
        """Update framedataset superclass to remove frames from data/idx that don't have a tackle"""
        merged_df = merged_df.reset_index()

        self.data = {}
        position_cols = [col for col in merged_df.columns if col.startswith("position")]
        self.cols = ['x', 'y', 'ox', 'oy', 'vx', 'vy', 'a', 'offense', 'defense', 'height_z', 'weight_z', 'isBallCarrier', 'nflId'] # + position_cols

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
        self.data = {key: values[pos, :11] for key, pos in groups.items()}
        self.pids = {key: values[pos, 12] for key, pos in groups.items()}
        self.bc = {key: values[pos, 11] for key, pos in groups.items()}

        self.idx = list(self.data.keys())  
        logger.info(f"{len(self.idx)} frames loaded")
        logger.info(f"Loaded {self.channels()} channels")

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

        self.target = dict(zip(kept_keys, target_values))
        self.bc_loc = dict(zip(kept_keys, bc_locations))
        self.data = {k: self.data[k] for k in kept_keys}
        logger.info(f"deleting {len(self.idx) - len(kept_keys)} frames from plays without tackles")
        self.idx = kept_keys

        logger.info(f"{len(self.idx)} frames loaded")

        return 
        
        

    def __getitem__(self, idx: int):
        key = self.idx[idx]

        return {
            "feat": torch.from_numpy(self.data[key]), # drop nflId
            # "coord": torch.from_numpy(self.data[key][:, :2]), # x and y only,
            "target": torch.from_numpy(self.target[key]), 
            "bc": torch.from_numpy(self.bc_loc[key]), # ball carrier x/y at the given frame
            "target_mask": torch.from_numpy(self.bc[key]).bool() # bool is ballcarrier for all players
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
        return TacklePosLoss()

    def channels(self): 
        return len(self.cols) - 2

    @staticmethod
    def eval_statistic(pred: torch.Tensor, minibatch: dict[torch.Tensor]):
        target = minibatch['target'].detach().cpu()
        de = (((target - pred.detach().cpu()) ** 2).sum(axis=1) ** .5)
        return de


class TacklePosLoss(torch.nn.Module):
    def __init__(self): 
        super().__init__()
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, minibatch: dict):
        return self.loss(pred, minibatch['target'])


class ManZoneDataset(FrameDataset):
    """Dataset with man/zone/other label target for each play"""
    def __init__(self, merged_df, extra_cols: list=[]):

        self.man_zone_cols = [col for col in merged_df.columns if col.startswith('pff_manZone')]
        # filter to where man_zone columns have a value
        merged_df['man_zone_sum'] = merged_df[self.man_zone_cols].sum(axis=1)
        merged_df = merged_df.loc[merged_df['man_zone_sum'] > 0].drop(columns='man_zone_sum')
        super().__init__(merged_df, {'man_zone': self.man_zone_cols})

        # find the first frame for each play
        self.min_frame_ids = merged_df.groupby(['gameId', 'playId', 'mirrored'])['frameId'].min()

        return 

    def __getitem__(self, idx: int):
        key = self.idx[idx]

        key = self.idx[idx]
        frameid = key[-1]

        p = self.data[key].shape[0]

        # take the earliest reference frame
        min_frameid = self.min_frame_ids.loc[key[:-1]]
        newkey = (*key[:-1], min_frameid)

        return {
            "feat": torch.cat( (torch.from_numpy(self.data[newkey]), torch.from_numpy(self.data[key])), dim=0),
            "coord": torch.cat( (torch.from_numpy(self.data[newkey][:, :2]), torch.from_numpy(self.data[key][:, :2])), dim=0),
            "fids": torch.repeat_interleave(torch.Tensor([newkey[-1], frameid]), repeats=p),
            "pids": torch.cat( (torch.from_numpy(self.pids[newkey]), torch.from_numpy(self.pids[key])), dim=0),
            "target": torch.from_numpy(self.man_zone[key][0]),
        }

    def num_types(self): 
        return len(self.man_zone_cols)

    @staticmethod
    def eval_statistic(pred: torch.Tensor, minibatch: dict[str, torch.Tensor]):
        # find accuracy of classification
        bs = pred.shape[0]
        pred_cls = pred.detach().cpu().argmax(-1)
        gt_cls = minibatch['target'].detach().cpu().argmax(-1)

        return (pred_cls == gt_cls).float()

    def get_loss_fn(self): 
        return ManZoneLoss()
    

class ManZoneLoss(torch.nn.Module):
    def __init__(self): 
        super().__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, minibatch: dict[str, torch.Tensor]):

        
        return self.loss_fn(pred, minibatch['target'].argmax(-1).squeeze())


class RouteDataset(FrameDataset):
    def __init__(self, merged_df: pd.DataFrame):
        self.route_cols = [col for col in merged_df.columns if col.startswith('routeRan')]

        # filter plays without stop point

        # filter before stop point

        # filter routes without routeRan
        routes_ran = merged_df.groupby(['gameId', 'playId'])['wasRunningRoute'].sum() > 0
        idx = routes_ran.index[routes_ran]
        merged_df = merged_df.loc[idx].copy()

        super().__init__(merged_df, extra_cols={'running': ['wasRunningRoute'], 'routes': self.route_cols})

        # find the first frame for each play
        self.min_frame_ids = merged_df.groupby(['gameId', 'playId', 'mirrored'])['frameId'].min()
        return 
        
    def __getitem__(self, idx):
        key = self.idx[idx]
        frameid = key[-1]

        p = self.data[key].shape[0]

        # take the earliest reference frame
        min_frameid = self.min_frame_ids.loc[key[:-1]]
        newkey = (*key[:-1], min_frameid)

        return {
            "feat": torch.cat( (torch.from_numpy(self.data[newkey]), torch.from_numpy(self.data[key])), dim=0),
            "coord": torch.cat( (torch.from_numpy(self.data[newkey][:, :2]), torch.from_numpy(self.data[key][:, :2])), dim=0),
            "fids": torch.repeat_interleave(torch.Tensor([newkey[-1], frameid]), repeats=p),
            "pids": torch.cat( (torch.from_numpy(self.pids[newkey]), torch.from_numpy(self.pids[key])), dim=0),
            "running_route": torch.from_numpy(self.running[key]),
            "route": torch.from_numpy(self.routes[key]),
        }

    def num_routes(self): 
        return len(self.route_cols)

    def get_loss_fn(self): 
        return RouteRunLoss()

    @staticmethod
    def eval_statistic(pred: torch.Tensor, minibatch: dict[str, torch.Tensor]):
        """Calculate percentage of routes correctly predicted"""
        n_players = pred.shape[1] // 2
        last_frame = pred[:, -n_players:].detach().cpu()
        pred_route = last_frame[..., 1:].argmax(dim=-1)

        running_route = minibatch['running_route'].detach().cpu().squeeze(-1).bool()

        gt_route = minibatch['route'].detach().cpu().argmax(dim=-1)

        correct = (gt_route == pred_route) & running_route
        n_runners = running_route.sum(dim=1)
        return correct.sum(dim=1).float() / n_runners



class RouteRunLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.route_type_loss = torch.nn.CrossEntropyLoss()
        self.route_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, minibatch: dict[str, torch.Tensor]):
        # predict on only the players in the last frame

        n_players = pred.shape[1] // 2 # round down from cls token

        last_frame = pred[:, -n_players:]

        loss1 = self.route_loss(last_frame[..., 0], minibatch['running_route'].squeeze(-1))
        
        # loss only over the actual route runners
        loss2 = self.route_type_loss(
            last_frame[minibatch['running_route'].squeeze(-1).bool(), 1:],
              minibatch['route'][minibatch['running_route'].bool().squeeze(-1)])

        # weigh route type prediction higher
        return .5 * loss1 + loss2
        
