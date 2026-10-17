"""Common functions for loading and preparing Big Data Bowl 2025 data for modeling."""
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import itertools
import sys
import time
from scipy.interpolate import CubicSpline

from nflMAE.data import bdb_common

logger = logging.getLogger(__name__)

MIN_WEEK = 1
MAX_WEEK = 9


def load_player_play_data(
        folder: Optional[Path]=None,
) -> pd.DataFrame:
    
    if folder and folder.exists():
        logger.info(f"reading play data from {folder}")
        path = Path(folder, 'player_play.csv')

    return pd.read_csv(path)


### SIMPLE CLEANING

def football_relative_positions(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    # move all positions relative to the football's location
    ball = (
        tracking_df
        .loc[
            (tracking_df['club'] == 'football'),
            ['gameId', 'playId', 'frameId', 'x', 'y']
        ]
    )

    tracking_df = tracking_df.merge(
        ball, 
        how='left', 
        on=('gameId', 'playId', 'frameId'), 
        suffixes=('', '_ball')
    )

    tracking_df['dis_to_ball'] = np.sqrt(
        (tracking_df['x'] - tracking_df['x_ball'])**2 + (tracking_df['y'] - tracking_df['y_ball'])**2
    )

    return tracking_df



def filter_play_end_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    # create a stop point variable to track frames after the play
    tracking_df['stop_point'] = pd.NA
    stop_points = ['tackle', 'out_of_bounds', 'fumble', 'qb_slide', 'touchdown', 'safety', 'qb_sack', 'pass_outcome_incomplete']
    logger.info(f"tracking stopping points identified: {stop_points}")
    tracking_df.loc[tracking_df['event'].isin(stop_points), 'stop_point'] = True 
    # fill stopping points forward to filter out moments after a defined route stop
    tracking_df['stop_point'] = tracking_df.groupby(['gameId', 'playId', 'nflId'])['stop_point'].ffill()

    # filter only after the event
    tracking_df.loc[tracking_df['event'].isin(stop_points), 'stop_point'] = None

    return tracking_df.loc[tracking_df['stop_point'].isna()].drop(columns='stop_point') # remove redundant column


def filter_route_end_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    # potentially complex logic to find when a route runner's designed route has ended
    ...


def filter_football_tracking_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    logger.info("filtering football from tracking frames")
    return tracking_df.loc[tracking_df['club'] != 'football']


### TRANSFORMS

def clean_player_play_data(player_play_df: pd.DataFrame, play_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: 
    # create offense and defense columns from possesionTeam and club
    merged = player_play_df[['gameId', 'playId', 'club']].merge(play_df[['gameId', 'playId', 'possessionTeam', 'defensiveTeam']].copy(), how='left', on=('gameId', 'playId'))

    player_play_df['offense'] = (merged['club'] == merged['possessionTeam']).astype('float32')
    player_play_df['defense'] = (merged['club'] == merged['defensiveTeam']).astype('float32')

    logger.debug(f"{player_play_df['offense'].isna().sum()} NA \'offense\' values")
    
    return player_play_df, play_df


def interpolate_tracking_frames(tracking_df: pd.DataFrame, interp_frames: int=40) -> pd.DataFrame: 
    # idea: constant number of tracking frames per play by interpolation
    play_index = tracking_df.set_index(['gameId', 'playId']).sort_index()

    num_obs = interp_frames
    interpolated_columns = ['x', 'y', 'ox', 'oy', 'vx', 'vy', 'a']
    ids = play_index.index.unique()
    num_plays = len(ids)
    routes_per_play = tracking_df.groupby(['gameId', 'playId'])['nflId'].nunique(dropna=True).max()
    id_cols = ['gameId', 'playId', 'nflId', 'frameId']

    print(f"interpolating {len(interpolated_columns)} columns of data")
    print(f"columns: {interpolated_columns}")
    # array for interpolated movement
    movement_arr = np.zeros( (num_plays, routes_per_play, num_obs, len(interpolated_columns)), dtype=np.float32)
    nfl_ids = np.empty( (num_plays, routes_per_play), dtype=np.float32 )

    for i, (gameid, playid) in enumerate(ids): 
        
        play: pd.DataFrame = play_index.loc[gameid, playid]
        
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
            nfl_ids[i, ii] = nflid



    # create new tracking dataframe
    tracking_df_data = movement_arr.reshape((-1, len(interpolated_columns)))
    id_arr = np.stack([
        np.repeat(ids.get_level_values(0), num_obs * routes_per_play),
        np.repeat(ids.get_level_values(1), num_obs * routes_per_play),
        nfl_ids.flatten().repeat(num_obs),
        np.tile(np.arange(num_obs), routes_per_play * num_plays)
    ], axis=-1)


    tracking_df = pd.DataFrame(
        data = np.concat([id_arr, tracking_df_data], axis=-1), 
        columns=id_cols + interpolated_columns
    )

    return tracking_df


def player_relative_positions(tracking_df: pd.DataFrame, player_play_df: pd.DataFrame):
    """For each player/play, make the player's position relative to their previous frame,
    and add start postions for each player/play
    """
    
    starts = (
        tracking_df
        .groupby(['gameId', 'playId', 'nflId'])
        [['x', 'y']]
        .first()
        .rename(columns={'x': 'x_start', 'y': 'y_start'})
    )

    tracking_df[['x', 'y']] = tracking_df.groupby(['gameId', 'playId', 'nflId'])[['x', 'y']].transform(lambda x: x - x.iloc[0])

    # join starts with player_play_df
    player_play_df = player_play_df.merge(
        starts, 
        how='left', 
        on=('gameId', 'playId', 'nflId')
    )

    return tracking_df, player_play_df



### MERGE

def merge_dataframes(
    tracking_df: pd.DataFrame,
    player_play_df: pd.DataFrame,
    play_df: pd.DataFrame,
    player_df: pd.DataFrame,
): 
    # select columns from dataframes early
    tracking_cols = ['gameId', 'playId', 'nflId', 'frameId', 'mirrored', 'x', 'y', 'ox', 'oy', 'vx', 'vy', 'a', 'event']
    tracking_df = tracking_df[tracking_cols].copy()

    player_play_cols = ['gameId', 'playId', 'nflId', 'mirrored', 'offense', 'defense', 'wasRunningRoute', 'routeRan']
    player_play_df = player_play_df[player_play_cols].copy()

    player_cols = ['nflId', 'height_z', 'weight_z', 'position']
    player_df = player_df[player_cols].copy()

    play_cols = ['gameId', 'playId', 'mirrored', 'pff_manZone']
    play_df = play_df[play_cols].copy()

    merged = (
        tracking_df
        .merge(
            right=player_play_df, 
            how='left',
            on=['gameId', 'playId', 'nflId', 'mirrored']
        )
        .merge(
            right=play_df,
            how='left',
            on=['gameId', 'playId', 'mirrored']
        )
        .merge(
            right=player_df,
            how='left',
            on=['nflId']
        )
    )

    merged = pd.get_dummies(merged, columns=['position', 'pff_manZone', 'routeRan'])

    logger.debug(f"merged_df columns: {merged.columns}")

    logger.debug(f"{len(merged.loc[merged['offense'].isna(), 'nflId'].unique())} NA \'offense\' values")

    return merged
    


def merge_dataframes_wide(
    tracking_df: pd.DataFrame,
    player_play_df: pd.DataFrame, 
    play_df: pd.DataFrame,
    player_df: pd.DataFrame, 
    tracking_cols: list[str],
    player_play_cols: list[str],
    play_cols: list[str],
    player_cols: list[str]
) -> pd.DataFrame:
    
    tracking_df = tracking_df[tracking_cols].copy()

    play_lens = tracking_df.groupby(['gameId', 'playId'])['frameId'].max()
    max_frameid = int(play_lens.mean() + 2 * play_lens.std())
    
    new_index = itertools.product(tracking_df.set_index(['gameId', 'playId', 'nflId', 'mirrored']).index.unique().tolist(), range(max_frameid))
    new_index = [(*p, f) for p, f in new_index]

    new_index = pd.MultiIndex.from_tuples(new_index, names=['gameId', 'playId', 'nflId', 'mirrored', 'frameId'])

    tracking_df = tracking_df.set_index(['gameId', 'playId', 'nflId', 'mirrored', 'frameId']).reindex(new_index).fillna(0)
    tracking_df = tracking_df.reset_index()
    tracking_df['nflId'] = tracking_df['nflId'].fillna(-1)
    tracking_df = tracking_df.set_index(['gameId', 'playId', 'nflId', 'mirrored'])
    wide_tracking_df = tracking_df.pivot_table(index=tracking_df.index.names, columns='frameId').pipe(
        lambda d: d.set_axis([f'{col}_{id_}' for col, id_ in d.columns], axis=1)
    ).reset_index()


    merged_df = pd.get_dummies(
        wide_tracking_df
        .merge(player_play_df[player_play_cols], how='left', on=('gameId', 'playId', 'nflId'))
        .merge(play_df[play_cols], how='left', on=('gameId', 'playId'))
        .merge(player_df[player_cols], how='left', on=('nflId'))
    ).fillna(0)


    return merged_df


def run_data_pipeline(data_folder: Path, out_folder: Path, minimal: bool=False, weeks: list[int]| None=None, df_name: str | None=None):
    logger.info("Running data pipeline")
    logger.info(f"{data_folder}, minimal={minimal}")
    logger.info(f"saving to {out_folder}")

    merged_dfs = []

    if weeks:
        week_iter = weeks
    else: 
        week_iter = range(MIN_WEEK, MAX_WEEK + 1)

    for week in week_iter:
        if week < MIN_WEEK or week > MAX_WEEK:
            logger.error(f"invalid week {week}, skipping")
            continue

        if minimal:
            tracking = bdb_common.load_csv_data([Path("../2025_minimal_tracking.csv")])[0]
        else: 
            tracking = bdb_common.load_tracking_data(weeks=week, folder=data_folder)

        players = bdb_common.load_player_data(data_folder)
        plays = bdb_common.load_play_data(data_folder)
        player_plays = load_player_play_data(data_folder)

        # clean
        tracking = bdb_common.clean_tracking_angles(tracking)
        tracking = bdb_common.standardize_tracking_plays(tracking, plays)
        # tracking = football_relative_positions(tracking)
        players = bdb_common.clean_player_data(players)

        # filter
        tracking = bdb_common.filter_play_start_frames(tracking)
        tracking = filter_play_end_frames(tracking)
        # tracking = filter_football_tracking_frames(tracking)
        tracking, player_plays = bdb_common.filter_outlier_plays(tracking, player_plays)
        tracking = bdb_common.filter_outlier_frames(tracking)

        routes_per_play = tracking.groupby(['gameId', 'playId'])['nflId'].nunique(dropna=False).iloc[0]
        logger.debug(f"{routes_per_play} routes per play")

        # transform
        plays, tracking = bdb_common.find_play_invariant_columns(plays, tracking)
        player_plays, tracking = bdb_common.find_player_play_invariant_columns(player_plays, tracking)
        player_plays, plays = clean_player_play_data(player_plays, plays)
        # tracking, player_plays = player_relative_positions(tracking, player_plays)

        # mirror
        tracking, player_plays, plays = bdb_common.mirror_tracking_data(tracking, player_plays, plays)

        # merge
        merged = merge_dataframes(tracking, player_plays, plays, players).reindex()
        assert (merged.groupby(['gameId', 'playId'])['nflId'].nunique(dropna=False) == routes_per_play).all()

        merged_dfs.append(merged)
        if minimal: 
            break

    
    merged = pd.concat(merged_dfs)

    if not df_name: 
        df_name = ("bdb2025" + ("" if not minimal else "_minimal") + ".parquet")

    # save
    merged.to_parquet(
        path = out_folder / df_name,
        index=False
    ) 



if __name__ == "__main__":
    t1 = time.time()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    run_data_pipeline(Path("/Users/henrykraessig/code/bdb2025/data/"), Path("./data"), minimal=True)
    
    logging.info(f"data pipeline completed in {time.time() - t1: .1f} seconds")