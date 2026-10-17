"""Common functions for loading and preparing Big Data Bowl data for modeling."""
from pathlib import Path
import pandas as pd
import time
import sys
import logging

logger = logging.getLogger(__name__)

from nflMAE.data import bdb_common
from nflMAE import root

MIN_WEEK = 1
MAX_WEEK = 9


def filter_play_end_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    # find play end events
    end = (tracking_df.groupby(['playId', 'gameId'])['frameId'].max() - 4).reset_index().set_index(['playId', 'gameId', 'frameId'])
    events = tracking_df.groupby(['playId', 'gameId', 'frameId'])['event'].first()
    end_events = events.loc[end.index].dropna().unique()
    logger.info(f"play end events identified: {end_events}")

    # create a stop point variable to track frames after the play
    tracking_df['stop_point'] = pd.NA
    tracking_df.loc[tracking_df['event'].isin(end_events), 'stop_point'] = True 
    # fill stopping points forward to filter out moments after a defined route stop
    tracking_df['stop_point'] = tracking_df.groupby(['gameId', 'playId', 'nflId'])['stop_point'].ffill()

    # filter only after the event
    tracking_df.loc[tracking_df['event'].isin(end_events), 'stop_point'] = None

    return tracking_df.loc[tracking_df['stop_point'].isna()].drop(columns='stop_point') # remove redundant column


def filter_route_end_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    # complex logic to find when a route runner's designed route has ended
    ...


def filter_early_frames(tracking_df: pd.DataFrame) -> pd.DataFrame:
    # filter tracking frames within a margin from the earliest frame in the play
    n = 15
    logger.info(f"filtering relative frames < {n}")
    tracking_df['relative_frame'] = tracking_df['frameId'] - tracking_df.groupby(['gameId', 'playId'])['frameId'].transform('min')

    tracking_df = tracking_df.loc[tracking_df['relative_frame'] >= n]
    return tracking_df


def filter_football_tracking_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    logger.info("filtering football from tracking")
    return tracking_df.loc[tracking_df['club'] != "football"]


def clean_player_play_data(player_play_df: pd.DataFrame, play_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: 
    # create offense/defense columns from possesionTeam and club
    merged = (
        player_play_df
        [['gameId', 'playId', 'club']]
        .merge(
            play_df[['gameId', 'playId', 'possessionTeam', 'defensiveTeam']].copy(), 
            how='left', 
            on=('gameId', 'playId')
        )
    )

    player_play_df['offense'] = (merged['club'] == merged['possessionTeam']).astype('float32')
    player_play_df['defense'] = (merged['club'] == merged['defensiveTeam']).astype('float32')

    logger.debug(f"{player_play_df['offense'].isna().sum()} NA \'offense\' values")

    # create is_ballcarrier (bool) in player_play
    play_df['isBallCarrier'] = 1
    player_play_df = player_play_df.merge(
        play_df[['gameId', 'playId', 'ballCarrierId', 'isBallCarrier']],
        how='left',
        left_on=('gameId', 'playId', 'nflId'),
        right_on=('gameId', 'playId', 'ballCarrierId')
    )
    player_play_df['isBallCarrier'] = player_play_df['isBallCarrier'].fillna(0)

    return player_play_df, play_df



### MERGE

def merge_dataframes(
    tracking_df: pd.DataFrame,
    player_play_df: pd.DataFrame,
    play_df: pd.DataFrame,
    player_df: pd.DataFrame,
): 
    # filter columns first
    tracking_cols = ['gameId', 'playId', 'nflId', 'frameId', 'mirrored', 'x', 'y', 'ox', 'oy', 'vx', 'vy', 'a', 'event']
    tracking_df = tracking_df[tracking_cols].copy()

    player_play_cols = ['gameId', 'playId', 'nflId', 'mirrored', 'offense', 'defense', 'isBallCarrier']
    player_play_df = player_play_df[player_play_cols].copy()

    player_cols = ['nflId', 'height_z', 'weight_z', 'position']
    player_df = player_df[player_cols].copy()

    merged = (
        tracking_df
        .merge(
            right=player_play_df, 
            how='left',
            on=['gameId', 'playId', 'nflId', 'mirrored']
        )
        .merge(
            right=player_df,
            how='left',
            on=['nflId']
        )
    )
    # create one-hot columns for categorical variables
    merged = pd.get_dummies(merged, columns=['position'])

    logger.debug(f"merged_df columns: {merged.columns}")

    logger.debug(f"{len(merged.loc[merged['offense'].isna(), 'nflId'].unique())} NA \'offense\' values")

    return merged



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
            tracking = bdb_common.load_csv_data([Path(root.parent, "2024_minimal_tracking.csv")])[0]
        else: 
            tracking = bdb_common.load_tracking_data(weeks=week, folder=data_folder)

        players = bdb_common.load_player_data(data_folder)
        plays = bdb_common.load_play_data(data_folder)

        # clean
        tracking = bdb_common.clean_tracking_angles(tracking)
        tracking = bdb_common.standardize_tracking_plays(tracking, plays)
        # tracking = football_relative_positions(tracking)
        players = bdb_common.clean_player_data(players)

        # filter
        # tracking = filter_early_frames(tracking)
        # tracking = filter_play_start_frames(tracking)
        # tracking = filter_play_end_frames(tracking)
        tracking = bdb_common.filter_outlier_frames(tracking)
        # tracking = filter_football_tracking_frames(tracking)

        routes_per_play = tracking.groupby(['gameId', 'playId'])['nflId'].nunique(dropna=False).iloc[0]
        logger.debug(f"{routes_per_play} routes per play")

        # transform
        plays, tracking = bdb_common.find_play_invariant_columns(plays, tracking)
        player_plays, tracking = bdb_common.find_player_play_invariant_columns(None, tracking)
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
        df_name = ("bdb2024" + ("" if not minimal else "_minimal") + ".parquet")

    # save
    merged.to_parquet(
        path = out_folder / df_name,
        index=False
    )
    return


if __name__ == '__main__': 
    t1 = time.time()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    run_data_pipeline(Path("/Users/henrykraessig/code/bdb2024/data/"), Path("./data"), minimal=True)
    
    logging.info(f"data pipeline completed in {time.time() - t1: .1f} seconds")
