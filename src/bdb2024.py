"""Common functions for loading and preparing Big Data Bowl data for modeling."""
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import time
import sys
import logging
logger = logging.getLogger(__name__)

MIN_WEEK = 1
MAX_WEEK = 9

# Base data loading functions
def load_csv_data(
    paths: list[Path]
) -> list[pd.DataFrame]:
    # TODO: look into reading from template, i.e. tracking_week_*.csv
    return [pd.read_csv(path) for path in paths]


def load_tracking_data(
    weeks: Optional[int | list[int]]=None,
    folder: Optional[Path]=None,
) -> pd.DataFrame:

    if not weeks:
        # defualt to all data
        weeks = range(MIN_WEEK, MAX_WEEK + 1)
    
    if isinstance(weeks, int):
        weeks = [weeks]
    weeks = set(weeks)
    
    for week in weeks:
        assert isinstance(week, int) and week >= MIN_WEEK and week <= MAX_WEEK, \
        f"provide integer weeks between {MIN_WEEK} and {MAX_WEEK}"

    print(f"reading data from weeks: {weeks}")

    # load data from a folderectory
    if folder and folder.exists():
        logger.info(f"reading tracking from {folder}")
        paths = [Path(folder, f'tracking_week_{n}.csv') for n in weeks]

        dfs = load_csv_data(paths)
        return pd.concat(dfs)
    
    raise FileNotFoundError


def load_player_data(
    folder: Optional[Path]=None,
) -> pd.DataFrame:

    if folder and folder.exists():
        logger.info(f"reading player data from {folder}")
        path = Path(folder, 'players.csv')

        return pd.read_csv(path)
    

def load_play_data(
        folder: Optional[Path]=None,
) -> pd.DataFrame:
    

    if folder and folder.exists():
        logger.info(f"reading play data from {folder}")
        path = Path(folder, 'plays.csv')

    return pd.read_csv(path)


### SIMPLE CLEANING

def clean_tracking_angles(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    logger.info("cleaning tracking data angles")
    # angles to radians
    tracking_df['o'] = ((-1 * tracking_df['o'] + 90) % 360) * np.pi / 180
    tracking_df['dir'] = ((-1 * tracking_df['dir'] + 90) % 360) * np.pi / 180

    return tracking_df

def standardize_tracking_plays(tracking_df: pd.DataFrame, play_df: pd.DataFrame) -> pd.DataFrame: 
    logger.info("standardizing tracking data")
    # standardize locations to the ball snap location
    ball = (
        tracking_df
        .loc[
            (tracking_df['event'] == 'ball_snap') & 
            (tracking_df['club'] == 'football'),
            
            ['gameId', 'playId', 'frameId', 'x', 'y']
        ]
    )
    # if we don't know where the ball was snapped, make an assumption at the middle of the field
    
    tracking_df = tracking_df.merge(
        ball, # ball info for starting time
        how='left', 
        on=('gameId', 'playId'),
        suffixes=('', '_ball_snap')
    ).merge(
        play_df,
        how='left',
        on=['gameId', 'playId']
    )
    tracking_df['x_ball_snap'] = tracking_df['x_ball_snap'].fillna(tracking_df['absoluteYardlineNumber'])
    tracking_df['y_ball_snap'] = tracking_df['y_ball_snap'].fillna(53.3/2)

    tracking_df['x'] = tracking_df['x'] - tracking_df['x_ball_snap']
    tracking_df['y'] = tracking_df['y'] - tracking_df['y_ball_snap']

    # normalize play directions to the right
    tracking_df['x'] = ((tracking_df['playDirection'] == 'left') * -2 + 1) * tracking_df['x']
    tracking_df['y'] = ((tracking_df['playDirection'] == 'left') * -2 + 1) * tracking_df['y']
    tracking_df['o'] = ( ((tracking_df['playDirection'] == 'left') * np.pi) + tracking_df['o'] ) % (2 * np.pi) 
    tracking_df['dir'] = ( ((tracking_df['playDirection'] == 'left') * np.pi) + tracking_df['dir'] ) % (2 * np.pi) 

    # x and y components of orientation and velocity
    tracking_df['ox'] = np.cos(tracking_df['o'])
    tracking_df['oy'] = np.sin(tracking_df['o'])

    tracking_df['vx'] = np.cos(tracking_df['dir']) * tracking_df['s']
    tracking_df['vy'] = np.sin(tracking_df['dir']) * tracking_df['s']

    tracking_df['nflId'] = tracking_df['nflId'].fillna(-1) # football

    play_cols = [col for col in play_df.columns if not col.endswith('Id') or col == "ballCarrierId"]

    return tracking_df.drop(columns=play_cols)


def clean_player_data(player_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("cleaning player data")
    
    # clean height
    feet = pd.to_numeric(player_df['height'].str.split('-', expand=True)[0])
    inches = pd.to_numeric(player_df['height'].str.split('-', expand=True)[1])
    player_df['height_inches'] = 12 * feet + inches

    # create zscore for height
    player_df['height_z'] = (player_df['height_inches'] - player_df['height_inches'].mean()) / player_df['height_inches'].std()
    player_df['weight_z'] = (player_df['weight'] - player_df['weight'].mean()) / player_df['weight'].std()

    return player_df


### FILTERING

def filter_play_start_frames(tracking_df: pd.DataFrame) -> pd.DataFrame:
    # standardize frameId to 0 at snap & remove presnap frames
    tracking_df['frameId'] = tracking_df['frameId'] - tracking_df['frameId_ball_snap'].fillna(1)
    return tracking_df.loc[tracking_df['frameId'] >=  0]


def filter_play_end_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    # find play end events
    end = (tracking_df.groupby(['playId', 'gameId'])['frameId'].max() - 4).reset_index().set_index(['playId', 'gameId', 'frameId'])
    events = tracking_df.groupby(['playId', 'gameId', 'frameId'])['event'].first()
    end_events = events.loc[end.index].dropna().unique()
    print(end_events)

    #print plays with tackles

    # create a stop point variable to track frames after the play
    tracking_df['stop_point'] = pd.NA
    tracking_df.loc[tracking_df['event'].isin(end_events), 'stop_point'] = True 
    # fill stopping points forward to filter out moments after a defined route stop
    tracking_df['stop_point'] = tracking_df.groupby(['gameId', 'playId', 'nflId'])['stop_point'].ffill()

    # filter only after the event
    tracking_df.loc[tracking_df['event'].isin(end_events), 'stop_point'] = None


    return tracking_df.loc[tracking_df['stop_point'].isna()].drop(columns='stop_point') # remove redundant column


def filter_route_end_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    # potentially complex logic to find when a route runner's designed route has ended
    ...

def filter_early_frames(tracking_df: pd.DataFrame) -> pd.DataFrame:
    # filter tracking frames within a margin from the earliest frame in the play
    n = 15
    logger.info(f"filtering relative frames < {n}")
    tracking_df['relative_frame'] = tracking_df['frameId'] - tracking_df.groupby(['gameId', 'playId'])['frameId'].transform('min')

    tracking_df = tracking_df.loc[tracking_df['relative_frame'] >= n]
    return tracking_df


def filter_football_tracking_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    return tracking_df.loc[tracking_df['club'] != 'football']


def filter_outlier_plays(tracking_df: pd.DataFrame, player_play_df: pd.DataFrame): 

    # filter out the one play from player plays that doesn't have 22 players
    n_players = player_play_df.groupby(['gameId', 'playId']).transform('size')
    bad_ids = player_play_df.loc[n_players != 22].set_index(['gameId', 'playId']).index.unique()
    player_play_df = player_play_df.loc[n_players == 22]

    
    if bad_ids.isin(tracking_df.set_index(['gameId', 'playId']).index): 
        tracking_df = tracking_df.set_index(['gameId', 'playId']).drop(index=bad_ids).reset_index()
        logger.info(f"filtering {bad_ids}")

    return tracking_df, player_play_df


def filter_outlier_frames(tracking_df: pd.DataFrame) -> pd.DataFrame: 
    good_frame = tracking_df.groupby(['gameId', 'playId', 'frameId'])['nflId'].nunique() == 23

    good_frame_ids = good_frame.loc[good_frame]

    tracking_df = tracking_df.set_index(['gameId', 'playId', 'frameId']).loc[good_frame_ids.index]
    return tracking_df.reset_index().copy()


### TRANSFORMS

# perform unique data transforms
def find_play_invariant_columns(play_df: pd.DataFrame, tracking_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: 
    # grab the column names for columns in the tracking data that won't change over the course of a given play
    cols_static = (tracking_df.groupby(['gameId', 'playId']).nunique() == 1).all(axis=0) # series w/ index colname and boolean value
    invariant_columns = cols_static.index[cols_static].tolist()

    tracking_invariant = tracking_df.groupby(['gameId', 'playId'])[invariant_columns].first()

    # tack them on to the plays dataframe
    play_df = play_df.merge(
        tracking_invariant,
        how='left', 
        on=('gameId', 'playId'),
    )    
    logger.info(f"moving {invariant_columns} to play df")
    # remove them from the tracking dataframe
    return play_df, tracking_df.drop(columns=invariant_columns)


def find_player_play_invariant_columns(player_play_df: pd.DataFrame, tracking_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: 
    # grab the column names for columns in the tracking data that won't change over the course of a given play for a given player
    cols_static = (tracking_df.groupby(['gameId', 'playId', 'nflId']).nunique() == 1).all(axis=0) # series w/ index colname and boolean value
    invariant_columns = cols_static.index[cols_static].tolist()

    tracking_invariant = tracking_df.groupby(['gameId', 'playId', 'nflId'])[invariant_columns].first()

    if player_play_df is None: 
        player_play_df = tracking_invariant.reset_index()

    else:
        # tack them on to the plays dataframe
        player_play_df = player_play_df.merge(
            tracking_invariant,
            how='left', 
            on=('gameId', 'playId', 'nflId')
        )    

    # remove them from the tracking dataframe
    logger.info(f"moving {invariant_columns} to player play df")
    return player_play_df, tracking_df.drop(columns=invariant_columns)


def clean_player_play_data(player_play_df: pd.DataFrame, play_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: 

    # player_play_df['is_ball'] = (player_play_df['club'] == 'football') # no football frames in the player_play dataset

    merged = player_play_df[['gameId', 'playId', 'club']].merge(play_df[['gameId', 'playId', 'possessionTeam', 'defensiveTeam']].copy(), how='left', on=('gameId', 'playId'))

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

### MIRROR
def mirror_tracking_data(tracking_df: pd.DataFrame, player_play_df: pd.DataFrame, play_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("mirroring tracking data")
    mirrored_tracking = tracking_df.copy()
    mirrored_starts = player_play_df.copy()
    mirrored_snaps = play_df.copy()

    mirrored_tracking['y'] = mirrored_tracking['y'] * -1
    mirrored_tracking['vy'] = mirrored_tracking['vy'] * -1
    mirrored_tracking['oy'] = mirrored_tracking['oy'] * -1
    mirrored_tracking['mirrored'] = True
    tracking_df['mirrored'] = False

    if 'y_start' in player_play_df.columns:
        mirrored_starts['y_start'] = mirrored_starts['y_start'] * -1
    mirrored_starts['mirrored'] = True
    player_play_df['mirrored'] = False

    mirrored_snaps['y_ball_snap'] = mirrored_snaps['y_ball_snap'] * -1
    mirrored_snaps['mirrored'] = True
    play_df['mirrored'] = False


    return pd.concat([tracking_df, mirrored_tracking]), pd.concat([player_play_df, mirrored_starts]), pd.concat([play_df, mirrored_snaps])


### MERGE


def merge_dataframes(
    tracking_df: pd.DataFrame,
    player_play_df: pd.DataFrame,
    play_df: pd.DataFrame,
    player_df: pd.DataFrame,
): 
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
            tracking = load_csv_data([Path("../2024_minimal_tracking.csv")])[0]
        else: 
            tracking = load_tracking_data(weeks=week, folder=data_folder)

        players = load_player_data(data_folder)
        plays = load_play_data(data_folder)

        # clean
        tracking = clean_tracking_angles(tracking)
        tracking = standardize_tracking_plays(tracking, plays)
        # tracking = football_relative_positions(tracking)
        players = clean_player_data(players)

        # filter
        # tracking = filter_early_frames(tracking)
        # tracking = filter_play_start_frames(tracking)
        # tracking = filter_play_end_frames(tracking)
        tracking = filter_outlier_frames(tracking)
        # tracking = filter_football_tracking_frames(tracking)

        routes_per_play = tracking.groupby(['gameId', 'playId'])['nflId'].nunique(dropna=False).iloc[0]
        logger.debug(f"{routes_per_play} routes per play")

        # transform
        plays, tracking = find_play_invariant_columns(plays, tracking)
        player_plays, tracking = find_player_play_invariant_columns(None, tracking)
        player_plays, plays = clean_player_play_data(player_plays, plays)
        # tracking, player_plays = player_relative_positions(tracking, player_plays)

        # mirror
        tracking, player_plays, plays = mirror_tracking_data(tracking, player_plays, plays)

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



if __name__ == '__main__': 
    t1 = time.time()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    run_data_pipeline(Path("/Users/henrykraessig/code/bdb2024/data/"), Path("./data"), minimal=True)
    
    logging.info(f"data pipeline completed in {time.time() - t1: .1f} seconds")
