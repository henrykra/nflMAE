"""Common functions for loading and preparing Big Data Bowl 2025 data for modeling."""
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

def load_csv_data(
    paths: list[Path]
) -> list[pd.DataFrame]:
    # TODO: look into reading from template, i.e. tracking_week_*.csv
    return [pd.read_csv(path) for path in paths]


def load_tracking_data(
    weeks: Optional[int | list[int]]=None,
    folder: Optional[Path]=None,
) -> pd.DataFrame:
   
    MIN_WEEK = 1
    MAX_WEEK = 9

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


def load_player_play_data(
        folder: Optional[Path]=None,
) -> pd.DataFrame:
    

    if folder and folder.exists():
        logger.info(f"reading play data from {folder}")
        path = Path(folder, 'player_play.csv')

    return pd.read_csv(path)


def clean_player_data(player_df: pd.DataFrame) -> pd.DataFrame:
    print("cleaning player data")
    
    # clean height
    feet = pd.to_numeric(player_df['height'].str.split('-', expand=True)[0])
    inches = pd.to_numeric(player_df['height'].str.split('-', expand=True)[1])
    player_df['height_inches'] = 12 * feet + inches

    # create zscore for height
    player_df['height_z'] = (player_df['height_inches'] - player_df['height_inches'].mean()) / player_df['height_inches'].std()
    player_df['weight_z'] = (player_df['weight'] - player_df['weight'].mean()) / player_df['weight'].std()

    return player_df


def clean_tracking_data(tracking_df: pd.DataFrame) -> pd.DataFrame:
    print("cleaning tracking data")

    # angles to radians
    tracking_df['o'] = ((-1 * tracking_df['o'] + 90) % 360) * np.pi / 180
    tracking_df['dir'] = ((-1 * tracking_df['dir'] + 90) % 360) * np.pi / 180

    # standardize locations to the ball snap location
    ball = (
        tracking_df
        .loc[
            (tracking_df['event'] == 'ball_snap') & 
            (tracking_df['club'] == 'football'),
             
            ['gameId', 'playId', 'frameId', 'x', 'y']
        ]
    )
    tracking_df = tracking_df.merge(
        ball, # ball info for starting time
        how='left', 
        on=('gameId', 'playId'),
        suffixes=('', '_ball')
    )

    tracking_df['x'] = tracking_df['x'] - tracking_df['x_ball']
    tracking_df['y'] = tracking_df['y'] - tracking_df['y_ball']

    # normalize play directions to the right
    tracking_df['x'] = ((tracking_df['playDirection'] == 'left') * -2 + 1) * tracking_df['x']
    tracking_df['o'] = ( ((tracking_df['playDirection'] == 'left') * np.pi) + tracking_df['o'] ) % (2 * np.pi) 
    tracking_df['dir'] = ( ((tracking_df['playDirection'] == 'left') * np.pi) + tracking_df['dir'] ) % (2 * np.pi) 

    # x and y components of orientation and velocity
    tracking_df['ox'] = np.cos(tracking_df['o'])
    tracking_df['oy'] = np.sin(tracking_df['o'])

    tracking_df['vx'] = np.cos(tracking_df['dir']) * tracking_df['s']
    tracking_df['vy'] = np.sin(tracking_df['dir']) * tracking_df['s']

    tracking_df['stop_point'] = pd.NA
    tracking_df.loc[tracking_df['event'].isin(['pass_outcome_incomplete', 'qb_sack', 'tackle']), 'stop_point'] = True 
    # fill stopping points forward to filter out moments after a defined route stop
    tracking_df['stop_point'] = tracking_df.groupby(['gameId', 'playId', 'nflId'])['stop_point'].ffill()
    
    tracking_df = tracking_df.loc[pd.isna(tracking_df['stop_point'])].copy()

    tracking_df['frameId'] = tracking_df['frameId'] - tracking_df['frameId_ball']

    # filter
    tracking_df = tracking_df.loc[
        (tracking_df['frameId'] >= 0) & # after the starting point
        (tracking_df['club'] != 'football')
        
        #['gameId', 'playId', 'frameId', 'nflId', 'x', 'y', 'vx', 'vy', 'a', 'ox', 'oy']
    ].copy()
    
    return tracking_df


def mirror_tracking_plays(tracking_df: pd.DataFrame) -> pd.DataFrame:
    print("mirroring tracking data")
    mirrored_df = tracking_df.copy()

    mirrored_df['y'] = mirrored_df['y'] * -1
    mirrored_df['vy'] = mirrored_df['vy'] * -1
    mirrored_df['oy'] = mirrored_df['oy'] * -1

    mirrored_df['mirrored'] = True
    tracking_df['mirrored'] = False


    return pd.concat([tracking_df, mirrored_df])


def prepare_static_data(tracking_df: pd.DataFrame, play_df: pd.DataFrame, player_df: pd.DataFrame) -> np.ndarray:
    print("preparing static data")
    # join 

    first_frames = (
        tracking_df
        .groupby(['gameId', 'playId', 'nflId', 'mirrored'], as_index=False)
        .first()
    )

    joined_df = (
        first_frames
        .merge(
            play_df,
            how='left',
            on=('gameId', 'playId')
        )
        .merge( # player info for positions
            player_df, 
            how='left',
            on='nflId'
        )
    )
    joined_df['offense'] = joined_df['club'] == joined_df['possessionTeam']

    # filter
    input_df = joined_df[        
        ['gameId', 'playId', 'nflId', 'mirrored', 'height_z', 'weight_z', 'position', 'offense'] 
    ]
    static_df = pd.get_dummies(input_df)



    input_play_index = static_df.set_index(['gameId', 'playId', 'mirrored']).sort_index()

    num_plays = len(input_play_index.index.unique())
    routes_per_play = static_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=True).iloc[0]

    assert (static_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=False) == routes_per_play).all(), \
    "Different values across player dimension"
    
    num_input_columns = len(input_play_index.columns) - 1 # subtract the nflId column
    input_player_info = np.zeros( (num_plays, routes_per_play, num_input_columns), dtype=np.float32) 

    for i, (name, group) in enumerate(input_play_index.groupby(['gameId', 'playId', 'mirrored'])):
        input_player_info[i, ...] = group.drop(columns=['nflId'])
        
    static_in = input_player_info.reshape(num_plays, routes_per_play, num_input_columns)

    return static_in



def get_num_static_cols(static_df: pd.DataFrame) -> int:

    return len([col for col in static_df.columns if 'Id' not in col and 'mirrored' not in col])


def write_static(static: pd.DataFrame) -> None: 
    print("writing dataframes")
    static.to_parquet("./data/static.parquet")
    return


def read_static(static_path: Path) -> pd.DataFrame:
    return pd.read_parquet(static_path)



def compile_static_dataframe(tracking_df: pd.DataFrame, play_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
        # join 

        first_frames = ( # join the player info on the players who are in each play
            tracking_df
            .groupby(['gameId', 'playId', 'nflId', 'mirrored'], as_index=False)
            .first()
        )

        joined_df = (
            first_frames
            .merge(
                play_df,
                how='left',
                on=('gameId', 'playId')
            )
            .merge( # player info for positions
                player_df, 
                how='left',
                on='nflId'
            )
        )
        joined_df['offense'] = joined_df['club'] == joined_df['possessionTeam']

        # filter
        input_df = joined_df[        
            ['gameId', 'playId', 'nflId', 'mirrored', 'height_z', 'weight_z', 'position', 'offense'] 
        ]
        static_df = pd.get_dummies(input_df)

        return static_df


def create_data_arrays(tracking_df: pd.DataFrame, play_df: pd.DataFrame, player_df: pd.DataFrame, interp: int, target_cols: list=[]):   
    print("preparing data arrays") 
    num_obs = interp
    interpolated_columns = ['x', 'y', 'ox', 'oy', 'vx', 'vy', 'a']


    static_df = compile_static_dataframe(tracking_df, play_df, player_df)

    # check for mismatch
    routes_per_play = static_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=True).iloc[0]
    assert (static_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=False) == routes_per_play).all(), \
    "Different values across player dimension"



    static_df = static_df.set_index(['gameId', 'playId', 'mirrored']).sort_index()
    ids = static_df.index.unique()
    num_plays = len(ids)
    tracking_df = tracking_df.set_index(['gameId', 'playId', 'mirrored']).sort_index()

    num_static_cols = len(static_df.columns)
    assert static_df.columns[0] == "nflId", static_df.columns[0]


    movement_arr = np.zeros( (num_plays, routes_per_play, num_obs, len(interpolated_columns)), dtype=np.float32)
    static_arr = np.zeros( (num_plays, routes_per_play, num_static_cols), dtype=np.float32)

    if target_cols:
        target_df = pd.get_dummies(play_df.set_index(['gameId', 'playId'])[target_cols])

        num_target_cols = len(target_df.columns)
        target_arr = np.zeros( (num_plays, num_target_cols ))
    else:
        target_arr = None


    for i, (gameid, playid, mirrored) in enumerate(ids):
        

        # intepolate each player's movement
        play: pd.DataFrame = tracking_df.loc[gameid, playid, mirrored]

        # populate static array
        static_arr[i] = static_df.loc[(gameid, playid, mirrored)].to_numpy()
        

        # populate movement array
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

            # ensure nflIds line up
            assert static_arr[i, ii, 0] == nflid


        # populate target array
        if target_cols:
            target_arr[i] = target_df.loc[(gameid, playid)].to_numpy()

        


    # delete nflId from static
    static_arr = np.delete(static_arr, 0, axis=-1)

    return static_arr, movement_arr, target_arr


