import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from nflMAE.data import bdb_common

logger = logging.getLogger(__name__)


DEFAULT_WIDTH = 560
DEFAULT_HEIGHT = 420

OFFENSE_COLOR = "#1d4ed8"
DEFENSE_COLOR = "#64748b"

ROUTE_PALETTE = qualitative.Set3


def decode_onehots(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.Series:
    """Turn one-hot columns into their label names"""
    if not cols:
        return pd.Series("", index=df.index, dtype=object)

    values = df[cols].to_numpy(dtype=np.float64)
    labels = np.array([col[len(prefix) + 1:] for col in cols], dtype=object)

    best = values.argmax(axis=1)
    has_label = values.max(axis=1) > 0
    return pd.Series(np.where(has_label, labels[best], ""), index=df.index, dtype=object)


def select_play(
    df: pd.DataFrame,
    game_id: int,
    play_id: int,
    mirrored: bool = False,
) -> pd.DataFrame:
    rows = df.loc[
        (df['gameId'] == game_id) & (df['playId'] == play_id) & (df['mirrored'] == mirrored)
    ].sort_values(['frameId', 'nflId'])

    if rows.empty:
        raise ValueError(f"no rows for game {game_id} play {play_id} (mirrored={mirrored})")

    return rows.copy()


def snap_positions(play_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per entity (22 players + the football) at the snap frame."""
    frame = int(play_rows['frameId'].min())
    return play_rows.loc[play_rows['frameId'] == frame].copy()


def route_trajectories(play_rows: pd.DataFrame) -> dict[int, pd.DataFrame]:
    runners = play_rows.loc[play_rows['wasRunningRoute'].fillna(0) == 1, 'nflId'].unique()

    return {
        int(nfl_id): play_rows.loc[play_rows['nflId'] == nfl_id].sort_values("frameId").copy()
        for nfl_id in runners
    }


def _route_color_map(df: pd.DataFrame) -> dict[str, str]:
    route_cols = [col for col in df.columns if col.startswith("routeRan")]
    return {
        label: ROUTE_PALETTE[i % len(ROUTE_PALETTE)]
        for i, label in enumerate(col[len("routeRan") + 1:] for col in route_cols)
    }


def _describe_entity(row: pd.Series, route: str) -> str:
    """Multi-line hover text for one entity at the snap."""
    nfl_id = int(row['nflId'])
    if nfl_id == -1:
        who = "football"  # the merged data uses nflId -1 for the ball
    else:
        who = f"nflId {nfl_id}"

    lines = [who]
    pos = row.get("_position", "")
    if pos:
        lines.append(f"position {pos}")
    if nfl_id != -1:
        lines.append(f"nflId {nfl_id}")
    lines.append(f"route {route}" if route else "")
    return "<br>".join(lines)


def route_figure_plotly(
    play_df: pd.DataFrame,
    width: int=DEFAULT_WIDTH,
    height: int=DEFAULT_HEIGHT,
) -> go.Figure:
    """Draw a figure of a play's routes run using plotly"""
    plays = play_df[['gameId', 'playId']].drop_duplicates()
    game_id, play_id = plays.iloc[0]

    play_df = play_df.sort_values(['frameId', 'nflId']).copy()

    # return 1-hot columns to description
    position_cols = [col for col in play_df.columns if col.startswith("position")]
    play_df['_position'] = decode_onehots(play_df, position_cols, "position")
    route_cols = [col for col in play_df.columns if col.startswith("routeRan")]
    play_df['_route'] = decode_onehots(play_df, route_cols, "routeRan")

    colors = _route_color_map(play_df)

    fig = go.Figure()

    # get location from ball snap for reference
    snaps = snap_positions(play_df)
    for nfl_id, traj in route_trajectories(play_df).items():
        route = traj['_route'].iloc[0]

        snap_row = snaps.loc[snaps['nflId'] == nfl_id].iloc[0]

        hover = "<br>".join([
            _describe_entity(snap_row, route),
            "frame %{customdata}",  \
        ])
        
        color = colors.get(route, DEFENSE_COLOR)

        fig.add_trace(go.Scatter(
            x=-traj['y'].to_numpy(),
            y=traj['x'].to_numpy(),
            mode="lines",
            name=route or "route",
            line=dict(color=color, width=3),
            customdata=traj['frameId'].to_numpy(),
            hovertemplate=hover + "<extra></extra>",
        ))

    # plot locations from snap
    for team, color in (("offense", OFFENSE_COLOR), ("defense", DEFENSE_COLOR)):
        team_rows = snaps.loc[snaps[team].fillna(0) == 1]

        fig.add_trace(go.Scatter(
            x=-team_rows['y'].to_numpy(),
            y=team_rows['x'].to_numpy(),
            mode="markers",
            name=f"snap · {team}",
            marker=dict(
                size=11,
                color=color,
                symbol="circle" if team == "offense" else "circle-open",
                line=dict(width=2),
            ),
            text=[
                _describe_entity(row, row['_route'])
                for _, row in team_rows.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>",
        ))


    # plot LOS 
    fig.add_hline(
        y=0, line_width=1, line_dash="dot", line_color="#9ca3af",
    )

    # figure titles
    fig.update_layout(
        title=f"game {game_id} · play {play_id}",
        width=width,
        height=height,
        margin=dict(l=50, r=20, t=60, b=50),
        showlegend=False,  # no 
        hovermode="closest",
        xaxis_title="",
        yaxis_title="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    # equal yard scale on both axes
    fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, constrain="domain")

    return fig


def play_route_figure_plotly(
    df: pd.DataFrame,
    game_id: int,
    play_id: int,
    mirrored: bool = False,
    **figure_kwargs,
) -> go.Figure:

    return route_figure_plotly(select_play(df, game_id, play_id, mirrored), **figure_kwargs)