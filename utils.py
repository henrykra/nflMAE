import numpy as np
import torch
from typing import Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


def plot_frame_tensor(
    input_tensor: torch.Tensor,
    target_loc: Optional[torch.Tensor]=None,
    pred: Optional[torch.Tensor]=None
):
    
    x = input_tensor[:, 0]
    y = input_tensor[:, 1]
    ox = input_tensor[:, 2]
    oy = input_tensor[:, 3]

    offense = input_tensor[:, 7]
    defense = input_tensor[:, 8]

    club = defense * 2 + offense # combine to categorical: 0, 1, 2
    
    play_df = pd.DataFrame({
        'x': x,
        'y': y,
        'ox': ox,
        'oy': oy,
        'club': club
    })
    play_df['club'] = play_df['club'].astype('object')


    # input base plot
    fig = px.scatter(
        play_df,
        x="x",
        y="y",
        color="club",          
    )

    arrows = []

    for i, row in play_df.iterrows():
        if row.club == 0: 
            arrows.append(dict(
                text="", showarrow=False
            ))
        else:
            arrows.append(dict(
                ax=row.x,
                ay=row.y,
                x=row.x + 1.5 * row.ox,
                y=row.y + 1.5 * row.oy,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                text="",
                arrowhead=1
            ))

    fig.layout.annotations = arrows

    if target_loc is not None: 
        # add ball tackle location
        fig.add_trace(
            go.Scatter(
                x=[target_loc[0]],
                y=[target_loc[1]],
                marker=dict(size=12, color='green', symbol='x'),
                text=["Ball Landing"]
            )
        )


    if pred is not None: 
        fig.add_trace(
            go.Scatter(
                x=[pred[0]],
                y=[pred[1]],
                marker=dict(size=12, color='yellow', symbol='x'),
                text=["Predicted Ball Landing"]
            )
        )
    
    # Keep aspect ratio fixed and field bounds constant
    fig.update_layout(
        width=600, height=500,
        xaxis=dict(range=[-10, 30]),
        yaxis=dict(range=[-53.3 // 2, 53.3 // 2], scaleanchor="x", scaleratio=1),
        showlegend=False
    )

    return fig



def register_hooks(model: torch.nn.Module):

    norms = {}
    stds = {}
    def get_act(idx): 
        
        def hook(module, args, output): 
            act = output.detach()
            norms[idx] = act.norm().item()
            stds[idx] = act.std().item()

        return hook
        

    for i, module in enumerate(model.blocks): 
        module.register_forward_hook(get_act(i))


    return norms, stds

    


def plot_activation_stats(norms: dict, stds: dict): 

    df = pd.DataFrame({'norm': norms.values(), 'std': stds.values()})

    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['norm'],
            name="Activation Norm",
            mode='lines+markers'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['std'],
            name="Activation std",
            mode='lines+markers'
        ),
        secondary_y=True
    )

    fig.update_layout(title_text="Block Activations")
    fig.update_yaxes(title_text="Norm", secondary_y=False)
    fig.update_yaxes(title_text="Std", secondary_y=True)
    fig.update_xaxes(title_text="Layer")

    return fig