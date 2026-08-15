import numpy as np
import torch
from typing import Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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


# TODO: do these hooks only set a value for the last batch in each train set?
def register_hooks(model: torch.nn.Module):

    handles = []

    grad_norms = {}
    def make_block_grad_norm_hook(name): 

        def hook(module, grad_input, grad_output): 
            grads = [g for g in grad_output if g is not None]

            if not grads: 
                return

            norm = torch.cat([g.detach().flatten() for g in grads]).norm().item()

            grad_norms[name] = norm

        return hook


    act_norms = {}
    def make_block_act_norm_hook(name): 

        def hook(module, input, output): 

            act_norm = output.detach().norm().item()
            act_norms[name] = act_norm


        return hook


    def make_rbp_norm_hook(name): 
        def hook(module, input, output): 

            pass
            # logger.debug(output.detach().norm())

        return hook
            


    if hasattr(model, 'blocks'): 
        for i, block in enumerate(model.blocks): 
            handle = block.register_full_backward_hook(make_block_grad_norm_hook(f'block.{i}'))
            handles.append(handle)

            handle = block.register_forward_hook(make_block_act_norm_hook(f'block_{i}'))
            handles.append(handle)


        if hasattr(model, 'add_rpb') and model.add_rpb:
            for i, block in enumerate(model.blocks): 
                handle = block.attn_layer.RPB.register_forward_hook(make_rbp_norm_hook(f'block_{i}'))
                handles.append(handle)
                
    elif hasattr(model, 'transformer_encoder'): 

        for i, layer in enumerate(model.transformer_encoder.layers): 
            handle = layer.register_forward_hook(make_block_act_norm_hook(f'block_{i}'))
            handles.append(handle)

            handle = layer.register_full_backward_hook(make_block_grad_norm_hook(f'block_{i}'))
            handles.append(handle)


    return grad_norms, act_norms, handles


def total_grad_norm(model: torch.nn.Module):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().norm().item() ** 2
    return total_sq ** 0.5



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


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        start_value=0,
        warmup_iters=0,
        freeze_value=None,
        freeze_iters=0,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters

        warmup_schedule = np.linspace(start_value, base_value, warmup_iters)

        if freeze_value is None:
            freeze_value = final_value
        freeze_schedule = np.ones(freeze_iters) * freeze_value

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((warmup_schedule, schedule, freeze_schedule))
        self.iter = 0

    def get(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

    def step(self):
        value = self.get(self.iter)
        self.iter += 1
        return value

    def reset(self):
        self.iter = 0

    def __getitem__(self, it):
        return self.get(it)