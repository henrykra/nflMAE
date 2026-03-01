"""Post training task setup for predicting the targetted reciever given a set of frames"""
import bdb2025
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path
import datasets
import json
import modeling
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from train import create_dataloaders, check_early_stopping

# TODO: save model as state_dict

def parse_config():
    with open("./target_task/target_config.json", "r") as configfile:
        config = json.load(configfile)
    return config


def filter_tracking_data(tracking_df: pd.DataFrame, play_df: pd.DataFrame) -> pd.DataFrame:
    
    # find plays with a good value for the target column

    good_plays_df = play_df.loc[play_df['pff_manZone'].notna()].copy()
    good_plays = good_plays_df.set_index(['gameId', 'playId']).index


    if False:
        # note frames with forward pass
        tracking_df['pass_forward_frame'] = (tracking_df['event'] == 'pass_forward').map({False: pd.NA, True: True})

        # forward fill to mark frames after a pass
        tracking_df['after_pass'] = (
            tracking_df
            .groupby(['gameId', 'playId', 'nflId'])
            ['pass_forward_frame']
            .ffill()
        )

        # filter plays with a forward pass
        pass_forward_plays = (
            tracking_df
            .groupby(['gameId', 'playId'])
            ['pass_forward_frame']
            .agg(passing_play = "sum")
            .query("passing_play > 0")
            .index
        )

        # remove the marked frames too
        filtered_tracking = (
            tracking_df
            .loc[tracking_df['after_pass'].isna()]
            .set_index(['gameId', 'playId'])
            .loc[pass_forward_plays]
            .reset_index()
        )

        # remove temp columns
        return filtered_tracking.drop(columns=['pass_forward_frame', 'after_pass'])
    

    filtered_tracking = tracking_df.set_index(['gameId', 'playId']).loc[good_plays].reset_index()
    return filtered_tracking
        
    
def subsample_tracking_data(tracking_df: pd.DataFrame): 

    play_frames = tracking_df.groupby(['gameId', 'playId'])['frameId'].max().rename('max_frame')

    num_plays = play_frames.index.nunique()
    # for each play, select a couple random frames, then duplicate the play until those frames

    # rather than selecting frames, select random 0-1 then convert to a frame for each play
    random_times = np.random.rand(num_plays)
    #random_times = 1
    random_frames = play_frames * random_times
    random_frames.loc[random_frames < 5] = 5

    # filter tracking so that each play cuts off at the random frame
    tmp = tracking_df.merge(random_frames, on=['gameId', 'playId'], how='left')
    tmp['after'] = tmp['frameId'] > tmp['max_frame']
    tracking_df = tmp.loc[~tmp['after']].drop(columns=['max_frame', 'after']).copy()

    return tracking_df


def prepare_target(tracking_df: pd.DataFrame, play_df: pd.DataFrame):
    print("preparing target data")
    # join 

    first_frames = (
        tracking_df
        .groupby(['gameId', 'playId', 'nflId', 'mirrored'], as_index=False)
        .first()
    )

    joined_df = (
        first_frames
        .merge(
            play_df[['gameId', 'playId', 'pff_manZone']],
            how='left',
            on=('gameId', 'playId')
        )
    )

    # filter
    input_df = joined_df[        
        ['gameId', 'playId', 'mirrored', 'pff_manZone'] 
    ]


    static_df = pd.get_dummies(input_df)

    play_index = static_df.set_index(['gameId', 'playId', 'mirrored']).sort_index()

    num_plays = len(play_index.index.unique())
    
    num_columns = len(play_index.columns)
    target = np.zeros( (num_plays, num_columns), dtype=np.float32) 

    for i, (name, group) in enumerate(play_index.groupby(['gameId', 'playId', 'mirrored'])):
        target[i, ...] = group
    
    return target


class TargetsDataset(Dataset):
    
    def __init__(self, dynamic_arr: np.ndarray, static_arr: np.ndarray, target_arr: np.ndarray, idx: np.ndarray):

        # make sure there's the right amount of input observatinos and target observations
        assert static_arr.shape[0] == dynamic_arr.shape[0], "# Observations mismatch between input and target"

        n_plays = len(idx)
        routes_per_play =  static_arr.shape[1]

        self.static_t: torch.Tensor = torch.Tensor(static_arr[idx].copy())
        self.dynamic_t: torch.Tensor = torch.Tensor(dynamic_arr[idx].reshape(n_plays, routes_per_play, -1).copy())
        self.target_t: torch.Tensor = torch.Tensor(target_arr[idx].copy())
        return
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        tar = self.target_t[idx]
        inp = torch.cat([self.static_t[idx], self.dynamic_t[idx]], dim=-1)

        return inp, tar
        

    def __len__(self) -> int:

        return self.static_t.shape[0]
    

class ManZoneModel(modeling.NflBERT):
    def __init__(self,
        feature_dim: int,
        output_dim: int,
        hidden_size: int=512,
        num_layers: int=8,
        num_heads: int=8,
        ffn_size: int=2048,
        ffn_act: str="gelu",
        dropout: float=.3,
    ):
        
        super().__init__(feature_dim, output_dim, hidden_size,num_layers,num_heads, ffn_size, ffn_act, dropout)

        self.player_pooling_layer = torch.nn.AdaptiveAvgPool1d(1)

        self.target_decoder_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(hidden_size // 4),
            torch.nn.Linear(hidden_size // 4, 1),
        )   
        

    def forward(self, x: torch.Tensor):
        # x: [B: batch_size, P: # of players, F: feature_len]
        B, P, F = x.size()

        # Normalize features
        x = x.permute(0, 2, 1)

        x = self.norm_layer(x).permute(0, 2, 1)  # [B,P,F] -> [B,P,F]

        # Embed features
        x = self.embed(x)  # [B,P,F] -> [B,P,M: model_dim]

        # Apply transformer encoder
        x = self.transformer(x)  # [B,P,M] -> [B,P,M]

        # Pool over player dimension
        x = torch.squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)  # [B,M,P] -> [B,M]

        # Decode to predict tackle location
        x = self.target_decoder_head(x)  # [B, M] -> [B,O]

        return x



def load_and_prepare_model(hparams: dict, model_path: Path) -> modeling.NflBERT:
    model = ManZoneModel()
    model.load_state_dict(torch.load(model_path), strict=False)

    
    # freeze
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze head
    for param in model.target_decoder_head.parameters():
        param.requires_grad = True

    for param in model.player_pooling_layer.parameters():
        param.requires_grad = True

    return model


def train(hparams: dict, model: torch.nn.Module, train_set: torch.utils.data.Dataset, val_set: torch.utils.data.Dataset, test_set: torch.utils.data.Dataset, experiment_folder: Path):
    writer = SummaryWriter(experiment_folder)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=hparams["lr"])

    train_loader, val_loader, test_loader = create_dataloaders(train_set, val_set, test_set, hparams)

    for epoch in range(hparams["epochs"]):

        c = 0
        train_loss = 0
        val_loss = 0
        best_val_loss = float("inf")
        no_improvement = 0
        model.train()
        for (inp, tar) in train_loader:
            inp, tar = inp.to(hparams["device"]), tar.to(hparams["device"])
            optim.zero_grad()

            x = model(inp)
            # print(f'Pred: {np.argmax(x.detach().cpu()[0], axis=0)}')
            # print(f'Target: {np.argmax(tar.detach().cpu()[0], axis=0)}')

            loss = loss_fn(x, tar)
            train_loss += loss
            c += 1
            loss.backward()
            
            optim.step()
        # log average loss
        writer.add_scalar('Loss/train', train_loss / c, epoch)

        c = 0
        model.eval()
        with torch.no_grad():
            for (inp, tar) in val_loader:
                inp, tar = inp.to(hparams["device"]), tar.to(hparams["device"])
                x = model(inp)
                #print(f'Pred: {np.argmax(x.cpu()[0], axis=0)}')
                #print(f'Target: {np.argmax(tar.cpu()[0], axis=0)}')

                loss = loss_fn(x, tar)
                val_loss += loss
                c += 1

        # log average loss
            writer.add_scalar('Loss/val', val_loss / c, epoch)

            cont, best_val_loss, no_improvement = check_early_stopping(val_loss, best_val_loss, no_improvement, hparams["patience"])
            if not cont: # early stopped
                print(f"Early stopping at epoch {epoch}\nBest Loss: {best_val_loss}")
                break

            if epoch % 10 == 0:
                # log a sample play
                print(f"Epoch: {epoch}\tVal Loss: {val_loss/c:.3f}")
                writer.flush()
            
    writer.flush()
    writer.close()

    # get test loss
    test_loss = 0
    c = 0
    with torch.no_grad():
        for (inp, tar) in test_loader:
            inp, tar = inp.to(hparams["device"]), tar.to(hparams["device"])
            x = model(inp)

            loss = loss_fn(x, tar)
            test_loss += loss
            c += 1

    print(f"Test Loss: {test_loss / c:.3f}")
    return test_loss / c



def main(config: dict):

    if config["minimal"]:
        tracking = bdb2025.load_csv_data([Path("minimal_tracking.csv")])[0]
    else:
        tracking = bdb2025.load_tracking_data(folder=Path(config["folder"]))
    players = bdb2025.load_player_data(folder=Path(config["folder"]))
    plays = bdb2025.load_play_data(folder=Path(config["folder"]))
    player_play = bdb2025.load_player_play_data(folder=Path(config["folder"]))
    # clean
    players = bdb2025.clean_player_data(players)
    tracking = filter_tracking_data(tracking, plays) #### 
    tracking = bdb2025.clean_tracking_data(tracking)
    # tracking = subsample_tracking_data(tracking) ####
    tracking = bdb2025.mirror_tracking_plays(tracking)


    return
    static = bdb2025.prepare_static_data(tracking, plays, players)
    target = prepare_target(tracking, plays)
    dynamic = datasets.interpolate_movement(tracking, config["interp"])
    # prepare

    n = static.shape[0]
    train_idx, val_idx, test_idx = datasets.train_val_test_split(n, config["train"], config["val"], config["test"])
    train_dataset = TargetsDataset(dynamic, static, target, train_idx)
    val_dataset = TargetsDataset(dynamic, static, target, val_idx)
    test_dataset = TargetsDataset(dynamic, static, target, test_idx)
    # write train in, train out, train mask
    if False:
        datasets.write_dataset(train_dataset, Path(config["targetfolder"]) / "train.pt")
        datasets.write_dataset(val_dataset, Path(config["targetfolder"]) / "val.pt")
        datasets.write_dataset(test_dataset, Path(config["targetfolder"]) / "test.pt")
        write_config(config)
    print(f"Generated {len(train_idx)} training samples")
    
    model = load_and_prepare_model(config["model_path"]).to(torch.device("mps"))

    now = datetime.now().strftime("%m-%d-%y--%H:%M")
    experiment_folder = Path(config["out_path"]) / ("experiment" + now)


    test_loss = train(config, model, train_dataset, val_dataset, test_dataset, experiment_folder)

    with open(experiment_folder / "out.txt", "a") as outfile:
        json.dump(config, outfile)
        outfile.write(f"\nTest Loss: {test_loss}")
    
    return

if __name__ == "__main__":
    config = parse_config()

    main(config)

