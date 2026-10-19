from pathlib import Path

from nflMAE.scripts import run_pretrain, run_tackle, run_route, run_manzone

# Pretrain
scripts_path = Path(__file__).parent
config_path = scripts_path.parent.parent / "configs" / "pretrain_config.yaml"
pretrain_config = {
    "data": {
        "minimal": False,
        "force": False,
        "bdb_year": 2025,
        "data_folder": "datasets",
        "train": 0.85,
        "val": 0.15,
    },
    "train": {
        "epochs": 15,
        "batch_size": 256,
        "target_folder": "experiments",
        "lr_base": 0.001,
        "lr_final": 0.00005,
        "lr_warmup": 1000,
        "mask_players": True,
        "grid_size": 10,
        "mask_ratio_base": 0.40,
        "mask_ratio_final": 0.80,
        "reference_frame": True
    },
    "model": {
        "encode_masks": True,
        "encoder": {
            "attn_drop": .3,
            "proj_drop": .3,
            "ffn_drop": .3,
            "norm_first": True,
            "add_cls_token": True,
        },
        "decoder": {
            "attn_drop": .3,
            "proj_drop": .3,
            "ffn_drop": .3,
            "depth": 1,
            "norm_first": True,
            "add_cls_token": False
        }
    }
}
# small
pretrain_config['model']['encoder']['model_dimension'] = 32
pretrain_config['model']['encoder']['hidden_dimension'] = 128
pretrain_config['model']['encoder']['heads'] = 2
pretrain_config['model']['encoder']['depth'] = 4
pretrain_config['model']['decoder']['model_dimension'] = 32
pretrain_config['model']['decoder']['hidden_dimension'] = 128
pretrain_config['model']['decoder']['heads'] = 2
pretrain_config['train']['folder_name'] = "small"

run_pretrain.main(pretrain_config)

# large
pretrain_config['model']['encoder']['model_dimension'] = 512
pretrain_config['model']['encoder']['hidden_dimension'] = 2048
pretrain_config['model']['encoder']['heads'] = 16
pretrain_config['model']['encoder']['depth'] = 4
pretrain_config['model']['decoder']['model_dimension'] = 512
pretrain_config['model']['decoder']['hidden_dimension'] = 2048
pretrain_config['model']['decoder']['heads'] = 16
pretrain_config['train']['folder_name'] = "large"

# run_pretrain.main(pretrain_config)

# 1. Tackle position
scripts_path = Path(__file__).parent
config_dir = scripts_path.parent.parent / "configs"

config_scratch = {
    "data": {
        "bdb_year": 2024,
        "data_folder": "datasets",
        "force": True,
        "minimal": False,
        "train": .80,
        "val": .20,
        "train_weeks": [1, 2, 3, 4, 5, 6, 7, 8],
        "test_weeks": [9],
    },
    "train": {
        "target_folder": "experiments",
        "target_features": 2,
        "epochs": 15,
        "batch_size": 256,
        "lr": 0.0005,
        "use_cls_token": False,
        "patience": 2,
        "trainable": "full",
    },
    "model": {
        "attn_drop": .3,
        "proj_drop": .3,
        "ffn_drop": .3,
        "norm_first": True,
        "add_cls_token": True,
    }
}
# a. Train from scratch
config_path = config_dir / "tackle_config.yaml"

# small
config_scratch['model']['model_dimension'] = 32
config_scratch['model']['hidden_dimension'] = 128
config_scratch['model']['heads'] = 2
config_scratch['model']['depth'] = 4

run_tackle.main(config_scratch)

# large
config_scratch['model']['model_dimension'] = 512
config_scratch['model']['hidden_dimension'] = 2048
config_scratch['model']['heads'] = 16
config_scratch['model']['depth'] = 4

# run_tackle.main(config_scratch)

# b. Finetune
config_path = config_dir / "tackle_config.yaml"
config_finetune = config_scratch.copy()
config_finetune['data']['train_weeks'] = [1, 2]

# small
config_finetune['model'] = config_scratch['train']['target_folder'] + "/small/nflmae.pt"

run_tackle.main(config_finetune)

# large
config_finetune['model'] = config_scratch['train']['target_folder'] + "/large/nflmae.pt"

# run_tackle.main(config_finetune)


# 2. Route running
config_path = config_dir / "route_config.yaml"

# a. Train from scratch
config_path = config_dir / "route_config.yaml"

config_scratch['data']['bdb_year'] = 2025

# small
config_scratch['model']['model_dimension'] = 32
config_scratch['model']['hidden_dimension'] = 128
config_scratch['model']['heads'] = 2
config_scratch['model']['depth'] = 4

run_route.main(config_scratch)

# large
config_scratch['model']['model_dimension'] = 512
config_scratch['model']['hidden_dimension'] = 2048
config_scratch['model']['heads'] = 16
config_scratch['model']['depth'] = 4

# run_route.main(config_scratch)


# b. Finetune
config_path = config_dir / "route_config.yaml"
config_finetune = config_scratch.copy()
config_finetune['data']['train_weeks'] = [1, 2]

# small
config_finetune['model'] = config_scratch['train']['target_folder'] + "/small/nflmae.pt"

run_route.main(config_finetune)

# large
config_finetune['model'] = config_scratch['train']['target_folder'] + "/large/nflmae.pt"

# run_route.main(config_finetune)


# 3. Man v zone
config_path = config_dir / "manzone_config.yaml"

# a. Train from scratch
config_path = config_dir / "manzone_config.yaml"
config_scratch['data']['bdb_year'] = 2025

# small
config_scratch['model']['model_dimension'] = 32
config_scratch['model']['hidden_dimension'] = 128
config_scratch['model']['heads'] = 2
config_scratch['model']['depth'] = 4
config_scratch['train']['use_cls_token'] = True

run_manzone.main(config_scratch)

# large
config_scratch['model']['model_dimension'] = 512
config_scratch['model']['hidden_dimension'] = 2048
config_scratch['model']['heads'] = 16
config_scratch['model']['depth'] = 4

run_manzone.main(config_scratch)


# b. Finetune
config_path = config_dir / "manzone_config.yaml"
config_finetune = config_scratch.copy()
config_finetune['data']['train_weeks'] = [1, 2]
# small
config_finetune['model'] = config_scratch['train']['target_folder'] + "/small/nflmae.pt"

run_manzone.main(config_finetune)

# large
config_finetune['model'] = config_scratch['train']['target_folder'] + "/large/nflmae.pt"

# run_manzone.main(config_finetune)