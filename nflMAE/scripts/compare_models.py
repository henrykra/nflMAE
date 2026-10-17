import subprocess
import yaml
from pathlib import Path
import sys

# Pretrain
config_path = Path("/Users/henrykraessig/code/route2vec/config.yaml")
pretrain_config = {
    "data": {
        "minimal": True,
        "force": False,
        "bdb_folder": "/Users/henrykraessig/code/bdb2024/data",
        "data_folder": "/Users/henrykraessig/code/route2vec/data",
        "train": 0.85,
        "val": 0.15,
    },
    "train": {
        "epochs": 15,
        "batch_size": 256,
        "target_folder": "/Users/henrykraessig/code/route2vec/experiments",
        "lr_base": 0.001,
        "lr_final": 0.00005,
        "lr_warmup": 1000,
        "mask_players": True,
        "grid_size": 10,
        "mask_ratio": 0.75,
        "reference_frame": True
    },
    "model": {
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
pretrain_config['model']['encoder']['dimension'] = 32
pretrain_config['model']['encoder']['hidden_dimension'] = 128
pretrain_config['model']['encoder']['heads'] = 2
pretrain_config['model']['encoder']['depth'] = 4
pretrain_config['model']['decoder']['dimension'] = 32
pretrain_config['model']['decoder']['hidden_dimension'] = 128
pretrain_config['model']['decoder']['heads'] = 2
pretrain_config['folder_name'] = "small"


with config_path.open('w') as configstream:
    yaml.dump(pretrain_config, configstream)
subprocess.run([sys.executable, "run_pretrain.py"])

# large
pretrain_config['model']['encoder']['dimension'] = 512
pretrain_config['model']['encoder']['hidden_dimension'] = 2048
pretrain_config['model']['encoder']['heads'] = 16
pretrain_config['model']['encoder']['depth'] = 4
pretrain_config['model']['decoder']['dimension'] = 512
pretrain_config['model']['decoder']['hidden_dimension'] = 2048
pretrain_config['model']['decoder']['heads'] = 16
pretrain_config['folder_name'] = "large"

with config_path.open('w') as configstream:
    yaml.dump(pretrain_config, configstream)
subprocess.run([sys.executable, "run_pretrain.py"])


# 1. Tackle position
config_dir = Path("/Users/henrykraessig/code/route2vec/configs")

config_scratch = {
    "data": {
        "bdb_folder": "/Users/henrykraessig/code/bdb2024/data",
        "data_folder": "/Users/henrykraessig/code/route2vec/data",
        "force": True,
        "minimal": True,
        "train": .80,
        "val": .20,
        "train_weeks": [1, 2],
        "test_weeks": [9],
    },
    "train": {
        "target_folder": "/Users/henrykraessig/code/route2vec/experiments",
        "target_features": 2,
        "epochs": 25,
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
config_path = config_dir / "tackle_prediction_config.yaml"

# small
config_scratch['model']['model_dimension'] = 32
config_scratch['model']['hidden_dimension'] = 128
config_scratch['model']['heads'] = 2
config_scratch['model']['depth'] = 4

with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_tackle.py"])

# large
config_scratch['model']['model_dimension'] = 512
config_scratch['model']['hidden_dimension'] = 2048
config_scratch['model']['heads'] = 16
config_scratch['model']['depth'] = 4

with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_tackle.py"])


# b. Finetune
config_path = config_dir / "tackle_prediction_config.yaml"

# small
config_scratch['model'] = config_scratch['train']['target_folder'] + "/small/rest.pt"
with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_tackle.py"])

# large
config_scratch['model'] = config_scratch['train']['target_folder'] + "/large/rest.pt"
with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_tackle.py"])


# 2. Route running
config_path = config_dir / "route_config.yaml"

config_scratch = {
    "data": {
        "bdb_folder": "/Users/henrykraessig/code/bdb2025/data",
        "data_folder": "/Users/henrykraessig/code/route2vec/data",
        "force": True,
        "minimal": True,
        "train": .80,
        "val": .20,
        "train_weeks": [1, 2],
        "test_weeks": [9],
    },
    "train": {
        "target_folder": "/Users/henrykraessig/code/route2vec/experiments",
        "target_features": 13,
        "epochs": 25,
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
config_path = config_dir / "route_config.yaml"

# small
config_scratch['model']['model_dimension'] = 32
config_scratch['model']['hidden_dimension'] = 128
config_scratch['model']['heads'] = 2
config_scratch['model']['depth'] = 4

with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_route.py"])

# large
config_scratch['model']['model_dimension'] = 512
config_scratch['model']['hidden_dimension'] = 2048
config_scratch['model']['heads'] = 16
config_scratch['model']['depth'] = 4

with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_route.py"])


# b. Finetune
config_path = config_dir / "route_config.yaml"

# small
config_scratch['model'] = config_scratch['train']['target_folder'] + "/small/rest.pt"
with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_route.py"])

# large
config_scratch['model'] = config_scratch['train']['target_folder'] + "/large/rest.pt"
with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_route.py"])


# 3. Man v zone
config_path = config_dir / "manzone_config.yaml"

config_scratch = {
    "data": {
        "bdb_folder": "/Users/henrykraessig/code/bdb2025/data",
        "data_folder": "/Users/henrykraessig/code/route2vec/data",
        "force": True,
        "minimal": True,
        "train": .80,
        "val": .20,
        "train_weeks": [1, 2],
        "test_weeks": [9],
    },
    "train": {
        "target_folder": "/Users/henrykraessig/code/route2vec/experiments",
        "target_features": 3,
        "epochs": 25,
        "batch_size": 256,
        "lr": 0.0005,
        "use_cls_token": True,
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
config_path = config_dir / "manzone_config.yaml"

# small
config_scratch['model']['model_dimension'] = 32
config_scratch['model']['hidden_dimension'] = 128
config_scratch['model']['heads'] = 2
config_scratch['model']['depth'] = 4

with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_manzone.py"])

# large
config_scratch['model']['model_dimension'] = 512
config_scratch['model']['hidden_dimension'] = 2048
config_scratch['model']['heads'] = 16
config_scratch['model']['depth'] = 4

with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_manzone.py"])


# b. Finetune
config_path = config_dir / "manzone_config.yaml"

# small
config_scratch['model'] = config_scratch['train']['target_folder'] + "/small/rest.pt"
with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_manzone.py"])

# large
config_scratch['model'] = config_scratch['train']['target_folder'] + "/large/rest.pt"
with config_path.open('w') as configstream:
    yaml.dump(config_scratch, configstream)
subprocess.run([sys.executable, "run_manzone.py"])