import os
# from pycharm
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tempfile import gettempdir

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDatasetVectorized
from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.vectorization.vectorizer_builder import build_vectorizer

import sys
from pathlib import Path
project_path = str(Path(__file__).parents[1])
print("project path: ", project_path)
sys.path.append(project_path)
print(sys.path)

from scripts.vectorized_offlin_rl_model import VectorOfflineRLModel


# prepare data path and load cfg
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/prediction"

dm = LocalDataManager(None)
# get config

# Home path
from pathlib import Path
print(Path.home())
# Project path
project_path = Path("/mnt/share_disk/user/daixingyuan/l5kit/")

dm = LocalDataManager(None)
# get config
cfg = load_config_data(str(Path(project_path, "examples/urban_driver/config.yaml")))

# ===== INIT DATASET
# cfg["train_data_loader"]["key"] = "train.zarr"
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()

vectorizer = build_vectorizer(cfg, dm)
train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)
print(train_dataset)


URBAN_DRIVER = "Urban Driver"
OPEN_LOOP_PLANNER = "Open Loop Planner"
OFFLINE_RL_PLANNER = "Offline RL Planner"

# model_name = URBAN_DRIVER
# model_name = OPEN_LOOP_PLANNER
model_name = OFFLINE_RL_PLANNER

weights_scaling = [1.0, 1.0, 1.0]

_num_predicted_frames = cfg["model_params"]["future_num_frames"]
_num_predicted_params = len(weights_scaling)


if model_name == URBAN_DRIVER:
    model = VectorizedUnrollModel(
        history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        num_targets=_num_predicted_params * _num_predicted_frames,
        weights_scaling=weights_scaling,
        criterion=nn.L1Loss(reduction="none"),
        global_head_dropout=cfg["model_params"]["global_head_dropout"],
        disable_other_agents=cfg["model_params"]["disable_other_agents"],
        disable_map=cfg["model_params"]["disable_map"],
        disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
        detach_unroll=cfg["model_params"]["detach_unroll"],
        warmup_num_frames=cfg["model_params"]["warmup_num_frames"],
        discount_factor=cfg["model_params"]["discount_factor"],
    )

elif model_name == OPEN_LOOP_PLANNER:
    model = VectorizedModel(
        history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        num_targets=_num_predicted_params * _num_predicted_frames,
        weights_scaling=weights_scaling,
        criterion=nn.L1Loss(reduction="none"),
        global_head_dropout=cfg["model_params"]["global_head_dropout"],
        disable_other_agents=cfg["model_params"]["disable_other_agents"],
        disable_map=cfg["model_params"]["disable_map"],
        disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
    )
elif model_name == OFFLINE_RL_PLANNER:
    model = VectorOfflineRLModel(
        history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        num_targets=_num_predicted_params * _num_predicted_frames,
        weights_scaling=weights_scaling,
        criterion=nn.L1Loss(reduction="none"),
        global_head_dropout=cfg["model_params"]["global_head_dropout"],
        disable_other_agents=cfg["model_params"]["disable_other_agents"],
        disable_map=cfg["model_params"]["disable_map"],
        disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
    )
else:
    raise ValueError(f"{model_name=} is invalid")


# prepare for training
train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training loops
tr_it = iter(train_dataloader)
cfg["train_params"]["max_num_steps"] = int(1e8)
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []
model.train()
torch.set_grad_enabled(True)

for _ in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    # Forward pass
    data = {k: v.to(device) for k, v in data.items()}
    result = model(data)
    loss = result["loss"]
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")


# save model
# to_save = torch.jit.script(model.cpu())
# path_to_save = Path(project_path, "tmp", f"urban_driver_train_iter_{cfg['train_params']['max_num_steps']}.pt")
# to_save.save(path_to_save)
# print(f"MODEL STORED at {path_to_save}")