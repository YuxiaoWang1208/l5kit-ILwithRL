from distutils.command.build import build
from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from l5kit.planning.rasterized.multimodal_model import RasterizedMultiModalPlanningModel
from prettytable import PrettyTable
from pathlib import Path

import os


# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/l5kit/prediction"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("/mnt/share_disk/user/wangyuxiao/l5kit-RL-master/scripts/rast_multimodal_pred_config.yaml")


# rasterisation, perturbation not included
rasterizer = build_rasterizer(cfg, dm)
# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = EgoDataset(cfg, train_zarr, rasterizer)


# bulid the rasterized multimodal model
def build_model(cfg: Dict) -> torch.nn.Module:
    # change input channels number to match the rasterizer's output
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels

    # number of predicted future frame states
    future_num_frames = cfg["model_params"]["future_num_frames"]
    # size of output (X, Y, Heading) in every frame
    num_outputs = cfg["model_params"]["num_outputs"]
    # number of predicted multimodal trajectories
    num_modes = cfg["model_params"]["num_modes"]

    model = RasterizedMultiModalPlanningModel(model_arch="resnet50",num_input_channels=num_in_channels,
            future_num_frames=future_num_frames,num_outputs=num_outputs,num_modes=num_modes,
            weights_scaling= [1., 1., 1.],criterion=nn.MSELoss(reduction="none"),pretrained=True)

    return model

model = build_model(cfg)
# print(model)


# prepare to train
train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(train_dataset)


# training loop
tr_it = iter(train_dataloader)
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []
model.train()
torch.set_grad_enabled(True)

checkpoint_every_n_steps = cfg["train_params"]["checkpoint_every_n_steps"]
eval_every_n_steps = cfg["train_params"]["eval_every_n_steps"]

for step in progress_bar:
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

    if step % checkpoint_every_n_steps == 0:
        # store the model
        to_save = torch.jit.script(model)
        path_to_save = f"{os.path.dirname(__file__)}/rast_multimodal_pred_model_{step}_{loss}.pt"
        to_save.save(path_to_save)
        # torch.jit.save(to_save, path_to_save)
        print(f"MODEL STORED at {path_to_save}")
