from typing import Dict

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.planning.vectorized.multimodal_model import VectorizedMultiModalPlanningModel
# from l5kit.planning.vectorized.open_loop_model import VectorizedModel

import os


# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/l5kit/prediction"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("/mnt/share_disk/user/wangyuxiao/l5kit-RL-pre_train/scripts/vect_multimodal_pred_config.yaml")


# vectorization
vectorizer = build_vectorizer(cfg, dm)
# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)


# bulid the vectorized multimodal model
def build_model(cfg: Dict) -> torch.nn.Module:
    history_num_frames_ego = cfg["model_params"]["history_num_frames_ego"]
    history_num_frames_agents = cfg["model_params"]["history_num_frames_agents"]

    # number of predicted future frame states
    future_num_frames = cfg["model_params"]["future_num_frames"]
    # size of output (X, Y, Heading) in every frame
    num_outputs = cfg["model_params"]["num_outputs"]
    # number of predicted multimodal trajectories
    num_modes = cfg["model_params"]["num_modes"]

    global_head_dropout = cfg["model_params"]["global_head_dropout"]
    disable_other_agents = cfg["model_params"]["disable_other_agents"]
    disable_map = cfg["model_params"]["disable_map"]
    disable_lane_boundaries = cfg["model_params"]["disable_lane_boundaries"]

    model = VectorizedMultiModalPlanningModel(
            history_num_frames_ego=history_num_frames_ego,
            history_num_frames_agents=history_num_frames_agents,
            future_num_frames=future_num_frames,num_outputs=num_outputs,num_modes=num_modes,
            weights_scaling= [1., 1., 1.],criterion=nn.MSELoss(reduction="none"),
            global_head_dropout=global_head_dropout,disable_other_agents=disable_other_agents,
            disable_map=disable_map,disable_lane_boundaries=disable_lane_boundaries,
            )

    return model

model = build_model(cfg)
# weights_scaling = [1.0, 1.0, 1.0]
# _num_outputs = len(weights_scaling)
# _num_predicted_frames = cfg["model_params"]["future_num_frames"]
# _num_predicted_params = _num_outputs
# model = VectorizedModel(
#     history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
#     history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
#     num_targets=_num_predicted_params * _num_predicted_frames,
#     weights_scaling=weights_scaling,
#     criterion=nn.L1Loss(reduction="none"),
#     global_head_dropout=cfg["model_params"]["global_head_dropout"],
#     disable_other_agents=cfg["model_params"]["disable_other_agents"],
#     disable_map=cfg["model_params"]["disable_map"],
#     disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
# )

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
        path_to_save = f"{os.path.dirname(__file__)}/vect_multimodal_pred_model_{step}_{loss}.pt"
        to_save.save(path_to_save)
        # torch.jit.save(to_save, path_to_save)
        print(f"MODEL STORED at {path_to_save}")
