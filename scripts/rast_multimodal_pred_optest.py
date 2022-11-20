from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points, angular_distance
from l5kit.visualization import TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR, draw_trajectory, write_video
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

import os

import cv2

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/l5kit/prediction"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("/mnt/share_disk/user/wangyuxiao/l5kit-RL-pre_train/scripts/rast_multimodal_pred_config.yaml")


# load model
# model_path = "/tmp/rast_multimodal_pred_model.pt"
model_path = f"{os.path.dirname(__file__)}/rast_multimodal_pred_model_99000_0.2505820393562317.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path).to(device)
model = model.eval()


# load evaluation data
eval_cfg = cfg["val_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
# ===== INIT DATASET
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                             num_workers=eval_cfg["num_workers"])
print(eval_dataset)


# # evaluating loop
# position_preds = []
# yaw_preds = []

# position_gts = []
# yaw_gts = []

# torch.set_grad_enabled(False)

# for idx_data, data in enumerate(tqdm(eval_dataloader)):
#     data = {k: v.to(device) for k, v in data.items()}
#     result = model(data)
#     position_preds.append(result["positions"].detach().cpu().numpy())
#     yaw_preds.append(result["yaws"].detach().cpu().numpy())

#     position_gts.append(data["target_positions"].detach().cpu().numpy())
#     yaw_gts.append(data["target_yaws"].detach().cpu().numpy())
#     if idx_data == 10:
#         break

# position_preds = np.concatenate(position_preds)
# yaw_preds = np.concatenate(yaw_preds)

# position_gts = np.concatenate(position_gts)
# yaw_gts = np.concatenate(yaw_gts)


# visualise results
for frame_number in range(0, len(eval_dataset), len(eval_dataset) // 100):  # //20 10
    
    data = eval_dataloader.dataset[frame_number]
    data_batch = default_collate([data])
    data_batch = {k: v.to(device) for k, v in data_batch.items()}
    result = model(data_batch)
    predicted_positions_modes = result["positions_all"].detach().cpu().numpy().squeeze()

    im_ego = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    # im_ego0 = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    # im_ego1 = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    # im_ego2 = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    # im_ego = [im_ego0, im_ego1, im_ego2]
    target_positions = data["target_positions"]

    # draw target trajectory
    target_positions = transform_points(target_positions, data["raster_from_agent"])
    draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)

    # draw multimodal predicted trajectories
    num_modes = cfg["model_params"]["num_modes"]
    for mode in range(num_modes):
        predicted_positions = predicted_positions_modes[mode,:,:]
        predicted_positions = transform_points(predicted_positions, data["raster_from_agent"])
        draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)
        # cv2.imwrite(f"{os.path.dirname(__file__)}/rast_multimodal_pred_image{frame_number}{mode}.jpg",im_ego[mode])
    
    # plt.imshow(im_ego)
    # plt.axis("off")
    # plt.show()
    cv2.imwrite(f"{os.path.dirname(__file__)}/rast_multimodal_pred_image{frame_number}.jpg",im_ego)

