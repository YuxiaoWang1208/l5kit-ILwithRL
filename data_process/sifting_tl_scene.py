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

# from torch.utils.tensorboard import SummaryWriter

import sys
from pathlib import Path
# project_path = str(Path(__file__).parents[1])
project_path = "/mnt/share_disk/user/huangjun/l5kit"
print("project path: ", project_path)
sys.path.append(project_path)
print(sys.path)

# from scripts.vectorized_offlin_rl_model import VectorOfflineRLModel

# prepare data path and load cfg
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/prediction"

dm = LocalDataManager(None)
# get config

# Home path
from pathlib import Path
# print(Path.home())
# Project path


dm = LocalDataManager(None)
# get config
cfg = load_config_data(str(Path(project_path, "examples/urban_driver/config.yaml")))

# ===== INIT DATASET
# cfg["train_data_loader"]["key"] = "train.zarr"
# dataset_path = dm.require(cfg["train_data_loader"]["key"])
# cfg["train_full_data_loader"]["key"] = "train_full.zarr"
dataset_path = dm.require(cfg["train_full_data_loader"]["key"])

train_zarr = ChunkedDataset(dataset_path).open()
vectorizer = build_vectorizer(cfg, dm)
train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)
# train_dataset = train_dataset.get_scene_dataset(1)

print(train_zarr)
# print(train_dataset)

#train_full.zarr
def get_tl_scene_id(_dataset, num_scene_return=134622):
#train.zarr
# def get_tl_scene_id(_dataset, num_scene_return=16265):

    _traffic_light_scene_id = []
    traffic_light_active_threshold = 0.05

    for i in range(_dataset.scenes.size):
        scene_i = _dataset.get_scene_dataset(i)
        tl_faces_in_scene = scene_i.tl_faces

        num_tl_faces_of_scene = len(tl_faces_in_scene)
        if num_tl_faces_of_scene == 0:
            continue

        traffic_light_status = [sc[2] for sc in tl_faces_in_scene]
        # https://woven-planet.github.io/l5kit/data_format.html?highlight=light#traffic-light-faces
        # https://github.com/woven-planet/l5kit/blob/71eb4dae2c8230f7ca2c60e5b95473c91c715bb8/l5kit/l5kit/data/labels.py#L22
        traffic_light_status = np.mean(traffic_light_status, axis=0)  # active, inactive, unknown
        traffic_light_active_ratio = traffic_light_status[0]
        if num_tl_faces_of_scene > 0 and traffic_light_active_ratio > traffic_light_active_threshold:
            _traffic_light_scene_id.append(i)
            if len(_traffic_light_scene_id) >= num_scene_return:
                # print("light id", len(_traffic_light_scene_id))
                break

    return _traffic_light_scene_id

traffic_light_scene_id = get_tl_scene_id(train_zarr)
print("total scenes: ", len(traffic_light_scene_id))
print(traffic_light_scene_id)

#Visualisation
output_notebook()
mapAPI = MapAPI.from_cfg(dm, cfg)
for scene_idx in traffic_light_scene_id[:]:
    out = zarr_to_visualizer_scene(train_zarr.get_scene_dataset(scene_idx), mapAPI)
    out_vis = visualize(scene_idx, out)
    show(out_vis)

