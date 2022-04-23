import matplotlib.pyplot as plt

import numpy as np

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

import os

from l5kit.visualization.visualizer.zarr_utils import zarr_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI

from IPython.display import display, clear_output
import PIL


# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
from pathlib import Path
# get config
project_path = Path("/storage/guest/dxy/parallel_intelligence_planning/l5kit/")

# get config
cfg = load_config_data(Path(project_path, "examples/visualisation", "./visualisation_config.yaml"))

# cfg = load_config_data("./visualisation_config.yaml")
print(cfg)


dm = LocalDataManager()
# dataset_path = dm.require(cfg["val_data_loader"]["key"])
dataset_path = dm.require(cfg["train_data_loader"]["key"])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)

cfg["raster_params"]["map_type"] = "py_semantic"
rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)
scene_idx = 1
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])
    clear_output(wait=True)
    display(PIL.Image.fromarray(im))


# 选择带有红绿灯的场景！
print("scene 0")
traffic_light_scene_id = []

total_scene = 100

for i in range(total_scene):
    scene_1 = dataset.get_scene_dataset(i)
    #     print(scene_1)
    #     print(len(scene_1))
    if len(scene_1.dataset.tl_faces) > 0:
        traffic_light_scene_id.append(i)

print("total scenes: ", len(traffic_light_scene_id))
print(traffic_light_scene_id)
print(traffic_light_scene_id[:10])

output_notebook()
mapAPI = MapAPI.from_cfg(dm, cfg)
for scene_idx in traffic_light_scene_id[:10]:
    out = zarr_to_visualizer_scene(zarr_dataset.get_scene_dataset(scene_idx), mapAPI)
    out_vis = visualize(scene_idx, out)
    show(out_vis)