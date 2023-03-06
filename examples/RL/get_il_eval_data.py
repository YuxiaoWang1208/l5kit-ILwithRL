import os

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
import torch as th
from torch.utils.data import DataLoader, ConcatDataset

import numpy as np

from l5kit.kinematic import AckermanPerturbation
from l5kit.random.random_generator import GaussianRandomGenerator

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

# IL env config
dm = LocalDataManager(None)
# config_path = '/root/zhufenghua12/wangyuxiao/l5kit-wyx/examples/RL/gym_config.yaml'
config_path = os.environ["CONFIG_PATH"]
cfg = load_config_data(config_path)
batch_size = 64

# rasterisation
rasterizer = build_rasterizer(cfg, dm)
raster_size = cfg["raster_params"]["raster_size"][0]
n_channels = rasterizer.num_channels()

# load dataset for imitation learning
loader_key = cfg["val_data_loader"]["key"]
dataset_zarr = ChunkedDataset(dm.require(loader_key)).open()

# add perturbation
mean_value=np.array([0.0,0.0,0.0])
std_value2 = np.array([1, 1, np.pi / 3])
AckermanPerturbation2 = AckermanPerturbation(
    random_offset_generator=GaussianRandomGenerator(mean=mean_value, std=std_value2), perturb_prob=0.3)
if cfg['val_data_loader']['perturbation']:
    perturbation = AckermanPerturbation2
else:
    perturbation = None
# dataset = EgoDataset(cfg, dataset_zarr, rasterizer, perturbation=AckermanPerturbation2)

dataset = EgoDataset(cfg, dataset_zarr, rasterizer, perturbation=perturbation)
if cfg["gym_params"]["max_scene_id"] < len(dataset):
    data_list = []
    num_of_scenes = 0
    for scene_id in range(cfg["gym_params"]["max_val_scene_id"]):
        scene_1 = dataset.get_scene_dataset(scene_id)
        data_list.append(scene_1)
        num_of_scenes += 1  # 累计有多少个场景
    dataset = ConcatDataset(data_list)
    print('num_of_scenes:', num_of_scenes)
if cfg["gym_params"]["overfit"]:  # overfit
    scene_index = cfg["gym_params"]["overfit_id"]
    dataset = dataset.get_scene_dataset(scene_index=scene_index)  # 单场景数据集
print(dataset)

# eval_cfg = cfg["val_data_loader"]
# dataloader = DataLoader(dataset, shuffle=eval_cfg["shuffle"], batch_size=int(batch_size/4),
#                                 num_workers=eval_cfg["num_workers"])
# print("Imitation data has been loaded.")
# data_it = iter(dataloader)

# # print(dataset.dataset.frames)

# def get_data():
#     global data_it
#     try:
#         il_data_buffer = next(data_it)
#     except StopIteration:
#         data_it = iter(dataloader)
#         il_data_buffer = next(data_it)
#     # to cuda device                  
#     il_data_buffer = {k: v.to(device) for k, v in il_data_buffer.items()}

#     return il_data_buffer

def get_frame_data(scene_id, n):
    frame_data = dataset.datasets[scene_id].get_frame(0, n)
    # to cuda device                  
    # frame_data = {k: th.tensor(v).to(device) for k, v in frame_data.items()}
    return frame_data

def un_rescale(env, action):
    # get resacle params and rescale the targets
    rescale_action = env.get_attr('rescale_action')[0]
    use_kinematic = env.get_attr('use_kinematic')[0]
    if rescale_action:
        if use_kinematic:
            kin_rescale = env.get_attr('kin_rescale')[0]
            action[..., 0] = action[..., 0] / kin_rescale.steer_scale
            action[..., 1] = action[..., 1] / kin_rescale.acc_scale
        else:
            non_kin_rescale = env.get_attr('non_kin_rescale')[0]
            action[..., 0] = (action[..., 0] - non_kin_rescale.x_mu) / non_kin_rescale.x_scale
            action[..., 1] = (action[..., 1] - non_kin_rescale.y_mu) / non_kin_rescale.y_scale
            action[..., 2] = (action[..., 2] - non_kin_rescale.yaw_mu) / non_kin_rescale.yaw_scale
    
    return action

def rescale(env, action):
    # get resacle params and rescale the targets
    rescale_action = env.get_attr('rescale_action')[0]
    use_kinematic = env.get_attr('use_kinematic')[0]
    if rescale_action:
        if use_kinematic:
            kin_rescale = env.get_attr('kin_rescale')[0]
            action[..., 0] = kin_rescale.steer_scale * action[..., 0]
            action[..., 1] = kin_rescale.acc_scale * action[..., 1]
        else:
            non_kin_rescale = env.get_attr('non_kin_rescale')[0]
            action[..., 0] = non_kin_rescale.x_mu + non_kin_rescale.x_scale * action[..., 0]
            action[..., 1] = non_kin_rescale.y_mu + non_kin_rescale.y_scale * action[..., 1]
            action[..., 2] = non_kin_rescale.yaw_mu + non_kin_rescale.yaw_scale * action[..., 2]
    
    return action
