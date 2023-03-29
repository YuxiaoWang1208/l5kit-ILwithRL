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
loader_key = cfg["train_data_loader"]["key"]
dataset_zarr = ChunkedDataset(dm.require(loader_key)).open()

# add perturbation
mean_value=np.array([0.0,0.0,0.0])
std_value2 = np.array([1, 1, np.pi / 3])
AckermanPerturbation2 = AckermanPerturbation(
    random_offset_generator=GaussianRandomGenerator(mean=mean_value, std=std_value2), perturb_prob=0.3)
if cfg['train_data_loader']['perturbation']:
    perturbation = AckermanPerturbation2
else:
    perturbation = None
# dataset = EgoDataset(cfg, dataset_zarr, rasterizer, perturbation=AckermanPerturbation2)

dataset = EgoDataset(cfg, dataset_zarr, rasterizer, perturbation=perturbation)
dataset = EgoDataset(cfg, dataset_zarr, rasterizer,)
# if cfg["gym_params"]["max_scene_id"] < len(dataset):
#     data_list = []
#     num_of_scenes = 0
#     for scene_id in range(cfg["gym_params"]["max_scene_id"]):
#         scene_1 = dataset.get_scene_dataset(scene_id)
#         data_list.append(scene_1)
#         num_of_scenes += 1  # 累计有多少个场景
#     dataset = ConcatDataset(data_list)
#     print('num_of_scenes:', num_of_scenes)
# if cfg["gym_params"]["overfit"]:  # overfit
#     scene_index = cfg["gym_params"]["overfit_id"]
#     dataset = dataset.get_scene_dataset(scene_index=scene_index)  # 单场景数据集
# print(dataset)


data_list = []
num_of_scenes = 0
for scene_id in [6]:
    scene_1 = dataset.get_scene_dataset(scene_id)
    data_list.append(scene_1)
    num_of_scenes += 1  # 累计有多少个场景
dataset = ConcatDataset(data_list)
print('num_of_scenes:', num_of_scenes)
# dataset = dataset.get_scene_dataset(scene_index=6)  # 单场景数据集


train_cfg = cfg["train_data_loader"]
dataloader = DataLoader(dataset, shuffle=train_cfg["shuffle"], batch_size=batch_size, # int(batch_size/4),
                                num_workers=train_cfg["num_workers"])
print("Imitation data has been loaded.")
data_it = iter(dataloader)

# print(dataset.dataset.frames)

def get_data():
    global data_it
    try:
        il_data_buffer = next(data_it)
    except StopIteration:
        data_it = iter(dataloader)
        il_data_buffer = next(data_it)

    # # ==== Turns  info ====
    # il_data_buffer["image"] = il_data_buffer["image"].cpu().numpy()
    # image_shape = np.array(il_data_buffer["image"].shape)[1:]
    # image_shape[0] = 1
    # turns_channel = np.zeros(shape=image_shape, dtype=np.uint8)
    # new_img_buffer_shape = np.array(il_data_buffer["image"].shape)
    # new_img_buffer_shape[1] += 1
    # new_img_buffer = np.zeros(shape=new_img_buffer_shape, dtype=np.uint8)
    # # 利用categorize_scenes的函数实时打turns标签
    # yaws_to_judge = il_data_buffer["future_yaws"].cpu().numpy()  # tensor(batch_size, turns_num_frames, 1)
    # TURN_THRESH1 = 0.5236  # Threshold in rads to determine front or side front
    # TURN_THRESH2 = 1.0472  # Threshold in rads to determine side front or side
    # batch_size = len(yaws_to_judge)
    # for i in range(batch_size):
    #     yaws_to_judge_i = yaws_to_judge[i, ...]
    #     yaw_diff = yaws_to_judge_i[1:] - yaws_to_judge_i[:-1]
    #     if TURN_THRESH2 >= np.sum(yaw_diff) >= TURN_THRESH1:
    #         turns = "left front"
    #         turns_channel[:image_shape[0], :int(image_shape[1]/2), :image_shape[2]].fill(1)
    #         turns_channel[:image_shape[0], :image_shape[1], :int(image_shape[2]/2)].fill(1)
    #     elif -TURN_THRESH2 <= np.sum(yaw_diff) <= -TURN_THRESH1:
    #         turns = "right front"
    #         turns_channel[:image_shape[0], :int(image_shape[1]/2), :image_shape[2]].fill(1)
    #         turns_channel[:image_shape[0], :image_shape[1], int(image_shape[2]/2):].fill(1)
    #     elif np.sum(yaw_diff) >= TURN_THRESH2:
    #         turns = "left"
    #         turns_channel[:image_shape[0], :image_shape[1], :int(image_shape[2]/2)].fill(1)
    #     elif np.sum(yaw_diff) <= -TURN_THRESH2:
    #         turns = "right"
    #         turns_channel[:image_shape[0], :image_shape[1], int(image_shape[2]/2):].fill(1)
    #     else:
    #         turns = "front"
    #         turns_channel[:image_shape[0], :int(image_shape[1]/2), :image_shape[2]].fill(1)
    #     new_img_buffer[i] = np.concatenate([il_data_buffer["image"][i], turns_channel], axis=0)
    # il_data_buffer["image"] = th.from_numpy(new_img_buffer)
    # # print(agents_past_polys[0, 0, -1])

    # to cuda device                  
    il_data_buffer = {k: v.to(device) for k, v in il_data_buffer.items()}
    # from get_il_eval_data import get_frame_data as get_frame_data1
    # aa = get_frame_data1(14, il_data_buffer['frame_index'][0])

    return il_data_buffer

def get_frame_data(scene_id, n):
    frame_data = dataset.datasets[scene_id].get_frame(0, n)
    # to cuda device                  
    # frame_data = {k: th.tensor(v).to(device) for k, v in frame_data.items()}
    return frame_data
