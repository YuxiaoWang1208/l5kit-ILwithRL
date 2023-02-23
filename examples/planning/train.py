import sys
project_path = "/root/zhufenghua12/wangyuxiao/l5kit-wyx/l5kit"
print("project path: ", project_path)
sys.path.append(project_path)

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

from l5kit.environment.envs.l5_env import SimulationConfigGym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import os


# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/root/zhufenghua12/l5kit/prediction"
os.chdir("/root/zhufenghua12/wangyuxiao/l5kit-wyx/examples/planning")
if "L5KIT_DATA_FOLDER" not in os.environ:
    raise KeyError("L5KIT_DATA_FOLDER environment variable not set")

config_path = os.getcwd() + "/config.yaml"
os.environ.setdefault('CONFIG_PATH', config_path)

dm = LocalDataManager(None)
# get config
cfg = load_config_data("./config.yaml")


perturb_prob = cfg["train_data_loader"]["perturb_probability"]

# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)
mean = np.array([0.0, 0.0, 0.0])  # lateral, longitudinal and angular
std = np.array([0.5, 1.5, np.pi / 6])
perturbation = AckermanPerturbation(
        random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=perturb_prob)

# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
# train_dataset = EgoDataset(cfg, train_zarr, rasterizer, perturbation)
train_dataset = EgoDataset(cfg, train_zarr, rasterizer)

# # plot same example with and without perturbation
# for perturbation_value in [1, 0]:
#     perturbation.perturb_prob = perturbation_value

#     data_ego = train_dataset[0]
#     im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
#     target_positions = transform_points(data_ego["target_positions"], data_ego["raster_from_agent"])
#     draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
#     plt.imshow(im_ego)
#     plt.axis('off')
#     plt.show()

# # before leaving, ensure perturb_prob is correct
# perturbation.perturb_prob = perturb_prob


# make train env
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = 128 + 1
train_sim_cfg.use_agents_gt = True
env_kwargs = {'env_config_path': "./config.yaml", 'use_kinematic': False, 'train': True,
                'sim_cfg': train_sim_cfg, 'simnet_model_path': None}
env = make_vec_env("L5-CLE-v0", env_kwargs=env_kwargs, n_envs=4,
                    vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

# get resacle params and rescale the targets
rescale_action = env.get_attr('rescale_action')[0]
use_kinematic = env.get_attr('use_kinematic')[0]
if rescale_action:
    if use_kinematic:
        kin_rescale = env.get_attr('kin_rescale')[0]
        rescale_paras = dict(x_mu=None, y_mu=None, yaw_mu=None, x_scale=None, y_scale=None, yaw_scale=None,
                               steer_scale=kin_rescale.steer_scale, acc_scale=kin_rescale.acc_scale)
    else:
        non_kin_rescale = env.get_attr('non_kin_rescale')[0]
        rescale_paras = dict(x_mu=non_kin_rescale.x_mu, y_mu=non_kin_rescale.y_mu, yaw_mu=non_kin_rescale.yaw_mu,
                               x_scale=non_kin_rescale.x_scale, y_scale=non_kin_rescale.y_scale,
                               yaw_scale=non_kin_rescale.yaw_scale, steer_scale=None, acc_scale=None)

model = RasterizedPlanningModel(
        rescale_action,
        use_kinematic,
        model_arch="resnet50",
        num_input_channels=rasterizer.num_channels(),
        num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states,
        weights_scaling= [1., 1., 1.],
        criterion=nn.MSELoss(reduction="none")
        )
print(model)

if cfg["gym_params"]["overfit"]:  # overfit
    scene_index = cfg["gym_params"]["overfit_id"]
    train_dataset = train_dataset.get_scene_dataset(scene_index=scene_index)  # 单场景数据集

train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
print(train_dataset)


tr_it = iter(train_dataloader)
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []
model.train()
torch.set_grad_enabled(True)

save_freq = 1000

for n_iter in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    # Forward pass
    data = {k: v.to(device) for k, v in data.items()}
    result = model(data, rescale_paras)
    loss = result["loss"]
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    # save the model
    if n_iter % save_freq == 0:
        to_save = torch.jit.script(model)
        path_to_save = f"{os.getcwd()}/models/planning_model{n_iter}.pt"
        to_save.save(path_to_save)
        print(f"MODEL STORED at {path_to_save}")

