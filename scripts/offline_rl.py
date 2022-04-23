import os
import sys
from pathlib import Path

# from pycharm
import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset
from l5kit.data import LocalDataManager
from l5kit.dataset import EgoDatasetVectorized
from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

project_path = str(Path(__file__).parents[1])
print("project path: ", project_path)
sys.path.append(project_path)
print(sys.path)

from scripts.vectorized_offlin_rl_model import VectorOfflineRLModel
from pathlib import Path

# prepare data path and load cfg

os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/prediction"


# print(Path.home())
# Project path
# project_path = Path("/mnt/share_disk/user/daixingyuan/l5kit/")


def load_dataset(cfg, traffic_signal_scene_id):
    dm = LocalDataManager(None)
    # ===== INIT DATASET
    # cfg["train_data_loader"]["key"] = "train.zarr"
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()

    vectorizer = build_vectorizer(cfg, dm)
    train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)

    # todo demo for single scene
    train_dataset = train_dataset.get_scene_dataset(traffic_signal_scene_id)
    print(train_dataset)
    return train_dataset


URBAN_DRIVER = "Urban Driver"
OPEN_LOOP_PLANNER = "Open Loop Planner"
OFFLINE_RL_PLANNER = "Offline RL Planner"


# model_name = URBAN_DRIVER
# model_name = OPEN_LOOP_PLANNER


def load_model(model_name):
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
            cfg=cfg
        )
    else:
        raise ValueError(f"{model_name=} is invalid")

    return model


def init_logger(log_name):
    # tensorboard for log
    # log_id = "7-debug"
    log_id = (
        f"signal_scene_{log_name['traffic_signal_scene_id']}"
        f"-il_weight_{log_name['imitate_loss_weight']}"
        f"-pred_weight_{log_name['pred_loss_weight']}"
        f"-1"
    )
    log_dir = Path(project_path, "logs")
    writer = SummaryWriter(log_dir=f"{log_dir}/{model_name}-{log_id}")
    return writer


def train(model, train_dataset, cfg, writer):
    # todo
    cfg["train_params"]["max_num_steps"] = int(1e8)

    train_cfg = cfg["train_data_loader"]
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                                  num_workers=train_cfg["num_workers"])
    losses_train = []
    # training loops
    tr_it = iter(train_dataloader)

    # prepare for training
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    torch.set_grad_enabled(True)

    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    for n_iter in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
        # Forward pass
        data = {k: v.to(device) for k, v in data.items()}
        result = model(data)
        loss = result["loss"]

        writer.add_scalar('Loss/train', loss.item(), n_iter)
        writer.add_scalar('Loss/train_policy_loss', result["loss_imitate"].item(), n_iter)
        writer.add_scalar('Loss/train_prediction_loss', result["loss_other_agent_pred"].item(), n_iter)

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


if __name__ == '__main__':
    import argparse

    import argparse
    import os

    os.environ["_TEST_TUNE_TRIAL_UUID"] = "_"  # 在log路径不包含uuid, 这样可以是文件夹完全按照创建时间排序

    parser = argparse.ArgumentParser()
    parser.add_argument("--imitate_loss_weight", type=float, default=1.0)
    parser.add_argument("--pred_loss_weight", type=float, default=1.0)
    parser.add_argument("--flag", type=str)

    args = parser.parse_args()
    imitate_loss_weight = args.imitate_loss_weight
    pred_loss_weight = args.pred_loss_weight

    cfg = load_config_data(str(Path(project_path, "examples/urban_driver/config.yaml")))
    cfg.update(vars(args))
    print(cfg)

    traffic_signal_scene_id = 13
    train_dataset = load_dataset(cfg, traffic_signal_scene_id)
    model_name = OFFLINE_RL_PLANNER
    model = load_model(model_name)

    log_name = {
        "traffic_signal_scene_id": traffic_signal_scene_id,
        "imitate_loss_weight": imitate_loss_weight,
        "pred_loss_weight": pred_loss_weight,
    }
    logger = init_logger(log_name)

    train(model, train_dataset, cfg, logger)
