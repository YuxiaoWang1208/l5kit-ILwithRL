import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
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

from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator

from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from multiprocessing import Process

project_path = str(Path(__file__).parents[1])
print("project path: ", project_path)
sys.path.append(project_path)
print(sys.path)

from vectorized_offline_rl_model import VectorOfflineRLModel, EnsembleOfflineRLModel
from pathlib import Path

os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/prediction"

URBAN_DRIVER = "Urban Driver"
OPEN_LOOP_PLANNER = "Open Loop Planner"
OFFLINE_RL_PLANNER = "Offline RL Planner"


def load_dataset(cfg, traffic_signal_scene_id=None):
    dm = LocalDataManager(None)
    # ===== INIT DATASET
    # cfg["train_data_loader"]["key"] = "train.zarr"
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()

    vectorizer = build_vectorizer(cfg, dm)
    train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)

    # todo demo for single scene
    if traffic_signal_scene_id:
        train_dataset = train_dataset.get_scene_dataset(traffic_signal_scene_id)
    print(train_dataset)
    return train_dataset


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


def init_logger(model_name, log_name,date):
    # tensorboard for log
    log_id = (
        f"train_flag_{log_name['train_flag']}"
        f"signal_scene_{log_name['traffic_signal_scene_id']}"
        f"-il_weight_{log_name['imitate_loss_weight']}"
        f"-pred_weight_{log_name['pred_loss_weight']}"
        f"-1"
    )
    model_log_id = f"{model_name}-{log_id}"

    log_dir = Path(project_path, "logs"+str(date))
    writer = SummaryWriter(log_dir=f"{log_dir}/{model_log_id}")
    return writer, model_log_id


def evaluation(model, eval_dataset, cfg, eval_zarr, eval_type):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.eval()
    torch.set_grad_enabled(False)

    # todo to variable

    num_scenes_to_unroll = 1
    num_simulation_steps = 249


    if eval_type == "closed_loop":
        # ==== DEFINE CLOSED-LOOP SIMULATION
        sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                                   distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                                   start_frame_index=0, show_info=True)

        sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model, model_agents=None)

        # ==== UNROLL
        scenes_to_unroll = list(range(0, len(eval_zarr.scenes), len(eval_zarr.scenes) // num_scenes_to_unroll))
        sim_outs = sim_loop.unroll(scenes_to_unroll)

        metrics = [DisplacementErrorL2Metric(),
                   DistanceToRefTrajectoryMetric(),
                   CollisionFrontMetric(),
                   CollisionRearMetric(),
                   CollisionSideMetric()]

        validators = [RangeValidator("displacement_error_l2", DisplacementErrorL2Metric, max_value=30),
                      RangeValidator("distance_ref_trajectory", DistanceToRefTrajectoryMetric, max_value=4),
                      RangeValidator("collision_front", CollisionFrontMetric, max_value=0),
                      RangeValidator("collision_rear", CollisionRearMetric, max_value=0),
                      RangeValidator("collision_side", CollisionSideMetric, max_value=0)]

        intervention_validators = ["displacement_error_l2",
                                   "distance_ref_trajectory",
                                   "collision_front",
                                   "collision_rear",
                                   "collision_side"]

        cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
                                                           validators=validators,
                                                           composite_metrics=[],
                                                           intervention_validators=intervention_validators))
        cle_evaluator.evaluate(sim_outs)
        validation_results = cle_evaluator.validation_results()
        agg = ValidationCountingAggregator().aggregate(validation_results)
        cle_evaluator.reset()

        print(agg)

    return agg


def train(model, train_dataset, eval_dataset, cfg, writer, date, model_name):
    # todo
    # cfg["train_params"]["max_num_steps"] = int(1e8)

    train_cfg = cfg["train_data_loader"]
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                                  num_workers=train_cfg["num_workers"])
    # eval_dataset = train_dataset

    eval_cfg = cfg["val_data_loader"]
    dm = LocalDataManager(None)
    eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()

    losses_train = []
    # training loops
    tr_it = iter(train_dataloader)

    # prepare for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    torch.set_grad_enabled(True)

    progress_bar = tqdm(range(int(cfg["train_params"]["max_num_steps"])))
    for n_iter in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
        # Forward pass
        data = {k: v.to(device) for k, v in data.items()}

        result_list = model(data)
        optimizer.zero_grad()

        result_list=[result_list]

        for idx, result in enumerate(result_list):
            loss = result["loss"]
            writer.add_scalar(f'Loss/model_{idx}_train', loss.item(), n_iter)
            writer.add_scalar(f'Loss/model_{idx}_train_policy_loss', result["loss_imitate"].item(), n_iter)
            writer.add_scalar(f'Loss/model_{idx}_train_prediction_loss', result["loss_other_agent_pred"].item(), n_iter)
            writer.add_scalar(f'Loss/model_{idx}_train_reward_loss', result["loss_reward"].item(), n_iter)
            writer.add_scalar(f'Loss/model_{idx}_train_value_loss', result["loss_value"].item(), n_iter)

            loss.backward()

        # Backward pass
        optimizer.step()

        # losses_train.append(loss.item())
        # progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

        if n_iter % cfg["train_params"]["checkpoint_every_n_steps"] == 0:

            # save model
            # to_save = torch.jit.script(model.cpu())
            dir_to_save = Path(project_path, "tmp"+str(date) , model_name)
            dir_to_save.mkdir(parents=True, exist_ok=True)
            path_to_save = Path(dir_to_save, f"iter_{n_iter:07}.pt")
            # to_save.save(path_to_save)
            # model.save(path_to_save)
            torch.save(model.state_dict(), path_to_save)
            print(f"MODEL STORED at {path_to_save}")

        if n_iter % cfg["train_params"]["eval_every_n_steps"] == 0:
            # evaluation(model, eval_type="close_loop")
            model.eval()
            eval_results = evaluation(model, eval_dataset, cfg, eval_zarr, eval_type="closed_loop")

            writer.add_scalar(f'Eval/displacement_error_l2', eval_results["displacement_error_l2"].item(), n_iter)
            writer.add_scalar(f'Eval/distance_ref_trajectory', eval_results["distance_ref_trajectory"].item(), n_iter)
            writer.add_scalar(f'Eval/collision_front', eval_results["collision_front"].item(), n_iter)
            writer.add_scalar(f'Eval/collision_rear', eval_results["collision_rear"].item(), n_iter)
            writer.add_scalar(f'Eval/collision_side', eval_results["collision_side"].item(), n_iter)

            model.train()
            torch.set_grad_enabled(True)




def load_config_data(path: str) -> dict:
    """Load a config data from a given path

    :param path: the path as a string
    :return: the config as a dict
    """
    with open(path) as f:
        cfg: dict = yaml.safe_load(f)
    return cfg


def evaluate_with_baseline():
    # ===== INIT DATASET  FOR EVALUATE
    dm = LocalDataManager(None)
    eval_cfg = cfg["val_data_loader"]
    eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()

    # MBOP
    model_name = OFFLINE_RL_PLANNER

    # model=load_model(model_name)
    # model.load_state_dict(torch.load('/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/tmponly_policy/'
    #                                  'Offline RL Planner-train_flag_0signal_scene_13-il_weight_1.0-pred_weight_1.0-1/iter_0268000.pt'))

    num_ensemble = 4
    model_list = [load_model(model_name) for _ in range(num_ensemble)]

    model_path0 = "/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/tmp0605/" \
                  "Offline RL Planner-train_flag_0signal_scene_13-il_weight_1.0-pred_weight_1.0-1/iter_0005000.pt"
    model_path1 = "/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/tmp0604_2/" \
                  "Offline RL Planner-train_flag_1signal_scene_13-il_weight_1.0-pred_weight_1.0-1/iter_0005000.pt"
    model_path2 = "/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/tmp0604_2/" \
                  "Offline RL Planner-train_flag_2signal_scene_13-il_weight_1.0-pred_weight_1.0-1/iter_0005000.pt"
    model_path3 = "/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/tmp0604_2/" \
                  "Offline RL Planner-train_flag_3signal_scene_13-il_weight_1.0-pred_weight_1.0-1/iter_0005000.pt"
    model_list[0].load_state_dict(torch.load(model_path0))
    model_list[1].load_state_dict(torch.load(model_path1))
    model_list[2].load_state_dict(torch.load(model_path2))
    model_list[3].load_state_dict(torch.load(model_path3))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnsembleOfflineRLModel(model_list)
    model = model.to(device)


    # baseline urban_driver
    # model_path = "/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/examples/urban_driver/MS.pt"

    # baseline urban_driver_without_BPTT
    # model_path = "/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/examples/urban_driver/BPTT.pt"

    # baseline open loop
    # model_path = "/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/examples/urban_driver/OL.pt"

    # baseline open loop with history
    # model_path = "/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/examples/urban_driver/OL_HS.pt"

    # model = torch.load(model_path)


    model.eval()
    torch.set_grad_enabled(False)

    eval_results = evaluation(model, eval_dataset, cfg, eval_zarr, eval_type="closed_loop")

def train_process(train_flag,date,traffic_signal_scene_id,imitate_loss_weight,pred_loss_weight,model_name,train_dataset,eval_dataset,cfg):
    log_name = {
        "traffic_signal_scene_id": traffic_signal_scene_id if not None else "all",
        "imitate_loss_weight": imitate_loss_weight,
        "pred_loss_weight": pred_loss_weight,
        "train_flag": train_flag,
    }
    logger, model_log_id = init_logger(model_name, log_name,date)

    model = load_model(model_name)

    train(model, train_dataset, eval_dataset, cfg, logger, date, model_name=model_log_id)


if __name__ == '__main__':
    import argparse
    import os

    os.environ["_TEST_TUNE_TRIAL_UUID"] = "_"  # 在log路径不包含uuid, 这样可以是文件夹完全按照创建时间排序


    parser = argparse.ArgumentParser()
    parser.add_argument("--imitate_loss_weight", type=float, default=1.0)
    parser.add_argument("--pred_loss_weight", type=float, default=1.0)
    parser.add_argument("--cuda_id", type=int, default=3)
    parser.add_argument("--flag", type=str,default='debug')   #训练模式

    args = parser.parse_args()

    # args.cuda_id = 1

    gpu_avaliable_list = [str(args.cuda_id)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_avaliable_list)

    imitate_loss_weight = args.imitate_loss_weight
    pred_loss_weight = args.pred_loss_weight
    flag = args.flag

    cfg = load_config_data(str(Path(project_path, "scripts_/offline_rl_config.yaml")))
    cfg.update(vars(args))
    print(cfg)

    # to test all traffic signal scenarios
    # traffic_signal_scene_id = None
    traffic_signal_scene_id = 13
    train_dataset = load_dataset(cfg, traffic_signal_scene_id)
    eval_dataset = train_dataset
    model_name = OFFLINE_RL_PLANNER

    # num_ensemble = 4
    # model_list = [load_model(model_name) for _ in range(num_ensemble)]





    process = [Process(target=train_process, args=(0,flag,traffic_signal_scene_id,imitate_loss_weight,pred_loss_weight,model_name,train_dataset,eval_dataset,cfg)),
               Process(target=train_process, args=(1,flag,traffic_signal_scene_id,imitate_loss_weight,pred_loss_weight,model_name,train_dataset,eval_dataset,cfg)),
               Process(target=train_process, args=(2,flag,traffic_signal_scene_id,imitate_loss_weight,pred_loss_weight,model_name,train_dataset,eval_dataset,cfg)),
               Process(target=train_process, args=(3,flag,traffic_signal_scene_id,imitate_loss_weight,pred_loss_weight,model_name,train_dataset,eval_dataset,cfg)), ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束



    #评估网络模型
    # evaluate_with_baseline()




