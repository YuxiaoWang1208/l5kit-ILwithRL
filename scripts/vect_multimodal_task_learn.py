import os
import sys
from pathlib import Path

project_path = str(Path(__file__).parents[1])
print("project path: ", project_path)
sys.path.append(project_path)
sys.path.append(project_path + "/l5kit")

from typing import Dict
import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.kinematic import AckermanPerturbation
from l5kit.random.random_generator import GaussianRandomGenerator
from l5kit.planning.vectorized.multimodal_model import VectorizedMultiModalPlanningModel
# from l5kit.planning.vectorized.open_loop_model import VectorizedModel

from vect_multimodal_task_model import VectorMultiModalTaskModel

from torch.utils.tensorboard import SummaryWriter



# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/root/zhufenghua12/l5kit/prediction"


def load_dataset(cfg, traffic_signal_scene_id_list=None, train=True):
    # ==== DATASET PERTURBATION
    mean_value=np.array([0.0,0.0,0.0])
    # std_value=np.array([0.5,1.5,np.pi/6])
    # AckermanPerturbation1=AckermanPerturbation(random_offset_generator=GaussianRandomGenerator(mean=mean_value,std=std_value),perturb_prob=0.5)
    std_value2 = np.array([1, 1, np.pi / 3])
    AckermanPerturbation2 = AckermanPerturbation(
        random_offset_generator=GaussianRandomGenerator(mean=mean_value, std=std_value2), perturb_prob=0.3)

    # ===== INIT DATASET
    dm = LocalDataManager(None)
    vectorizer = build_vectorizer(cfg, dm)
    if train:
        train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
        train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer, perturbation=AckermanPerturbation2)
    else:
        eval_zarr = ChunkedDataset(dm.require(cfg["val_data_loader"]["key"])).open()
        eval_dataset = EgoDatasetVectorized(cfg, eval_zarr, vectorizer)

    # todo demo for single scene
    data_list = []
    scene_id_list=[]
    num_of_scenes = 0
    if traffic_signal_scene_id_list == None:
        traffic_signal_scene_id_list = []
    if len(traffic_signal_scene_id_list) > 1:
        # if train==True:
        #     for scene_id in traffic_signal_scene_id_list:
        #         scene_1 = train_dataset.get_scene_dataset(scene_id)
        #         if len(scene_1.dataset.tl_faces) > 0:
        #             data_list.append(scene_1)
        #             num_of_scenes += 1  # 累计有多少个场景
        #     train_dataset = ConcatDataset(data_list)
        #     print('num_of_scenes:', num_of_scenes)
        #     return train_dataset
        # if train==False:
        for scene_id in traffic_signal_scene_id_list:
            if train:
                scene_1 = train_dataset.get_scene_dataset(scene_id)
            else:
                scene_1 = eval_dataset.get_scene_dataset(scene_id)
            if len(scene_1.dataset.tl_faces) > 0:
                data_list.append(scene_1)
                scene_id_list.append(scene_id)
                num_of_scenes += 1  # 累计有多少个场景
        print('num_of_scenes:', num_of_scenes)
        print('scene_id_list:',scene_id_list)
        print(data_list)
        return data_list
    if len(traffic_signal_scene_id_list) == 1:
        if train:
            train_dataset = train_dataset.get_scene_dataset(traffic_signal_scene_id_list[0])
            print(train_dataset)
            return train_dataset
        else:
            eval_dataset = eval_dataset.get_scene_dataset(traffic_signal_scene_id_list[0])
            print(eval_dataset)
            return eval_dataset
    if train:
        print(train_dataset)
        return train_dataset
    else:
        print(eval_dataset)
        return eval_dataset


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

    model = VectorMultiModalTaskModel(
            history_num_frames_ego=history_num_frames_ego,
            history_num_frames_agents=history_num_frames_agents,
            future_num_frames=future_num_frames,num_outputs=num_outputs,num_modes=num_modes,
            weights_scaling= [1., 1., 1.],criterion=nn.MSELoss(reduction="none"),
            global_head_dropout=global_head_dropout,disable_other_agents=disable_other_agents,
            disable_map=disable_map,disable_lane_boundaries=disable_lane_boundaries,
            cfg=cfg,
            )

    return model


def init_logger(model_name, log_name, date):
    # tensorboard for log
    log_id = (
        f"train_flag_{log_name['train_flag']}"
        f"-signal_scene_{log_name['traffic_signal_scene_id']}"
        f"-il_weight_{log_name['imitate_loss_weight']}"
        f"-pred_weight_{log_name['pred_loss_weight']}"
        f"-pretrained_{log_name['is_pretrained']}"
        f"-1"
    )
    model_log_id = f"{model_name}-{log_id}"

    log_dir = Path(project_path, "logs" + str(date))
    writer = SummaryWriter(log_dir=f"{log_dir}/{model_log_id}")
    return writer, model_log_id





def train(model, train_dataset, eval_dataset, cfg, writer, date, model_name):
    # todo
    # cfg["train_params"]["max_num_steps"] = int(1e8)
    eval_cfg = cfg["val_data_loader"]
    dm = LocalDataManager(None)
    eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()

    train_cfg = cfg["train_data_loader"]
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                                  num_workers=train_cfg["num_workers"])

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

        result_list = [result_list]

        for idx, result in enumerate(result_list):
            loss = result["loss"]
            writer.add_scalar(f'Loss/model_{idx}_train_loss', loss.item(), n_iter)
            writer.add_scalar(f'Loss/model_{idx}_train_policy_loss', result["loss_ego_imitate"].item(), n_iter)
            writer.add_scalar(f'Loss/model_{idx}_train_prediction_loss', result["loss_other_agents_pred"].item(), n_iter)
            writer.add_scalar(f'Loss/model_{idx}_train_reward_loss', result["loss_reward"].item(), n_iter)
            # writer.add_scalar(f'Loss/model_{idx}_train_value_loss', result["loss_value"].item(), n_iter)
            writer.add_scalar(f'Loss/model_{idx}_train_speed_loss', result["loss_speed"].item(), n_iter)
            writer.add_scalar(f'Loss/model_{idx}_train_tl_loss', result["loss_tl"].item(), n_iter)

            loss.backward()

        # Backward pass
        optimizer.step()

        # losses_train.append(loss.item())
        # progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

        if n_iter % cfg["train_params"]["checkpoint_every_n_steps"] == 0:
            # save model
            # to_save = torch.jit.script(model.cpu())
            dir_to_save = Path(project_path, "tmp" + str(date), model_name)
            dir_to_save.mkdir(parents=True, exist_ok=True)
            path_to_save = Path(dir_to_save, f"iter_{n_iter}.pt")
            # to_save.save(path_to_save)
            # model.save(path_to_save)
            torch.save(model.state_dict(), path_to_save)
            print(f"MODEL STORED at {path_to_save}")

        # if n_iter % cfg["train_params"]["eval_every_n_steps"] == 0:
        #     model.eval()
        #     # #开环评估
        #     open_loop_results = evaluation(model, eval_dataset, cfg, eval_zarr, eval_type="open_loop")
        #     writer.add_scalar(f'Eval/ade', open_loop_results["ade"].item(), n_iter)
        #     writer.add_scalar(f'Eval/fde', open_loop_results["fde"].item(), n_iter)
        #     writer.add_scalar(f'Eval/angle_dis', open_loop_results["angle_dis"].item(), n_iter)

        #     #闭环评估   evaluation(model, eval_type="close_loop")
        #     eval_results = evaluation(model, eval_dataset, cfg, eval_zarr, eval_type="closed_loop")

        #     writer.add_scalar(f'Eval/displacement_error_l2', eval_results["displacement_error_l2"].item(), n_iter)
        #     writer.add_scalar(f'Eval/distance_ref_trajectory', eval_results["distance_ref_trajectory"].item(), n_iter)
        #     writer.add_scalar(f'Eval/collision_front', eval_results["collision_front"].item(), n_iter)
        #     writer.add_scalar(f'Eval/collision_rear', eval_results["collision_rear"].item(), n_iter)
        #     writer.add_scalar(f'Eval/collision_side', eval_results["collision_side"].item(), n_iter)

        #     model.train()
        #     torch.set_grad_enabled(True)


if __name__ == '__main__':
    os.environ["_TEST_TUNE_TRIAL_UUID"] = "_"  # 在log路径不包含uuid, 这样可以是文件夹完全按照创建时间排序

    gpu_avaliable_list = [str(1)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_avaliable_list)

    # get config
    cfg = load_config_data("/root/zhufenghua12/wangyuxiao/l5kit-wyx/scripts/vect_multimodal_task_config.yaml")

    # train_traffic_signal_scene_id_list=list(np.arange(13, 14))
    # eval_traffic_signal_scene_id_list = list(np.arange(13, 14))

    # load dataset
    train_dataset= load_dataset(cfg, train=True)
    eval_dataset = load_dataset(cfg, train=False)

    # prepare model
    model_name = "Offline Multimodal Planner"
    model = build_model(cfg)

    # use pretrained model
    pretrained_model_dir = "/mnt/share_disk/user/wangyuxiao/l5kit-RL-pre_train/pre_trained_models"
    pretrained_model_name = "BPTT"
    pretrained_model_path = os.path.join(pretrained_model_dir, pretrained_model_name + ".pt")
    pretrained_model = torch.load(pretrained_model_path)

    pretrained_model_state_dict = pretrained_model.state_dict()
    new_pretrained_model_state_dict = copy.deepcopy(pretrained_model_state_dict)
    # print(pretrained_model.state_dict())
    # 去除预训练模型与多模态模型不匹配的几处：
    unfit_paras = ["xy_scale", "global_head.output_embed.layers.2.weight", "global_head.output_embed.layers.2.bias"]
    for unfit_para in unfit_paras:
        pretrained_model_state_dict.pop(unfit_para)
        new_pretrained_model_state_dict.pop(unfit_para)
    # 将预训练模型的ego预测网络部分复用到本模型other agents预测网络中
    for k,v in pretrained_model_state_dict.items():
        if "global_head" in k:
            new_key = k.replace("global_head", "global_prediction_head")
            new_dict = {new_key: v}
            new_pretrained_model_state_dict.update(new_dict)
    model.load_state_dict(new_pretrained_model_state_dict, strict=False)  
    # fix parameters of the model
    # unfixed last two layer of policy net
    unfixed_paras = ["global_head.output_embed.layers.1.weight", 'global_head.output_embed.layers.2.weight',
                     "global_head.output_embed.layers.1.bias", "global_head.output_embed.layers.2.bias",
                     "global_prediction_head.output_embed.layers.1.weight", 'global_prediction_head.output_embed.layers.2.weight',
                     "global_prediction_head.output_embed.layers.1.bias", "global_prediction_head.output_embed.layers.2.bias",]
    for name, param in model.named_parameters():
        if name in new_pretrained_model_state_dict and name not in unfixed_paras:
            param.requires_grad = False

    # logger
    date = 1117
    traffic_signal_scene_id = None
    imitate_loss_weight = cfg["loss_ego_imitate_weight"]
    pred_loss_weight = cfg["loss_other_agents_pred_weight"]
    train_flag = True
    is_pretrained = True  # True False
    log_name = {
        "traffic_signal_scene_id": traffic_signal_scene_id if not None else "all",
        "imitate_loss_weight": imitate_loss_weight,
        "pred_loss_weight": pred_loss_weight,
        "train_flag": train_flag,
        "is_pretrained": is_pretrained,
    }
    logger, model_log_id = init_logger(model_name, log_name, date)

    train(model, train_dataset, eval_dataset, cfg, logger, date, model_name=model_log_id)




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

