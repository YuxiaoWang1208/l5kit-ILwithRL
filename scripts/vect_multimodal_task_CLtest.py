import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from prettytable import PrettyTable

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.rasterization.rasterizer_builder import build_rasterizer

from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from multimodal_close_loop_simulator import MultimodalClosedLoopSimulator
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator

from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene, multimodal_simulation_out_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show, output_file
from l5kit.data import MapAPI

from collections import defaultdict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
from pathlib import Path
sys.path.append('/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/scripts')
sys.path.append('/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/scripts/vect_multimodal_task_model')
sys.path.append('/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/scripts/reward')
# import vectorized_offline_rl_model
# from vectorized_offline_rl_model import VectorOfflineRLModel, EnsembleOfflineRLModel
from vect_multimodal_task_model import VectorMultiModalTaskModel
from vectorized_offline_rl_model import VectorOfflineRLModel

from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel  
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR, draw_trajectory
import cv2


project_path = "/mnt/share_disk/user/wangyuxiao/l5kit-RL-pre_train"
print("project path: ", project_path)
sys.path.append(project_path)

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/l5kit/prediction"
dm = LocalDataManager(None)
# get config
cfg = load_config_data(str(Path(project_path, "scripts/vect_multimodal_task_config.yaml")))
print(cfg)

'''
# ==== Load the model1
model1_name = "Offline Multimodal Planner"
kwargs=dict(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            num_outputs=cfg["model_params"]["num_outputs"],
            num_modes=cfg["model_params"]["num_modes"],
            weights_scaling=[1.0, 1.0, 1.0],
            criterion=nn.MSELoss(reduction="none"),
            
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
            cfg=cfg
)
model1 = VectorMultiModalTaskModel(**kwargs)
model1_path="/mnt/share_disk/user/wangyuxiao/l5kit-RL-pre_train/tmp1/vect_multimodal_task_learn_model/9000.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1.load_state_dict(torch.load(model1_path))
# model_list[1].load_state_dict(torch.load(model_path1))
# model_list[2].load_state_dict(torch.load(model_path2))
# model_list[3].load_state_dict(torch.load(model_path3))
model1 = model1.to(device)
model1 = model1.eval()
torch.set_grad_enabled(False)


# ==== LOAD THE MODEL2
model2_name = "Offline RL Planner"
weights_scaling = [1.0, 1.0, 1.0]
_num_predicted_frames = cfg["model_params"]["future_num_frames"]
_num_predicted_params = len(weights_scaling)
kwargs=dict(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=_num_predicted_params * _num_predicted_frames,
            weights_scaling=weights_scaling,
            criterion=torch.nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
            cfg=cfg
)
model2 = VectorOfflineRLModel(**kwargs)
model2_path="/mnt/share_disk/user/xijinhao/l5kit-model-based-offline-rl/tmpurban_2_5000/Offline RL Planner-train_flag_0signal_scene_0-il_weight_1.0-pred_weight_1.0-pretrained_True-1/iter_0016500.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model2.load_state_dict(torch.load(model2_path))

# model_list[2].load_state_dict(torch.load(model_path2))
# model_list[3].load_state_dict(torch.load(model_path3))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model2 = model2.to(device)
model2 = model2.eval()
torch.set_grad_enabled(False)
'''

# === LOAD THE MODEL3
model3_path = f"{Path(__file__).parents[1]}/pre_trained_models/BPTT.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model3 = torch.load(model3_path).to(device)
model3 = model3.eval()
torch.set_grad_enabled(False)


# === LOAD THE MODEL4
kwargs=dict(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            num_outputs=cfg["model_params"]["num_outputs"],
            num_modes=cfg["model_params"]["num_modes"],
            weights_scaling=[1.0, 1.0, 1.0],
            criterion=nn.MSELoss(reduction="none"),
            
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
            cfg=cfg
)
model4 = VectorMultiModalTaskModel(**kwargs)  # 29500
# model4_path = f"{Path(__file__).parents[1]}/tmp1116/Offline Multimodal Planner-train_flag_True-signal_scene_None-il_weight_1.0-pred_weight_1.0-pretrained_True-1/iter_500.pt"
model4_path = f"{Path(__file__).parents[1]}/tmp1117/Offline Multimodal Planner-train_flag_True-signal_scene_None-il_weight_1.0-pred_weight_1.0-pretrained_True-1/iter_50000.pt"

# print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model4.load_state_dict(torch.load(model4_path))
model4 = model4.to(device)
model4 = model4.eval()
torch.set_grad_enabled(False)


# ===== INIT DATASET
eval_cfg = cfg["val_data_loader"]
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
vectorizer = build_vectorizer(cfg, dm)
eval_dataset = EgoDatasetVectorized(cfg, eval_zarr, vectorizer)
print(eval_dataset)


# close loop evaluation:
num_scenes_to_unroll = 10  # 10 1
num_simulation_steps = 248

# ==== DEFINE CLOSED-LOOP SIMULATION
sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                        distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                        start_frame_index=0, show_info=True)

sim_loop = MultimodalClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model4, model_agents=None)
# sim_loop =ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model3, model_agents=None)


# ==== UNROLL
scenes_to_unroll = list(range(0, len(eval_zarr.scenes), len(eval_zarr.scenes)//num_scenes_to_unroll))
# scenes_to_unroll=list(range(15))
sim_outs = sim_loop.unroll(scenes_to_unroll)


# ==== CLOSE LOOP METRICS
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


# ==== QUANTITATIVE EVALUATION
cle_evaluator.evaluate(sim_outs)
validation_results = cle_evaluator.validation_results()
agg = ValidationCountingAggregator().aggregate(validation_results)
cle_evaluator.reset()

fields = ["metric", "value"]
table = PrettyTable(field_names=fields)

values = []
names = []

for metric_name in agg:
    table.add_row([metric_name, agg[metric_name].item()])
    values.append(agg[metric_name].item())
    names.append(metric_name)

print(table)

plt.bar(np.arange(len(names)), values)
plt.xticks(np.arange(len(names)), names, rotation=60, ha='right')
# plt.show()
project_path = str(Path(__file__).parents[1])
plotPath = Path(project_path, "plot")
plt.savefig(plotPath)

# ==== VISUALIZATION
output_notebook()
mapAPI = MapAPI.from_cfg(dm, cfg)
count = 0
for sim_out in sim_outs: # for each scene
    vis_in = multimodal_simulation_out_to_visualizer_scene(sim_out, mapAPI)
    # vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
    output_file(Path(project_path, "plot" + str(count) + ".html"))
    count += 1
    show(visualize(sim_out.scene_id, vis_in))

