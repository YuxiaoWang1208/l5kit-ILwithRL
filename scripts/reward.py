import os
import sys
# Home path
from pathlib import Path

project_path = str(Path(__file__).parents[1])
print("reward.py project path: ", project_path)
sys.path.append(project_path)

# from pycharm
import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset
from l5kit.data import LocalDataManager
from l5kit.dataset import EgoDatasetVectorized
from l5kit.evaluation.metrics import distance_to_reference_trajectory
from l5kit.planning import utils
from l5kit.vectorization.vectorizer_builder import build_vectorizer


# prepare data path and load cfg
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/prediction"

dm = LocalDataManager(None)
# get config


dm = LocalDataManager(None)
# get config
cfg = load_config_data(str(Path(project_path, "examples/urban_driver/config.yaml")))

# ===== INIT DATASET
# dataset_path = dm.require(cfg["train_data_loader"]["key"])
#
# train_zarr = ChunkedDataset(dataset_path).open()
# vectorizer = build_vectorizer(cfg, dm)
# train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)
# traffic_signal_scene_id = 13


AGENT_TRAJECTORY_POLYLINE = "agent_trajectory_polyline"
AGENT_TRAJECTORY_POLYLINE_2 = "history_positions"
AGENT_YAWS = "history_yaws"
AGENT_EXTENT = "history_extents"

OTHER_AGENTS_POLYLINE = "other_agents_polyline"
OTHER_AGENTS_POLYLINE_2 = "all_other_agents_history_positions"
OTHER_AGENTS_YAWS = "all_other_agents_history_yaws"
OTHER_AGENTS_EXTENTS = "all_other_agents_history_extents"
LANES_MID = "lanes_mid"
LANES_MID_AVAIL = "lanes_mid_availabilities"

SAFE_DISTANCE_BETWEEN_EGO_AGENTS = 50  # todo


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
    # print(train_dataset)
    return train_dataset


# cfg = load_config_data(str(Path(project_path, "examples/urban_driver/config.yaml")))

# train_dataset = load_dataset(cfg, traffic_signal_scene_id)

# print(train_dataset)
# frame_id = 226
# tr_sample = train_dataset[frame_id]


def get_ego_current_state(_frame):
    centroid = _frame[AGENT_TRAJECTORY_POLYLINE][:, 0, :]
    yaws = _frame[AGENT_YAWS][:, 0, :]
    extent = _frame[AGENT_EXTENT][:, 0, :]

    return centroid, yaws, extent

def get_ego_state_by_frame_idx(_batch, _frame_idx):
    centroid = _batch[AGENT_TRAJECTORY_POLYLINE][:, _frame_idx, :]
    yaws = _batch[AGENT_YAWS][:, _frame_idx, :]
    extent = _batch[AGENT_EXTENT][:, _frame_idx, :]

    return centroid, yaws, extent

def get_ego_history_state(_frame):
    history_centroid = _frame[AGENT_TRAJECTORY_POLYLINE][:, :, :2]
    # yaws = _frame[AGENT_YAWS][:, 0, :]
    # extent = _frame[AGENT_EXTENT][:, 0, :]

    return history_centroid


def get_agent_current_state(_frame, agent_ix):
    centroid = _frame[OTHER_AGENTS_POLYLINE][agent_ix[0], agent_ix[1]][0]
    yaws = _frame[OTHER_AGENTS_YAWS][agent_ix[0], agent_ix[1]][0]
    extent = _frame[OTHER_AGENTS_EXTENTS][agent_ix[0], agent_ix[1]][0]

    return centroid, yaws, extent

def get_agent_state_by_frame_idx(_batch, agent_ix, _frame_idx):
    centroid = _batch[OTHER_AGENTS_POLYLINE][agent_ix[0], agent_ix[1]][_frame_idx]
    yaws = _batch[OTHER_AGENTS_YAWS][agent_ix[0], agent_ix[1]][_frame_idx]
    extent = _batch[OTHER_AGENTS_EXTENTS][agent_ix[0], agent_ix[1]][_frame_idx]

    return centroid, yaws, extent

#  Reward 1: distance to lane_mid
# lane_mid reward
def get_distance_to_centroid(
        current_centroid,
        ref_lanes_mid,
        consider_avail=False,
):
    current_centroid = torch.tensor(current_centroid)
    ref_lanes_mid = torch.tensor(ref_lanes_mid)
    distance = distance_to_reference_trajectory(current_centroid, ref_lanes_mid)
    return distance


def get_distance_to_centroid_by_element(ego_current_history_centroid, lanes_mid):
    # ego ~ batch, num_steps, 2
    # lane_mid ~ batch, num_elements, num_points_per_elements, 2
    assert ego_current_history_centroid.shape[0] == lanes_mid.shape[0]  # batch size
    assert ego_current_history_centroid.shape[-1] == lanes_mid.shape[-1]  # point feature

    batch_size, num_lane_elements = lanes_mid.shape[0], lanes_mid.shape[1]
    # print(history_centroid.device)

    # 筛选出距离历史四个位置距离之和最短的车道中线  即为行进路线的车道中线
    distance_to_mid_line = []
    sum_distance_to_mid_line = torch.zeros(
        (num_lane_elements, batch_size),
        device=ego_current_history_centroid.device
    )
    for lane_idx in range(num_lane_elements):
        distance_to_centroid = torch.zeros(
            (ego_current_history_centroid.shape[0], ego_current_history_centroid.shape[1]),
            device=ego_current_history_centroid.device)  # batch 4
        mid_line = lanes_mid[:, lane_idx, :, :]
        for step_idx in range(ego_current_history_centroid.shape[1]):
            distance_to_centroid[:, step_idx] = get_distance_to_centroid(ego_current_history_centroid[:, step_idx, :],
                                                                         mid_line)
        distance_to_mid_line.append(distance_to_centroid)
        sum_distance_to_mid_line[lane_idx, :] = torch.sum(distance_to_centroid, 1)

    # sum_distance_to_mid_line=np.array(sum_distance_to_mid_line)
    distance = []
    for batch_idx in range(lanes_mid.shape[0]):
        line_index = sum_distance_to_mid_line[:, batch_idx].argmin()
        distance.append(distance_to_mid_line[line_index][batch_idx, 0])

    return torch.tensor(distance, device=ego_current_history_centroid.device)


# def get_distance_to_centroid_multistep(agents_polys, lanes_mid):




def get_distance_to_centroid_per_batch(_frame):
    # todo error here. the returned reward is for last step!
    # ego_centroid, _, _ = get_ego_current_state(_frame)
    # ego_centroid = ego_centroid[:, :2]

    # 需要先筛选出ego行进路线的车道中线，而不是将所有车道中线都考虑
    # print("CENTROID")
    lanes_mid = _frame[LANES_MID]
    lanes_mid = lanes_mid[:, :, :, :2]  # only keep x, y batch 30 20 2
    history_centroid = get_ego_history_state(_frame)  # batch num_steps 2

    distance = get_distance_to_centroid_by_element(history_centroid, lanes_mid)

    # lanes_mid = lanes_mid.reshape(-1,600, 2)
    # distance = get_distance_to_centroid(ego_centroid, lanes_mid)
    return distance


# get_distance_to_centroid_per_frame(tr_sample)


# proximity reward 与其他agent最短距离


def get_distance_to_other_agents(
        ego_centroid,
        ego_yaw,
        ego_extent,
        agent_centroid,
        agent_yaw,
        agent_extent,
):
    ego_bbox = utils._get_bounding_box(ego_centroid.cpu().numpy(), ego_yaw.cpu().numpy(), ego_extent.cpu().numpy())
    agent_bbox = utils._get_bounding_box(agent_centroid.cpu().numpy(), agent_yaw.cpu().numpy(),
                                         agent_extent.cpu().numpy())
    distance = ego_bbox.distance(agent_bbox)
    return distance


# def get_distance_to_other_agents_per_frame(ego_frame, agent_frame):

def get_distance_to_other_agents_per_batch(_batch, _frame_idx):
    batch_size = _batch[OTHER_AGENTS_EXTENTS].shape[0]
    distance_list = [[] for i in range(batch_size)]  # batch
    ego_centroid, ego_yaws, ego_extent = get_ego_state_by_frame_idx(_batch, _frame_idx)
    # agent_ix = 25

    agents_type = _batch["all_other_agents_types"]
    agents_extent = torch.mean(_batch[OTHER_AGENTS_EXTENTS][:, :, _frame_idx, :], axis=2)
    # print("agent_extent")
    # print(agents_extent)

    # agent_type = 0 or agent_extent = [0, 0] is null and unavailable
    a = agents_type * agents_extent
    agent_ix_avail = torch.nonzero(a)
    # print(agent_ix_avail)

    for agent_ix in agent_ix_avail:
        # agent_info = get_agent_state_by_frame_idx(_batch, agent_ix)
        agent_info = get_agent_state_by_frame_idx(_batch, agent_ix, _frame_idx)
        dist = get_distance_to_other_agents(ego_centroid[agent_ix[0]], ego_yaws[agent_ix[0]], ego_extent[agent_ix[0]],
                                            *agent_info)
        distance_list[agent_ix[0]].append(dist)

    distance = torch.zeros(len(distance_list), device=_batch[OTHER_AGENTS_EXTENTS].device)

    for idx_in_batch in range(len(distance)):
        if distance_list[idx_in_batch]:
            distance[idx_in_batch] = np.min(distance_list[idx_in_batch])
        else:
            # there is no other agent for the frame
            distance[idx_in_batch] = SAFE_DISTANCE_BETWEEN_EGO_AGENTS

    return distance

# dist, dist_list = get_distance_to_other_agents_per_frame(tr_sample)
# dist, min_dist_index, dist_list = get_distance_to_other_agents_per_frame(train_dataset[226])
# print(dist)
# print(min_dist_index)


# print(train_dataset[226]["all_other_agents_types"])
#
# from shapely.geometry import Polygon
#
# poly_1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
# poly_2 = Polygon([[0.5, 0.5], [1, 3], [0.5, 4], [0, 3], [0.5, 0.5]])
#
# print(poly_1.distance(poly_2))
