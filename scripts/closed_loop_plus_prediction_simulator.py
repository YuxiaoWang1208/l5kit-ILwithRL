from l5kit.simulation.unroll import ClosedLoopSimulator, ClosedLoopSimulatorModes, SimulationOutput, UnrollInputOutput
from collections import defaultdict
from enum import IntEnum
from typing import DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

from l5kit.data import AGENT_DTYPE, PERCEPTION_LABEL_TO_INDEX
from l5kit.dataset import EgoDataset
from l5kit.dataset.utils import move_to_device, move_to_numpy
from l5kit.geometry import rotation33_as_yaw, transform_points
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset


class ClosedLoopPlusPredictionSimulator(ClosedLoopSimulator):
    def __init__(self,
                 sim_cfg: SimulationConfig,
                 dataset: EgoDataset,
                 device: torch.device,
                 model_ego: Optional[torch.nn.Module] = None,
                 model_agents: Optional[torch.nn.Module] = None,
                 keys_to_exclude: Tuple[str] = ("image",),
                 mode: int = ClosedLoopSimulatorModes.L5KIT):

        """
        Create a simulation loop object capable of unrolling ego and agents
        :param sim_cfg: configuration for unroll
        :param dataset: EgoDataset used while unrolling
        :param device: a torch device. Inference will be performed here
        :param model_ego: the model to be used for ego
        :param model_agents: the model to be used for agents
        :param keys_to_exclude: keys to exclude from input/output (e.g. huge blobs)
        :param mode: the framework that uses the closed loop simulator
        """
        super().__init__(
            sim_cfg,
            dataset,
            device,
            model_ego,
            model_agents,
            keys_to_exclude,
            mode,
        )

    def unroll(self, scene_indices: List[int]) -> List[SimulationOutput]:
        """
        Simulate the dataset for the given scene indices
        :param scene_indices: the scene indices we want to simulate
        :return: the simulated dataset
        """
        sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_indices, self.sim_cfg)

        agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        for frame_index in tqdm(range(len(sim_dataset)), disable=not self.sim_cfg.show_info):
            next_frame_index = frame_index + 1
            should_update = next_frame_index != len(sim_dataset)

            # EGO
            if not self.sim_cfg.use_ego_gt:
                ego_input = sim_dataset.rasterise_frame_batch(frame_index)
                ego_input_dict = default_collate(ego_input)
                ego_output_dict = self.model_ego(move_to_device(ego_input_dict, self.device))

                ego_input_dict = move_to_numpy(ego_input_dict)
                ego_output_dict = move_to_numpy(ego_output_dict)

                if should_update:
                    self.update_ego(sim_dataset, next_frame_index, ego_input_dict, ego_output_dict)

                ego_frame_in_out = self.get_ego_in_out(ego_input_dict, ego_output_dict, self.keys_to_exclude)
                for scene_idx in scene_indices:
                    ego_ins_outs[scene_idx].append(ego_frame_in_out[scene_idx])


                agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)
                if len(agents_input):  # agents may not be available
                    agents_input_dict = default_collate(list(agents_input.values()))
                    # agents_output_dict = self.model_agents(move_to_device(agents_input_dict, self.device))
                    # agents_output_dict = self.model_ego(move_to_device(agents_input_dict, self.device))
                    agents_output_dict = {k: v for k, v in ego_output_dict.items() if "agents" in k}

                    # for update we need everything as numpy
                    agents_input_dict = move_to_numpy(agents_input_dict)
                    agents_output_dict = move_to_numpy(agents_output_dict)

                    if should_update:
                        self.update_pred_agents(sim_dataset, next_frame_index, agents_input_dict, agents_output_dict)

                    # update input and output buffers
                    agents_frame_in_out = self.get_agents_in_out(agents_input_dict, agents_output_dict,
                                                                 self.keys_to_exclude)
                    for scene_idx in scene_indices:
                       agents_ins_outs[scene_idx].append(agents_frame_in_out.get(scene_idx, []))


        simulated_outputs: List[SimulationOutput] = []
        for scene_idx in scene_indices:
            simulated_outputs.append(SimulationOutput(scene_idx, sim_dataset, ego_ins_outs, agents_ins_outs))
        return simulated_outputs

    @staticmethod
    def update_pred_agents(dataset: SimulationDataset, frame_idx: int, input_dict: Dict[str, np.ndarray],
                           output_dict: Dict[str, np.ndarray]) -> None:
        """Update the agents in frame_idx (across scenes) using agents_output_dict

        :param dataset: the simulation dataset
        :param frame_idx: index of the frame to modify
        :param input_dict: the input to the agent model
        :param output_dict: the output of the agent model
        :return:
        """

        agents_update_dict: Dict[Tuple[int, int], np.ndarray] = {}

        world_from_agent = input_dict["world_from_agent"]
        yaw = input_dict["yaw"]
        pred_trs = transform_points(output_dict["positions"][:, :1], world_from_agent)[:, 0]
        pred_yaws = yaw + output_dict["yaws"][:, 0, 0]

        next_agents = np.zeros(len(yaw), dtype=AGENT_DTYPE)
        next_agents["centroid"] = pred_trs
        next_agents["yaw"] = pred_yaws
        next_agents["track_id"] = input_dict["track_id"]
        next_agents["extent"] = input_dict["extent"]

        next_agents["label_probabilities"][:, PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]] = 1

        for scene_idx, next_agent in zip(input_dict["scene_index"], next_agents):
            agents_update_dict[(scene_idx, next_agent["track_id"])] = np.expand_dims(next_agent, 0)
        dataset.set_agents(frame_idx, agents_update_dict)


if __name__ == '__main__':
    import os
    import numpy as np
    import torch
    from torch import nn, optim
    from tqdm import tqdm

    from l5kit.configs import load_config_data
    from l5kit.data import ChunkedDataset, LocalDataManager
    from l5kit.dataset import EgoDatasetVectorized
    from l5kit.vectorization.vectorizer_builder import build_vectorizer

    from pathlib import Path

    import sys
    from pathlib import Path

    # project_path = str(Path(__file__).parents[1])
    project_path = "/mnt/share_disk/user/daixingyuan/l5kit"
    print("project path: ", project_path)
    sys.path.append(project_path)

    from scripts_.vectorized_offline_rl_model import VectorOfflineRLModel
    # prepare data path and load cfg
    os.environ["L5KIT_DATA_FOLDER"] = "/mnt/share_disk/user/public/l5kit/prediction"

    dm = LocalDataManager(None)
    # get config
    cfg = load_config_data(str(Path(project_path, "examples/urban_driver/config.yaml")))

    # ===== INIT DATASET
    # cfg["train_data_loader"]["key"] = "train.zarr"
    dataset_path = dm.require(cfg["train_data_loader"]["key"])

    train_zarr = ChunkedDataset(dataset_path).open()
    vectorizer = build_vectorizer(cfg, dm)
    train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)

    val_dataset = train_dataset.get_scene_dataset(13)
    print(train_dataset)
    print(val_dataset)


    weights_scaling = [1.0, 1.0, 1.0]

    _num_predicted_frames = cfg["model_params"]["future_num_frames"]
    _num_predicted_params = len(weights_scaling)

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

    model_path_3 = "/mnt/share_disk/user/daixingyuan/l5kit/tmp/Offline RL Planner-signal_scene_13-il_weight_1.0-pred_weight_0.5-1/iter_0030000.pt"

    val_zarr = train_zarr.get_scene_dataset(13)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(_model_path):
        model.load_state_dict(torch.load(_model_path))
        model.eval()
        model.to(device)
        return model


    model = load_model(model_path_3)

    num_scenes_to_unroll = 1
    num_simulation_steps = 249

    # ==== DEFINE CLOSED-LOOP SIMULATION
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                               distance_th_far=500, distance_th_close=50,
                               num_simulation_steps=num_simulation_steps,
                               start_frame_index=0, show_info=True)

    sim_loop = ClosedLoopPlusPredictionSimulator(sim_cfg, val_dataset, device, model_ego=model, model_agents=None)

    # ==== UNROLL
    scenes_to_unroll = list(range(0, len(val_zarr.scenes), len(val_zarr.scenes) // num_scenes_to_unroll))
    sim_outs = sim_loop.unroll(scenes_to_unroll)

