from typing import Dict
from typing import List

import torch
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.planning.vectorized.common import build_target_normalization, pad_avail, pad_points
# from .global_graph import MultiheadAttentionGlobalHead, VectorizedEmbedding
# from .local_graph import LocalSubGraph, SinusoidalPositionalEmbedding

from torch import nn


class VectorizedUnrollModel(VectorizedModel):
    """ Vectorized closed-loop planning model.
    """

    def __init__(
            self,
            history_num_frames_ego: int,
            history_num_frames_agents: int,
            num_targets: int,
            weights_scaling: List[float],
            criterion: nn.Module,  # criterion is only needed for training and not for evaluation
            global_head_dropout: float,
            disable_other_agents: bool,
            disable_map: bool,
            disable_lane_boundaries: bool,
    ) -> None:
        """ Initializes the model.

        :history_num_frames_ego: number of history ego frames to include
        :param history_num_frames_agents: number of history agent frames to include
        :param num_targets: number of values to predict
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param gobal_head_dropout: float in range [0,1] for the dropout in the MHA global head. Set to 0 to disable it
        :param disable_other_agents: ignore agents
        :param disable_map: ignore map
        :param disable_lane_boundaries: ignore lane boundaries
        """

        num_targets = 3  # this will limit queries number to 1

        super().__init__(
            history_num_frames_ego,
            history_num_frames_agents,
            num_targets,
            weights_scaling,
            criterion,
            global_head_dropout,
            disable_other_agents,
            disable_map,
            disable_lane_boundaries,
        )


    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Load and prepare vectors for the model call, split into map and agents

        # ==== LANES ====
        # batch size x num lanes x num vectors x num features
        polyline_keys = ["lanes_mid", "crosswalks"]
        if not self.disable_lane_boundaries:
            polyline_keys += ["lanes"]
        avail_keys = [f"{k}_availabilities" for k in polyline_keys]

        max_num_vectors = max([data_batch[key].shape[-2] for key in polyline_keys])

        map_polys = torch.cat([pad_points(data_batch[key], max_num_vectors) for key in polyline_keys], dim=1)
        map_polys[..., -1].fill_(0)
        # batch size x num lanes x num vectors
        map_availabilities = torch.cat([pad_avail(data_batch[key], max_num_vectors) for key in avail_keys], dim=1)

        # ==== AGENTS ====
        # batch_size x (1 + M) x seq len x self._vector_length
        agents_polys = torch.cat(
            [data_batch["agent_trajectory_polyline"].unsqueeze(1), data_batch["other_agents_polyline"]], dim=1
        )
        # batch_size x (1 + M) x num vectors x self._vector_length
        agents_polys = pad_points(agents_polys, max_num_vectors)

        # batch_size x (1 + M) x seq len
        agents_availabilities = torch.cat(
            [
                data_batch["agent_polyline_availability"].unsqueeze(1),
                data_batch["other_agents_polyline_availability"],
            ],
            dim=1,
        )
        # batch_size x (1 + M) x num vectors
        agents_availabilities = pad_avail(agents_availabilities, max_num_vectors)

        # batch_size x (1 + M) x num features
        type_embedding = self.type_embedding(data_batch).transpose(0, 1)
        lane_bdry_len = data_batch["lanes"].shape[1]

        # call the model with these features
        outputs, attns = self.model_call(
            agents_polys, map_polys, agents_availabilities, map_availabilities, type_embedding, lane_bdry_len
        )

        # calculate loss or return predicted position for inference
        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            xy = data_batch["target_positions"]
            yaw = data_batch["target_yaws"]
            if self.normalize_targets:
                xy /= self.xy_scale
            targets = torch.cat((xy, yaw), dim=-1)
            target_weights = data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling
            loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            train_dict = {"loss": loss}
            return train_dict
        else:
            pred_positions, pred_yaws = outputs[..., :2], outputs[..., 2:3]
            if self.normalize_targets:
                pred_positions *= self.xy_scale

            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            if attns is not None:
                eval_dict["attention_weights"] = attns
            return eval_dict

