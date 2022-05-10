from typing import Dict,  Optional
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.planning.vectorized.common import build_target_normalization, pad_avail, pad_points
from l5kit.planning.vectorized.global_graph import MultiheadAttentionGlobalHead, VectorizedEmbedding
from l5kit.planning.rasterized.model import RasterizedPlanningModel

from l5kit.simulation.unroll import ClosedLoopSimulator
# from .local_graph import LocalSubGraph, SinusoidalPositionalEmbedding

from torch import nn


class VectorOfflineRLModel(VectorizedModel):
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
            cfg: dict,
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

        # num_targets = 3  # this will limit queries number to 1

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

        self.cfg = cfg
        num_outputs = len(weights_scaling)
        num_timesteps = num_targets // num_outputs

        if self._d_global != self._d_local:
            self.global_from_local = nn.Linear(self._d_local, self._d_global)

        # default values: num_timesteps = 1, num_outputs = 3
        self.global_head = MultiheadAttentionGlobalHead(
            self._d_global, num_timesteps, num_outputs, dropout=self._global_head_dropout
        )

        # Todo Dim for prediction outputs?
        num_other_agent = 30
        self.global_prediction_head = MultiheadAttentionGlobalHead(
            self._d_global, num_timesteps, num_outputs * num_other_agent, dropout=self._global_head_dropout
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
        outputs, attns, all_other_agent_prediction = self.model_call(
            agents_polys, map_polys, agents_availabilities, map_availabilities, type_embedding, lane_bdry_len
        )

        # call the prediction model

        # calculate loss or return predicted position for inference
        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            xy = data_batch["target_positions"]  # ~ (12, 12, 2)
            yaw = data_batch["target_yaws"]  # ~ (12, 12, 1)
            if self.normalize_targets:
                xy /= self.xy_scale
            targets = torch.cat((xy, yaw), dim=-1)  # ~ (12, 12, 3), (batch_size, timestamp, feature_num)
            target_weights = data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling

            # todo add other agents
            xy_other = data_batch["all_other_agents_future_positions"]  # ~ (12, 30, 12, 2)
            yaw_other = data_batch["all_other_agents_future_yaws"]  # ~ (12, 30, 12, 1)

            # todo normalize other agents target position
            # if self.normalize_targets:
            #     xy_other /= self.xy_scale
            all_other_agents_targets = torch.cat((xy_other, yaw_other), dim=-1)  # ~ (12, 30, 12, 3)
            all_other_agents_targets_weights = data_batch["all_other_agents_future_availability"].unsqueeze(-1)

            loss_imitate = torch.mean(self.criterion(outputs, targets) * target_weights)
            loss_other_agent_pred = torch.mean(
                self.criterion(all_other_agent_prediction, all_other_agents_targets) * all_other_agents_targets_weights)

            # from l5kit.geometry.transform import transform_points
            # transform_points(all_other_agents_targets[0], data_batch["agent_from_world"][0])

            # for other agent
            # data_batch["all_other_agents_history_positions"], data_batch['other_agents_polyline']
            # data_batch['all_other_agents_future_positions']

            # for ego
            # data_batch['agent_trajectory_polyline']
            # data_batch['target_positions']

            # data_batch['agent_from_world']
            # data_batch['agent_trajectory_polyline']
            # data_batch['target_positions']

            # all
            # data_batch,data_batch["all_other_agents_history_positions"][0], data_batch['other_agents_polyline'][0],data_batch['all_other_agents_future_positions'][0],data_batch['target_positions'][0], data_batch['agent_trajectory_polyline'][0],data_batch['agent_from_world']

            loss = self.cfg['imitate_loss_weight'] * loss_imitate + self.cfg['pred_loss_weight'] * loss_other_agent_pred

            train_dict = {"loss": loss, "loss_imitate": loss_imitate, "loss_other_agent_pred": loss_other_agent_pred}
            return train_dict
        else:
            pred_positions, pred_yaws = outputs[..., :2], outputs[..., 2:3]
            if self.normalize_targets:
                pred_positions *= self.xy_scale

            pred_positions_other_agents, pred_yaws_other_agents = all_other_agent_prediction[..., :2], all_other_agent_prediction[..., 2:3]

            eval_dict = {"positions": pred_positions, "yaws": pred_yaws,
                         "positions_other_agents": pred_positions_other_agents, "yaws_other_agents": pred_yaws_other_agents}

            if attns is not None:
                eval_dict["attention_weights"] = attns
            return eval_dict

    def model_call(
            self,
            agents_polys: torch.Tensor,
            static_polys: torch.Tensor,
            agents_avail: torch.Tensor,
            static_avail: torch.Tensor,
            type_embedding: torch.Tensor,
            lane_bdry_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """ Encapsulates calling the global_head (TODO?) and preparing needed data.

        :param agents_polys: dynamic elements - i.e. vectors corresponding to agents
        :param static_polys: static elements - i.e. vectors corresponding to map elements
        :param agents_avail: availability of agents
        :param static_avail: availability of map elements
        :param type_embedding:
        :param lane_bdry_len:
        """
        # Standardize inputs
        agents_polys_feats = torch.cat(
            [agents_polys[:, :1] / self.agent_std, agents_polys[:, 1:] / self.other_agent_std], dim=1
        )
        static_polys_feats = static_polys / self.other_agent_std

        all_polys = torch.cat([agents_polys_feats, static_polys_feats], dim=1)
        all_avail = torch.cat([agents_avail, static_avail], dim=1)

        # Embed inputs, calculate positional embedding, call local subgraph
        all_embs, invalid_polys = self.embed_polyline(all_polys, all_avail)
        if hasattr(self, "global_from_local"):
            all_embs = self.global_from_local(all_embs)

        # transformer - TODO?
        all_embs = F.normalize(all_embs, dim=-1) * (self._d_global ** 0.5)
        all_embs = all_embs.transpose(0, 1)

        other_agents_len = agents_polys.shape[1] - 1

        # disable certain elements on demand
        if self.disable_other_agents:
            invalid_polys[:, 1: (1 + other_agents_len)] = 1  # agents won't create attention

        if self.disable_map:  # lanes (mid), crosswalks, and lanes boundaries.
            invalid_polys[:, (1 + other_agents_len):] = 1  # lanes won't create attention

        if self.disable_lane_boundaries:
            type_embedding = type_embedding[:-lane_bdry_len]

        invalid_polys[:, 0] = 0  # make AoI always available in global graph

        # call and return global graph
        # all_embs ~ (81, 12, 256), invalid_polys ~ (12, 81)
        # outputs ~ (12, 1, 3), attns ~ (12, 1, 81)
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)

        batch_size, future_frames, agent_feature = outputs.shape

        all_other_agent_prediction, attns_prediction = self.global_prediction_head(all_embs, type_embedding, invalid_polys)
        # all_other_agent_prediction ~ (12, 30, 1, 4)
        all_other_agent_prediction = all_other_agent_prediction.reshape(batch_size, future_frames, agent_feature, -1)
        # ~ (batch_size, num_all_other_agents, future_frames, agent_feature)
        all_other_agent_prediction = all_other_agent_prediction.permute(0, 3, 1, 2)

        return outputs, attns, all_other_agent_prediction
