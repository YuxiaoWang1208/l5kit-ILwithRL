from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import build_target_normalization, pad_avail, pad_points
from .open_loop_model import VectorizedModel
from .global_graph import MultimodalMultiheadAttentionGlobalHead



class VectorizedMultiModalPlanningModel(VectorizedModel):
    """Vector-based multimodal planning model."""

    def __init__(
        self,
        history_num_frames_ego: int,
        history_num_frames_agents: int,
        future_num_frames: int,
        num_outputs: int,
        num_modes: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        global_head_dropout: float,
        disable_other_agents: bool,
        disable_map: bool,
        disable_lane_boundaries: bool,
        coef_alpha: float = 0.5,
    ) -> None:
        """Initializes the multimodal planning model.

        :param history_num_frames_ego: number of history ego frames to include
        :param history_num_frames_agents: number of history agent frames to include
        :param future_num_frames: number of future frames to predict
        :param num_outputs: number of output dimensions, by default is 3: x, y, heading
        :param num_modes: number of modes in predicted outputs
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param global_head_dropout: float in range [0,1] for the dropout in the MHA global head. Set to 0 to disable it
        :param disable_other_agents: ignore agents
        :param disable_map: ignore map
        :param disable_lane_boundaries: ignore lane boundaries
        :param coef_alpha: hyper-parameter used to trade-off between trajectory distance loss and classification
        cross-entropy loss
        """
        num_targets = (future_num_frames * num_outputs + 1) * num_modes
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
        self.num_modes = num_modes
        self.num_timestamps = future_num_frames
        self.num_outputs = num_outputs
        self.coef_alpha = coef_alpha

        self.normalize_targets = True
        num_timestamps = self.num_timestamps

        if self.normalize_targets:
            scale = build_target_normalization(num_timestamps)
            self.register_buffer("xy_scale", scale)

        # 重写父类的global_head，将输出[future_num_frames,num_outputs]合并成一个维度num_targets
        self.global_head = MultimodalMultiheadAttentionGlobalHead(
            self._d_global, num_targets, dropout=self._global_head_dropout
        )

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ## Load and prepare vectors for the model call, split into map and agents
        
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

        ## call the model with these features
        #[batch_size, (future_num_frames * num_outputs + 1) * num_modes]
        outputs_all, attns = self.model_call(
            agents_polys, map_polys, agents_availabilities, map_availabilities, type_embedding, lane_bdry_len
        )
        batch_size = len(outputs_all)

        # [batch_size, num_timestamps * num_outputs * num_modes]
        outputs = outputs_all[:, :-self.num_modes]
        # [batch_size, num_modes], classification flags
        outputs_nll = outputs_all[:, -self.num_modes:]

        # .view(-1, self.num_timesteps, self.num_outputs)


        # calculate loss or return predicted position for inference
        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
            
            # [batch_size, num_timestamps * num_modes, num_outputs]
            outputs = outputs.view(
                -1, self.num_timestamps * self.num_modes, self.num_outputs
            )

            # [batch_size, num_timestamps, 2]
            xy = data_batch["target_positions"]
            # [batch_size, num_timestamps, 1]
            yaw = data_batch["target_yaws"]

            if self.normalize_targets:
                xy /= self.xy_scale
            
            # [batch_size, num_timestamps, num_outputs]
            targets = torch.cat((xy, yaw), dim=-1)
            # [batch_size, num_timestamps]
            target_weights = data_batch["target_availabilities"] > 0.5
            # [batch_size, num_timestamps, num_outputs]
            target_weights = target_weights.unsqueeze(-1) * self.weights_scaling
            # [batch_size, num_timestamps * num_modes, num_outputs]
            losses = self.criterion(
                outputs, targets.repeat(1, self.num_modes, 1)
            ) * target_weights.repeat(1, self.num_modes, 1)
            # [batch_size, num_modes]
            cost_dist = losses.view(batch_size, self.num_modes, -1).mean(dim=-1)
            # [batch_size,],最接近target轨迹的output轨迹的编号
            assignment = cost_dist.argmin(dim=-1)
            # 最接近轨迹的距离loss
            loss_dist = cost_dist[torch.arange(batch_size, device=outputs.device), assignment].mean()
            # 每个output轨迹是接近target轨迹的分类概率loss
            loss_nll = F.cross_entropy(outputs_nll, assignment)
            train_dict = {"loss": loss_dist + self.coef_alpha * loss_nll}
            return train_dict
        else:
            outputs = outputs.view(batch_size, self.num_modes, self.num_timestamps, self.num_outputs)
            outputs_selected = outputs[torch.arange(batch_size, device=outputs.device), outputs_nll.argmax(-1)]
            pred_positions, pred_yaws = outputs_selected[..., :2], outputs_selected[..., 2:3]
            pred_pos_all = outputs[..., :2]
            pred_yaw_all = outputs[..., 2:3]
            if self.normalize_targets:
                pred_positions *= self.xy_scale
                pred_pos_all *= self.xy_scale

            eval_dict = {
                "positions": pred_positions,
                "yaws": pred_yaws,
                "positions_all": pred_pos_all,
                "yaws_all": pred_yaw_all,
            }
            if attns is not None:
                eval_dict["attention_weights"] = attns
            return eval_dict
