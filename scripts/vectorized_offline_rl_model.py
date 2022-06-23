from typing import Dict
from typing import List
from typing import Tuple

import torch
import torch.nn.functional as F
# from l5kit.planning.vectorized.common import
from l5kit.planning.vectorized.common import pad_avail
from l5kit.planning.vectorized.common import pad_points
from l5kit.planning.vectorized.common import transform_points
from l5kit.planning.vectorized.global_graph import MultiheadAttentionGlobalHead
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from torch import nn


import reward as reward
from l5kit.planning.vectorized.common import build_target_normalization

from l5kit.planning.vectorized.global_graph import VectorizedEmbedding
from l5kit.planning.vectorized.local_graph import SinusoidalPositionalEmbedding


from l5kit.planning.vectorized.local_graph import LocalSubGraph
from reward import AGENT_EXTENT
from reward import AGENT_TRAJECTORY_POLYLINE
from reward import AGENT_YAWS
from reward import OTHER_AGENTS_EXTENTS
from reward import OTHER_AGENTS_POLYLINE
from reward import OTHER_AGENTS_YAWS



# from .local_graph import LocalSubGraph, SinusoidalPositionalEmbedding

class EnsembleOfflineRLModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        # self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, data):
        if self.training:
            # results = [model(data) for model, data in zip(self.models, data_list)]
            results = [model(data) for model in self.models]
            return results
        else:
            first_step, trajectory, one_step_planning, one_step_other_agents_prediction, _ = self.mpc(data)
            return first_step

            # results = self.models[0](data)
            # return results

            # #只考虑策略网络输出
            # first_step, _, trajectory_value = self.models[0].inference(data)
            # eval_dict = {
            #     "positions": first_step[..., :2],
            #     "yaws": first_step[..., 2:3],
            # }
            # return eval_dict

    def mpc(self, data, inference_steps=None):
        first_step_list = []
        trajectory_value_list = []
        trajectory_planning_list = []

        one_step_planning_list = []
        one_step_other_agents_prediction_list = []

        for model in self.models:
            first_step, trajectory_planning, trajectory_value, one_step_planning, one_step_other_agents_prediction = \
                model.inference(data, inference_steps)
            # first_step, _, trajectory_value = model.inference(data)
            first_step_list.append(first_step)
            trajectory_value_list.append(trajectory_value)
            trajectory_planning_list.append(trajectory_planning)
            one_step_planning_list.append(one_step_planning)
            one_step_other_agents_prediction_list.append(one_step_other_agents_prediction)

        first_step = torch.stack(first_step_list, dim=0)
        trajectory_value = torch.stack(trajectory_value_list, dim=0)
        trajectory_planning = torch.stack(trajectory_planning_list, dim=0)
        one_step_planning = torch.stack(one_step_planning_list, dim=0)
        one_step_other_agents_prediction = torch.stack(one_step_other_agents_prediction_list, dim=0)

        index = torch.argmax(trajectory_value, dim=0)
        final_first_step = torch.zeros_like(first_step_list[0])
        final_trajectory_planning = torch.zeros_like(trajectory_planning_list[0])
        final_one_step_planning = torch.zeros_like(one_step_planning_list[0])
        final_one_step_other_agents_prediction = torch.zeros_like(one_step_other_agents_prediction_list[0])

        batch_size = len(index)
        for i in range(batch_size):
            final_first_step[i] = first_step[index[i], i, :, :]
            final_trajectory_planning = trajectory_planning[index[i], i, :, :]
            final_one_step_planning = one_step_planning[index[i], i, :, :]
            final_one_step_other_agents_prediction = one_step_other_agents_prediction[index[i], i, :, :]
        # print(index, final_first_step)

        first_step_output = {
            "positions": final_first_step[..., :2],
            "yaws": final_first_step[..., 2:3],
        }
        final_trajectory_output = {
            "positions": final_trajectory_planning[..., :2],
            "yaws": final_trajectory_planning[..., 2:3],
        }
        all_trajectory_and_value = {
            "trajectory": trajectory_planning,
            "trajectory_value": trajectory_value,
        }

        return first_step_output, final_trajectory_output, final_one_step_planning, final_one_step_other_agents_prediction, all_trajectory_and_value

    def plan_trajectory(self, data):
        return data


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
            limit_predicted_yaw: bool = True,
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



        self._d_local = 128
        self._d_global = 256

        self._agent_features = ["start_x", "start_y", "yaw"]
        self._lane_features = ["start_x", "start_y", "tl_feature"]
        self._vector_agent_length = len(self._agent_features)
        self._vector_lane_length = len(self._lane_features)
        self._subgraph_layers = 3

        self.register_buffer("weights_scaling", torch.as_tensor(weights_scaling))
        self.criterion = criterion

        self.normalize_targets = True
        num_outputs = len(weights_scaling)
        num_timesteps = num_targets // num_outputs

        if self.normalize_targets:
            scale = build_target_normalization(num_timesteps)
            self.register_buffer("xy_scale", scale)

        # normalization buffers
        self.register_buffer("agent_std", torch.tensor([1.6919, 0.0365, 0.0218]))
        self.register_buffer("other_agent_std", torch.tensor([33.2631, 21.3976, 1.5490]))

        self.input_embed = nn.Linear(self._vector_agent_length, self._d_local)
        self.positional_embedding = SinusoidalPositionalEmbedding(self._d_local)
        self.type_embedding = VectorizedEmbedding(self._d_global)

        self.disable_pos_encode = False

        self.local_subgraph = LocalSubGraph(num_layers=self._subgraph_layers, dim_in=self._d_local)

        if self._d_global != self._d_local:
            self.global_from_local = nn.Linear(self._d_local, self._d_global)

        self.global_head = MultiheadAttentionGlobalHead(
            self._d_global, num_timesteps, num_outputs, dropout=self._global_head_dropout
        )


        self.cfg = cfg
        num_outputs = len(weights_scaling)
        num_timesteps = num_targets // num_outputs

        self.limit_predicted_yaw = limit_predicted_yaw

        # todo
        self.normalize_targets = False

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

        self.reward_head = MultiheadAttentionGlobalHead(
            self._d_global, 1, 1, dropout=self._global_head_dropout
        )

        self.value_head = MultiheadAttentionGlobalHead(
            self._d_global, 1, 1, dropout=self._global_head_dropout
        )

        self.speed_head = MultiheadAttentionGlobalHead(
            self._d_global, 1, 1, dropout=self._global_head_dropout
        )

    def inference(self,
                  data_batch: Dict[str, torch.Tensor],
                  inference_step=None
                  ) -> Dict[str, torch.Tensor]:

        # calculate rewards
        # distance_to_center = reward.get_distance_to_centroid_per_batch(data_batch)
        # min_distance_to_other = reward.get_distance_to_other_agents_per_batch(data_batch)
        # target_reward = -distance_to_center + min_distance_to_other

        # trajectory_value = copy.deepcopy(target_reward)  # 初始轨迹值  相当于加上了r1

        # trajectory_len = 12  # 往前预测12次
        # future_num_frames = data_batch["target_availabilities"].shape[1]
        history_num_frames = data_batch["history_availabilities"].shape[1] - 1
        if inference_step is None:
            inference_step = self.cfg["train_data_loader"]["pred_len"]

        # batch_size = self.cfg["train_data_loader"]["batch_size"]

        batch_size = data_batch["history_availabilities"].shape[0]
        device = data_batch["history_availabilities"].device

        # ==== LANES ====
        # batch size x num lanes x num vectors x num features
        polyline_keys = ["lanes_mid", "crosswalks"]
        if not self.disable_lane_boundaries:
            polyline_keys += ["lanes"]
        avail_keys = [f"{k}_availabilities" for k in polyline_keys]

        max_num_vectors = max([data_batch[key].shape[-2] for key in polyline_keys])

        static_polys = torch.cat([pad_points(data_batch[key], max_num_vectors) for key in polyline_keys], dim=1)
        static_polys[..., -1].fill_(0)

        # batch size x num lanes x num vectors
        static_avail = torch.cat([pad_avail(data_batch[key], max_num_vectors) for key in avail_keys], dim=1)

        # batch_size x (1 + M) x num vectors
        # agents_availabilities = pad_avail(agents_availabilities, max_num_vectors)

        # batch_size x (1 + M) x num features
        lane_bdry_len = data_batch["lanes"].shape[1]

        # === agents ===
        agents_past_polys = torch.cat(
            [data_batch["agent_trajectory_polyline"].unsqueeze(1), data_batch["other_agents_polyline"]], dim=1
        )  # agent_trajectory_polyline 12，4，3    other_agents_polyline  12，30，4，3
        # batch_size x (1 + M) x seq len
        agents_past_avail = torch.cat(
            [
                data_batch["agent_polyline_availability"].unsqueeze(1),
                data_batch["other_agents_polyline_availability"],
            ],
            dim=1,
        )
        # ego_polys = torch.zeros(batch_size, )
        trajectory_point_dim = 3
        # avail_point_dim = agents_past_avail.shape[-1]
        num_agents = agents_past_avail.shape[1]
        agents_polys_horizon = torch.zeros((batch_size, num_agents, history_num_frames + 1 + inference_step,
                                            trajectory_point_dim),
                                           device=device)
        agents_avail_horizon = torch.ones((batch_size, num_agents, history_num_frames + 1 + inference_step),
                                          device=device)
        # agents_avail = torch.zeros(batch_size, 1+num_agents, history_num_frames+1+trajectory_len, )

        agents_polys_horizon[:, :, :history_num_frames + 1] = torch.flip(agents_past_polys, [2])
        agents_avail_horizon[:, :, :history_num_frames + 1] = torch.flip(agents_past_avail, [2])

        current_timestep = agents_past_polys.shape[2] - 1
        window_size = agents_past_polys.shape[2]

        one = torch.ones_like(data_batch["target_yaws"][:, 0])
        zero = torch.zeros_like(data_batch["target_yaws"][:, 0])

        # ====== Transformation between local spaces
        # NOTE: we use the standard convention A_from_B to indicate that a matrix/yaw/translation
        # converts a point from the B space into the A space
        # e.g. if pB = (1,0) and A_from_B = (-1, 1) then pA = (0, 1)
        # NOTE: we use the following convention for names:
        # t0 -> space at 0, i.e. the space we pull out of the data for which ego is in (0, 0) with no yaw
        # ts -> generic space at step t = s > 0 (predictions at t=s are in this space)
        # tsplus -> space at s+1 (proposal new ts, built from prediction at t=s)
        # A_from_B -> indicate a full 2x3 RT matrix from B to A
        # yaw_A_from_B -> indicate a yaw from B to A
        # tr_A_from_B -> indicate a translation (XY) from B to A
        # NOTE: matrices (and yaw) we need to keep updated while we loop:
        # t0_from_ts -> bring a point from the current space into the data one (e.g. for visualisation)
        # ts_from_t0 -> bring a point from data space into the current one (e.g. to compute loss
        t0_from_ts = torch.eye(3, device=one.device).unsqueeze(0).repeat(batch_size, 1, 1)
        ts_from_t0 = t0_from_ts.clone()
        yaw_t0_from_ts = zero
        yaw_ts_from_t0 = zero

        trajectory_value = torch.zeros(batch_size, device=one.device)

        for idx in range(inference_step):
            # ==== AGENTS ====
            # batch_size x (1 + M) x seq len x self._vector_length
            # agents_polys = torch.cat(
            #     [data_batch["agent_trajectory_polyline"].unsqueeze(1), data_batch["other_agents_polyline"]], dim=1
            # )  # agent_trajectory_polyline 12，4，3    other_agents_polyline  12，30，4，3

            agents_polys_step = torch.flip(
                agents_polys_horizon[:, :, current_timestep - window_size + 1:current_timestep + 1], [2]
            ).clone()

            # todo
            agents_avail_step = agents_past_avail

            # batch_size x (1 + M) x num vectors x self._vector_length
            agents_polys_step = pad_points(agents_polys_step, max_num_vectors)
            agents_avail_step = pad_avail(agents_avail_step, max_num_vectors)

            # transform agents and statics into right coordinate system (ts)
            agents_polys_step = transform_points(agents_polys_step, ts_from_t0, agents_avail_step, yaw_ts_from_t0)
            static_avail_step = static_avail.clone()
            static_polys_step = transform_points(static_polys.clone(), ts_from_t0, static_avail_step)

            type_embedding = self.type_embedding(data_batch).transpose(0, 1)

            # call the model with these features
            outputs, attns, all_other_agent_prediction, reward_outputs, value_outputs, speed_outputs= self.model_call(
                agents_polys_step,
                static_polys_step,
                agents_avail_step,
                static_avail_step,
                type_embedding,
                lane_bdry_len
            )  # outputs: 12,12,3  ,  all_other_agent_prediction: 12,30,12,3 , reward_outputs: 12  , value_outputs: 12

            # outputs are in ts space (optionally xy normalised)
            # only the first step
            pred_xy_step = outputs[:, 0, :2]
            pred_yaw_step = outputs[:, 0, 2:3] if not self.limit_predicted_yaw else 0.3 * torch.tanh(outputs[:, 0, 2:3])

            # only select the first step
            pred_other_agents_xy_step = all_other_agent_prediction[:, :, 0, :2]
            pred_other_agents_yaw_step = all_other_agent_prediction[:, :, 0,
                                         2:3] if not self.limit_predicted_yaw else 0.3 * torch.tanh(
                all_other_agent_prediction[:, :, 0, 2:3])

            # todo normalise
            pred_xy_step_unnorm = pred_xy_step
            pred_other_agents_xy_step_unnorm = pred_other_agents_xy_step
            if self.normalize_targets:
                raise NotImplementedError
                pred_xy_step_unnorm = pred_xy_step * self.xy_scale[0]
                pred_other_agents_xy_step = None

            # ==== SAVE PREDICTIONS & GT

            def get_point(xy, coord_transform):
                xy_new = xy[:, None, :] \
                         @ coord_transform[..., :2, :2].transpose(1, 2) \
                         + coord_transform[..., :2, -1:].transpose(1, 2)
                return xy_new

            # pred_xy_step_t0 = pred_xy_step_unnorm[:, None, :] \
            #                   @ t0_from_ts[..., :2, :2].transpose(1, 2) \
            #                   + t0_from_ts[..., :2, -1:].transpose(1, 2)
            pred_xy_step_t0 = get_point(pred_xy_step_unnorm, t0_from_ts)
            pred_xy_step_t0 = pred_xy_step_t0[:, 0]
            pred_yaw_step_t0 = pred_yaw_step + yaw_t0_from_ts

            # one implementation
            # pred_other_agents_xy_step_t0 = torch.zeros_like(pred_other_agents_xy_step_unnorm)
            # for agents_idx in range(pred_other_agents_xy_step_unnorm.shape[1]):
            #     agents_xy = pred_other_agents_xy_step_unnorm[:, agents_idx]
            #     agents_xy_t0 = get_point(agents_xy, t0_from_ts)
            #     pred_other_agents_xy_step_t0[:, agents_idx] = agents_xy_t0[:, 0]

            # another implementation
            pred_other_agents_xy_step_t0 = get_point(pred_other_agents_xy_step_unnorm, t0_from_ts)[:, 0]
            pred_other_agnets_yaw_step_t0 = pred_other_agents_yaw_step + yaw_t0_from_ts.unsqueeze(1)

            # ==== UPDATE HISTORY WITH INFORMATION FROM PREDICTION

            # update transformation matrices
            t0_from_ts, ts_from_t0, yaw_t0_from_ts, yaw_ts_from_t0 = self.update_transformation_matrices(
                pred_xy_step_unnorm, pred_yaw_step, t0_from_ts, ts_from_t0, yaw_t0_from_ts, yaw_ts_from_t0, zero, one
            )

            # update AoI
            agents_polys_horizon[:, 0, current_timestep + 1, :2] = pred_xy_step_t0
            agents_polys_horizon[:, 0, current_timestep + 1, 2:3] = pred_yaw_step_t0
            agents_polys_horizon[:, 1:, current_timestep + 1, :2] = pred_other_agents_xy_step_t0
            agents_polys_horizon[:, 1:, current_timestep + 1, 2:3] = pred_other_agnets_yaw_step_t0

            agents_avail_horizon[:, 0, current_timestep + 1] = 1
            agents_avail_horizon[:, 1:, current_timestep + 1] = agents_avail_horizon[:, 1:, current_timestep]

            # agents_availabilities[:, 0]

            # move time window one step into the future
            current_timestep += 1

            # todo
            # # 更新data_batch
            # data_batch["agent_trajectory_polyline"] = torch.flip(outputs[:, :4, :].detach(), dims=[1])
            # # 只取前4个点，作为下一个状态 并颠倒s
            # data_batch["history_yaws"] = torch.flip(outputs[:, :4, -1].view(batch_size, 4, 1).detach(), dims=[1])
            # data_batch["other_agents_polyline"] = torch.flip(all_other_agent_prediction[:, :, :4, :].detach(), dims=[2])
            # # 只取前4个点，作为下一个状态
            # data_batch["all_other_agents_history_yaws"] = torch.flip(
            #     all_other_agent_prediction[:, :, :4, -1].view(batch_size, 30, 4, 1).detach(), dims=[2])
            # # 只取前4个点，作为下一个状态
            if idx == 0:
                first_step = outputs[:, :1, :]  # 轨迹中的第一步
                one_step_planning = outputs.clone()
                one_step_other_agents_prediction = all_other_agent_prediction.clone()

            # ==== UPDATE REWARD
            # batch_bind_ = {
            #     AGENT_TRAJECTORY_POLYLINE: agents_polys[:, 0],
            #     AGENT_YAWS: agents_polys[:, 0, :, 2:3],
            #     AGENT_EXTENT: agents_extent[:, 0],
            #     OTHER_AGENTS_POLYLINE: agents_polys[:, 1:, :, :],
            #     OTHER_AGENTS_YAWS: agents_polys[:, 1:, :, 2:3],
            #     OTHER_AGENTS_EXTENTS: agents_extent[:, 1:, :],
            #     "all_other_agents_types": data_batch["all_other_agents_types"],
            # }

            # calculate rewards
            # distance_to_center = reward.get_distance_to_centroid_per_batch(data_batch)
            # min_distance_to_other = reward.get_distance_to_other_agents_per_batch(
            #     data_batch, )
            # target_reward = -distance_to_center + min_distance_to_other
            # target_reward = reward_outputs
            # trajectory_value += reward_outputs
            trajectory_value += reward_outputs

            # todo 好像应该输出下一步状态值
            if idx == inference_step - 1:
                trajectory_value += value_outputs


        return first_step, agents_polys_horizon, trajectory_value, one_step_planning, one_step_other_agents_prediction

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ==== get additional info from the batch, or fall back to sensible defaults
        future_num_frames = data_batch["target_availabilities"].shape[1]
        history_num_frames = data_batch["history_availabilities"].shape[1] - 1

        # ==== Past  info ====
        agents_past_polys = torch.cat(
            [data_batch["agent_trajectory_polyline"].unsqueeze(1), data_batch["other_agents_polyline"]], dim=1
        )
        agents_past_avail = torch.cat(
            [data_batch["agent_polyline_availability"].unsqueeze(1), data_batch["other_agents_polyline_availability"]],
            dim=1,
        )
        agents_past_extent = torch.cat(
            [data_batch["history_extents"].unsqueeze(1), data_batch["all_other_agents_history_extents"]], dim=1
        )


        # ==== Static (LANES) info ====
        # batch size x num lanes x num vectors x num features
        static_keys = ["lanes_mid", "crosswalks"]
        if not self.disable_lane_boundaries:
            static_keys += ["lanes"]
        avail_keys = [f"{k}_availabilities" for k in static_keys]

        max_num_vectors = max([data_batch[key].shape[-2] for key in static_keys])

        static_polys = torch.cat([pad_points(data_batch[key], max_num_vectors) for key in static_keys], dim=1)
        static_polys[..., -1].fill_(0)  # NOTE: this is a hack
        # batch size x num lanes x num vectors
        static_avail = torch.cat([pad_avail(data_batch[key], max_num_vectors) for key in avail_keys], dim=1)

        # ==== Future info ====
        agents_future_positions = torch.cat(
            [data_batch["target_positions"].unsqueeze(1), data_batch["all_other_agents_future_positions"]], dim=1
        )
        agents_future_yaws = torch.cat(
            [data_batch["target_yaws"].unsqueeze(1), data_batch["all_other_agents_future_yaws"]], dim=1
        )
        agents_future_avail = torch.cat(
            [data_batch["target_availabilities"].unsqueeze(1), data_batch["all_other_agents_future_availability"]],
            dim=1,
        )
        agents_future_extent = torch.cat(
            [data_batch["target_extents"].unsqueeze(1), data_batch["all_other_agents_future_extents"]], dim=1
        )

        # concat XY and yaw to mimic past
        agents_future_polys = torch.cat([agents_future_positions, agents_future_yaws], dim=3)

        # Combine past and future agent information.
        # Future information is ordered [T+1, T+2, ...], past information [T, T-1, T-2, ...].
        # We thus flip past vectors and by concatenating get [..., T-2, T-1, T, T+1, T+2, ...].
        # Now, at each step T the current time window of interest simply is represented by the indices
        # T + agents_past_polys.shape[2] - window_size + 1: T + agents_past_polys.shape[2] + 1.
        # During the training loop, we will fetch this information, as well as static features,
        # which is all represented in the space of T = 0.
        # We then transform this into the space of T and feed this to the model.
        # Eventually, we shift our time window one step into the future.
        # See below for more information about used coordinate spaces.

        # batch_size x (1 + num_agents) x (history_num_frames + 1 + future_num_frames) x dim
        agents_polys = torch.cat([torch.flip(agents_past_polys, [2]), agents_future_polys], dim=2)
        agents_avail = torch.cat([torch.flip(agents_past_avail.contiguous(), [2]), agents_future_avail], dim=2)
        agents_extent = torch.cat([torch.flip(agents_past_extent, [2]), agents_future_extent], dim=2)

        window_size = agents_past_polys.shape[2]  # (history_num_frames + 1)
        current_timestep = agents_past_polys.shape[2] - 1
        # num_frames_per_sample = agents_polys.shape[2]

        # Load and prepare vectors for the model call, split into map and agents

        # === compute reward ===
        # calculate rewards ~ (batch_size x (history_frame + 1 + future_frame))
        ego_polys = agents_polys[:, 0, :, :]
        # batch_size x (1 + future_frame)  x window_size x dim
        ego_polys_samples = ego_polys.unfold(1, window_size, 1).transpose(2, 3)
        # todo add aval for lanes_mid
        # flip() is used to transform [t-3, t-2, t-1, t] into [t, t-1, t-2, t-3]
        ego_distance_to_centroid = torch.stack([reward.get_distance_to_centroid_by_element(
            ego_polys_samples[:, idx, :, :2], data_batch["lanes_mid"][..., :2]) for idx in
            range(ego_polys_samples.shape[1])
        ])
        # batch_size x future_frame (t, t+1, ..., t+future_frame)
        ego_distance_to_centroid = ego_distance_to_centroid.transpose(1, 0)
        ego_distance_to_centroid_future = ego_distance_to_centroid[:, 1:]
        # distance_to_center = reward.get_distance_to_centroid_per_batch(data_batch)

        # min_distance_to_other = reward.get_distance_to_other_agents_per_batch(data_batch)

        # ego: batch_size x (history_frame + 1 + future_frame) x dim
        # agents: batch_size x num_agents x (history_frame + 1 + future_frame) x dim
        batch_bind_history_future = {
            AGENT_TRAJECTORY_POLYLINE: agents_polys[:, 0, :, :],
            AGENT_YAWS: agents_polys[:, 0, :, 2:3],
            AGENT_EXTENT: agents_extent[:, 0, :, :],
            OTHER_AGENTS_POLYLINE: agents_polys[:, 1:, :, :],
            OTHER_AGENTS_YAWS: agents_polys[:, 1:, :, 2:3],
            OTHER_AGENTS_EXTENTS: agents_extent[:, 1:, :, :],
            "all_other_agents_types": data_batch["all_other_agents_types"],
        }
        min_distance_to_other = torch.stack([
            reward.get_distance_to_other_agents_per_batch(batch_bind_history_future, idx) for idx in range(
                current_timestep + 1, history_num_frames + 1 + future_num_frames)]
        )
        # batch x future_frame
        min_distance_to_other = min_distance_to_other.transpose(1, 0)

        target_reward_all = -ego_distance_to_centroid_future + min_distance_to_other
        target_reward = target_reward_all[:, 0].clone()  # one-step reward

        # === compute values ===
        pred_len = self.cfg["train_data_loader"]["pred_len"]
        # batch_size = self.cfg["train_data_loader"]["batch_size"]

        assert pred_len <= future_num_frames
        truncated_value = target_reward_all[:, :pred_len].sum(axis=1)




        # for element_ix in range(batch_size):
        #     truncated_value = sum(target_reward[element_ix + 1:element_ix + pred_len + 1])
        #     truncated_value_batch.append(truncated_value)
        # truncated_value_batch = torch.stack(truncated_value_batch)

        # ==== AGENTS ====
        # batch_size x (1 + M) x num vectors x self._vector_length
        agents_polys = pad_points(agents_past_polys, max_num_vectors)

        # batch_size x (1 + M) x num vectors
        agents_availabilities = pad_avail(agents_past_avail, max_num_vectors)

        # batch_size x (1 + M) x num features
        type_embedding = self.type_embedding(data_batch).transpose(0, 1)
        lane_bdry_len = data_batch["lanes"].shape[1]

        # call the model with these features
        outputs, attns, all_other_agent_prediction , reward_outputs, value_outputs, speed_outputs= self.model_call(
            agents_polys,
            static_polys,
            agents_availabilities,
            static_avail,
            type_embedding,
            lane_bdry_len
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
            # loss_reward = torch.mean(self.criterion(target_reward, reward_outputs))
            # loss_value = torch.mean(self.criterion(truncated_value, value_outputs))

            # loss_speed = torch.mean(self.criterion(data_batch['speed'], speed_outputs))


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

            loss = self.cfg['imitate_loss_weight'] * loss_imitate + self.cfg[
                'pred_loss_weight'] * loss_other_agent_pred
            #+ loss_reward + loss_value * 0.02 + loss_speed

            train_dict = {"loss": loss, "loss_imitate": loss_imitate, "loss_other_agent_pred": loss_other_agent_pred
                          }
            # "loss_reward": loss_reward, "loss_value": loss_value, "loss_speed": loss_speed

            return train_dict
        else:
            # print("inference!!!")
            pred_positions, pred_yaws = outputs[..., :2], outputs[..., 2:3]
            if self.normalize_targets:
                pred_positions *= self.xy_scale

            pred_positions_other_agents, pred_yaws_other_agents = all_other_agent_prediction[...,
                                                                  :2], all_other_agent_prediction[..., 2:3]

            eval_dict = {"positions": pred_positions, "yaws": pred_yaws,
                         "positions_other_agents": pred_positions_other_agents,
                         "yaws_other_agents": pred_yaws_other_agents}

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
    ):
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

        all_other_agent_prediction, attns_prediction = self.global_prediction_head(all_embs, type_embedding,
                                                                                   invalid_polys)
        # all_other_agent_prediction ~ (12, 30, 1, 4)
        all_other_agent_prediction = all_other_agent_prediction.reshape(batch_size, future_frames, agent_feature, -1)
        # ~ (batch_size, num_all_other_agents, future_frames, agent_feature)
        all_other_agent_prediction = all_other_agent_prediction.permute(0, 3, 1, 2)

        reward_outputs, reward_attns = self.reward_head(all_embs, type_embedding, invalid_polys)

        value_outputs, _ = self.value_head(all_embs, type_embedding, invalid_polys)

        speed_outputs, _ =self.speed_head(all_embs, type_embedding, invalid_polys)

        return outputs, attns, all_other_agent_prediction ,reward_outputs.view(-1), value_outputs.view(-1), speed_outputs.view(-1)

    # refer to closed_loop_model
    def update_transformation_matrices(self, pred_xy_step_unnorm: torch.Tensor, pred_yaw_step: torch.Tensor,
                                       t0_from_ts: torch.Tensor, ts_from_t0: torch.Tensor, yaw_t0_from_ts: torch.Tensor,
                                       yaw_ts_from_t0: torch.Tensor, zero: torch.Tensor, one: torch.Tensor
                                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Updates the used transformation matrices to reflect AoI's new position.
        """
        tr_tsplus_from_ts = -pred_xy_step_unnorm
        yaw_tsplus_from_ts = -pred_yaw_step
        yaw_ts_from_tsplus = pred_yaw_step

        # NOTE: these are full roto-translation matrices. We use the closed form and not invert for performance reasons.
        # tsplus_from_ts will bring the current predictions at ts into 0.
        tsplus_from_ts = torch.cat(
            [
                yaw_tsplus_from_ts.cos(),
                -yaw_tsplus_from_ts.sin(),
                tr_tsplus_from_ts[:, :1] * yaw_tsplus_from_ts.cos()
                - tr_tsplus_from_ts[:, 1:] * yaw_tsplus_from_ts.sin(),
                yaw_tsplus_from_ts.sin(),
                yaw_tsplus_from_ts.cos(),
                tr_tsplus_from_ts[:, :1] * yaw_tsplus_from_ts.sin()
                + tr_tsplus_from_ts[:, 1:] * yaw_tsplus_from_ts.cos(),
                zero,
                zero,
                one,
            ],
            dim=1,
        ).view(-1, 3, 3)
        # this is only required to keep t0_from_ts updated
        ts_from_tsplus = torch.cat(
            [
                yaw_ts_from_tsplus.cos(),
                -yaw_ts_from_tsplus.sin(),
                -tr_tsplus_from_ts[:, :1],
                yaw_ts_from_tsplus.sin(),
                yaw_ts_from_tsplus.cos(),
                -tr_tsplus_from_ts[:, 1:],
                zero,
                zero,
                one,
            ],
            dim=1,
        ).view(-1, 3, 3)

        # update RTs and yaws by including tsplus (next step ts)
        t0_from_ts = t0_from_ts @ ts_from_tsplus
        ts_from_t0 = tsplus_from_ts @ ts_from_t0
        yaw_t0_from_ts = yaw_t0_from_ts + yaw_ts_from_tsplus
        yaw_ts_from_t0 = yaw_ts_from_t0 + yaw_tsplus_from_ts

        return t0_from_ts, ts_from_t0, yaw_t0_from_ts, yaw_ts_from_t0
