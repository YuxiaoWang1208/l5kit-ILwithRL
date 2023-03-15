from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch as th

from l5kit.cle.metric_set import L5MetricSet
from l5kit.environment.gym_metric_set import L2DisplacementYawMetricSet
from l5kit.simulation.unroll import SimulationOutputCLE
from l5kit.planning import utils
from l5kit.evaluation.metrics import distance_to_reference_trajectory


class Reward(ABC):
    """Base class interface for gym environment reward."""
    #: The prefix that will identify this reward class
    reward_prefix: str

    @abstractmethod
    def reset(self) -> None:
        """Reset the reward state when new episode starts.
        """
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, frame_index: int, simulated_outputs: List[SimulationOutputCLE]) -> Dict[str, float]:
        """Return the reward at a particular time-step during the episode.

        :param frame_index: the frame index for which reward is to be calculated
        :param simulated_outputs: the object contain the ego target and prediction attributes
        :return: reward at a particular frame index (time-step) during the episode containing total reward
            and individual components that make up the reward.
        """
        raise NotImplementedError


class L2DisplacementYawReward(Reward):
    """This class is responsible for calculating a reward based on
    (1) L2 displacement error on the (x, y) coordinates
    (2) Closest angle error on the yaw coordinate
    during close loop simulation within the gym-compatible L5Kit environment.

    :param reward_prefix: the prefix that will identify this reward class
    :param metric_set: the set of metrics to compute
    :param enable_clip: flag to determine whether to clip reward
    :param rew_clip_thresh: the threshold to clip the reward
    :param use_yaw: flag to penalize the yaw prediction
    :param yaw_weight: weight of the yaw error
    """

    def __init__(self, reward_prefix: str = "L2DisplacementYaw", metric_set: Optional[L5MetricSet] = None,
                 enable_clip: bool = True, rew_clip_thresh: float = 15.0,
                 use_yaw: Optional[bool] = True, yaw_weight: Optional[float] = 1.0) -> None:
        """Constructor method
        """
        self.reward_prefix = reward_prefix
        # Metric Set
        self.metric_set = metric_set if metric_set is not None else L2DisplacementYawMetricSet()

        # Verify that error metrics necessary for reward calculation are present in the metric set
        if 'yaw_error_closest_angle' not in self.metric_set.evaluation_plan.metrics_dict():
            raise RuntimeError('\'yaw_error_closest_angle\' missing in metric set')
        if 'displacement_error_l2' not in self.metric_set.evaluation_plan.metrics_dict():
            raise RuntimeError('\'displacement_error_l2\' missing in metric set')

        self.use_yaw = use_yaw
        self.yaw_weight = yaw_weight

        self.enable_clip = enable_clip
        self.rew_clip_thresh = rew_clip_thresh

    def reset(self) -> None:
        """Reset the closed loop evaluator when a new episode starts.
        """
        self.metric_set.reset()

    @staticmethod
    def slice_simulated_output(index: int, simulated_outputs: List[SimulationOutputCLE]) -> List[SimulationOutputCLE]:
        """ Slice the simulated output at a particular frame index.
        This prevent calculating metric over all frames.

        :param index: the frame index at which the simulation outputs is to be sliced
        :param simulated_outputs: the object contain the ego target and prediction attributes
        :return: the sliced simulation output
        """
        # Only the simulated and recorded ego states are used for metric calculation
        simulated_outputs[0].recorded_ego_states = simulated_outputs[0].recorded_ego_states[index:index + 1]
        simulated_outputs[0].simulated_ego_states = simulated_outputs[0].simulated_ego_states[index:index + 1]
        return simulated_outputs

    def get_reward(self, frame_index: int, simulated_outputs: List[SimulationOutputCLE]) -> Dict[str, float]:
        """Get the reward for the given step in close loop training.

        :param frame_index: the frame index for which reward is to be calculated
        :param simulated_outputs: the object contain the ego target and prediction attributes
        :return: the dictionary containing total reward and individual components that make up the reward
        """
        scene_id = simulated_outputs[0].scene_id

        # Get the simulated output value at frame index + 1
        simulated_outputs = self.slice_simulated_output(frame_index + 1, simulated_outputs)

        # Evaluate metrics on the sliced simulated output
        self.metric_set.evaluate(simulated_outputs)
        scene_metrics = self.metric_set.evaluator.scene_metric_results[scene_id]
        dist_error = scene_metrics['displacement_error_l2']
        yaw_error = self.yaw_weight * scene_metrics['yaw_error_closest_angle']

        # clip the distance error (in x, y) only, not the yaw error (yaw error is bounded).
        dist_reward = float(-dist_error.item())
        if self.enable_clip:
            dist_reward = max(-self.rew_clip_thresh, -dist_error.item())

        # use yaw
        yaw_reward = 0.0
        if self.use_yaw:
            yaw_reward -= yaw_error.item()

        # Total reward
        total_reward = dist_reward + yaw_reward

        reward_dict = {"total": total_reward, "distance": dist_reward, "yaw": yaw_reward}
        return reward_dict


class CollisionOffroadReward(Reward):
    """This class is responsible for calculating a reward based on
    (1) Distance between the ego and a nearest bounding box of other agents
    (2) Distance of the ego to the mid lane
    during close loop simulation within the gym-compatible L5Kit environment.

    :param reward_prefix: the prefix that will identify this reward class
    :param metric_set: the set of metrics to compute
    :param enable_clip: flag to determine whether to clip reward
    :param rew_clip_thresh: the threshold to clip the reward
    :param use_yaw: flag to penalize the yaw prediction
    :param yaw_weight: weight of the yaw error
    """

    def __init__(self, reward_prefix: str = "CollisionOffroad", metric_set: Optional[L5MetricSet] = None,
                 enable_clip: bool = True, rew_clip_thresh: float = 15.0,
                 use_yaw: Optional[bool] = True, yaw_weight: Optional[float] = 1.0) -> None:
        """Constructor method
        """
        self.reward_prefix = reward_prefix
        # Metric Set
        self.metric_set = metric_set if metric_set is not None else L2DisplacementYawMetricSet()

        # Verify that error metrics necessary for reward calculation are present in the metric set
        if 'yaw_error_closest_angle' not in self.metric_set.evaluation_plan.metrics_dict():
            raise RuntimeError('\'yaw_error_closest_angle\' missing in metric set')
        if 'displacement_error_l2' not in self.metric_set.evaluation_plan.metrics_dict():
            raise RuntimeError('\'displacement_error_l2\' missing in metric set')

        self.use_yaw = use_yaw
        self.yaw_weight = yaw_weight

        self.enable_clip = enable_clip
        self.rew_clip_thresh = rew_clip_thresh

    def reset(self) -> None:
        """Reset the closed loop evaluator when a new episode starts.
        """
        self.metric_set.reset()

    @staticmethod
    def get_distance_to_other_agents(
        ego_centroid,
        ego_yaw,
        ego_extent,
        agent_centroid,
        agent_yaw,
        agent_extent,
    ):
        if type(ego_centroid) is th.Tensor:
            ego_bbox = utils._get_bounding_box(ego_centroid.cpu().numpy(), ego_yaw.cpu().numpy(), ego_extent.cpu().numpy())
            agent_bbox = utils._get_bounding_box(agent_centroid.cpu().numpy(), agent_yaw.cpu().numpy(),
                                                agent_extent.cpu().numpy())
        else:
            ego_bbox = utils._get_bounding_box(ego_centroid, ego_yaw, ego_extent)
            agent_bbox = utils._get_bounding_box(agent_centroid, agent_yaw, agent_extent)
        distance = ego_bbox.distance(agent_bbox)
        return distance
    
    @staticmethod
    def get_distance_to_centroid(
        current_centroid,
        ref_lanes,
        consider_avail=False,
    ):
        if type(current_centroid) is th.Tensor:
            distance = distance_to_reference_trajectory(current_centroid, ref_lanes)
        else:
            current_centroid = th.tensor(np.array(current_centroid))
            ref_lanes = th.tensor(np.array(ref_lanes))
            distance = distance_to_reference_trajectory(current_centroid, ref_lanes)
        return distance.cpu().numpy()[0]

    def get_reward(self, frame_ego: List[Dict[str, np.ndarray]], frame_agents: List[Dict[str, np.ndarray]], lanes_mid) -> Dict[str, float]:
        """Get the reward for the given step in close loop training.

        :param frame_ego: all ego info in current frame of simulation
        :param frame_agents: all agents info in current frame of simulation
        :return: the dictionary containing total reward and individual components that make up the reward
        """
        # compute the collision avoid reward
        dist_car_list = []
        ego_centroid = frame_ego[0]['centroid']
        ego_yaw = frame_ego[0]['yaw']
        ego_extent = frame_ego[0]['extent']
        if len(frame_agents) > 0:
            for agent_info in frame_agents:
                agent_centroid = agent_info['centroid']
                agent_yaw = agent_info['yaw']
                agent_extent = agent_info['extent']
                dist = self.get_distance_to_other_agents(ego_centroid, ego_yaw, ego_extent,
                                                    agent_centroid, agent_yaw, agent_extent)
                dist_car_list.append(dist)
        else:
            dist_car_list.append(100.0)
        min_dist_car = min(dist_car_list)
        col_reward = min(min_dist_car - 2, 0)  # -2~0

        # # compute the off-road avoid reward
        
        # dist_lane_list = []
        # for mid_lane in lanes_mid:
        #     dist_lane_list.append(
        #         self.get_distance_to_centroid([frame_ego[0]['history_positions'][0]], [mid_lane])
        #         )
        # min_dist_lane = min(dist_lane_list)
        # off_reward = min(-(min_dist_lane - 1), 0)  # -inf~0

        # # clip the off-road reward -2~0
        # if self.enable_clip:
        #     off_reward = max(-self.rew_clip_thresh , min(self.rew_clip_thresh, off_reward))

        # # Total reward
        # total_reward = col_reward + off_reward

        off_reward = 0.0
        total_reward = col_reward + off_reward

        reward_dict = {"total": total_reward, "collision": col_reward, "off-road": off_reward}
        return reward_dict
