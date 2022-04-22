from typing import Dict
from typing import List

import torch
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
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
        pass
