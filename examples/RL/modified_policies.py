import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch import nn

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule

from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class MultiInputActorCriticPredPolicy(MultiInputActorCriticPolicy):
    def __init__(
    self,
    observation_space: spaces.Dict,
    action_space: spaces.Space,
    lr_schedule: Schedule,
    net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
    activation_fn: Type[nn.Module] = nn.Tanh,
    ortho_init: bool = True,
    use_sde: bool = False,
    log_std_init: float = 0.0,
    full_std: bool = True,
    use_expln: bool = False,
    squash_output: bool = False,
    features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
    features_extractor_kwargs: Optional[Dict[str, Any]] = None,
    share_features_extractor: bool = True,
    normalize_images: bool = True,
    optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.pred_net = nn.Linear(self.action_net.in_features, out_features=self.action_net.out_features)  # just default settings
        # self.vf_features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:  # False
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob
    
    def pred_traj(self, obs: th.Tensor) -> th.Tensor:
        """
        Forward pass in trajectory prediction network.

        :param obs: Observation
        :return: predicted trajectories
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        traj = self.pred_net(latent_pi)
        return traj
    
    def pred_traj1(self, obs: th.Tensor) -> th.Tensor:
        """
        Forward pass in trajectory prediction network.

        :param obs: Observation
        :return: predicted trajectories
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        
        traj = self.pred_net(features)
        return traj
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1,) + self.action_space.shape)

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)
    

