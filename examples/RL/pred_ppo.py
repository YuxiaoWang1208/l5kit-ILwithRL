import os
import warnings
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

import sys
sys.path.append("/root/zhufenghua12/wangyuxiao/l5kit-wyx/examples/RL")
from modified_policies import MultiInputActorCriticPredPolicy

from get_il_data import get_data


SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PRED_PPO(PPO):
    """
    Imitation Learning(IL) and Proximal Policy Optimization algorithm (PPO) (clip version)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MultiInputPredPolicy": MultiInputActorCriticPredPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        weights_scaling: List[float] = [1., 1., 1.],
        future_num_frames: Optional[int] = 12,
        criterion: nn.Module = nn.MSELoss(reduction="none"),
        # dmg: Optional[LocalDataManager] = None,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_IL_epochs: int = 50,
        n_RL_epochs: int = 50,
        n_epochs: int = 50,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        il_coef: float = 1.0,  # 0.0 1.0
        ent_coef: float = 0.0,  # 0.0 0.01
        vf_coef: float = 0.5,  # 0.5 0.1
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # n_epochs = n_IL_epochs + n_RL_epochs

        self.future_num_frames = future_num_frames

        assert self.future_num_frames > 1, "Future num frames should be more than 1 in prediction networks!"
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )

        # # Required to register environment
        # if env_config_path is None:
        #     return

        self.weights_scaling = th.tensor(weights_scaling).to(self.device)
        self.criterion = criterion
        self.il_coef = il_coef
        self.n_IL_epochs = n_IL_epochs
        self.n_RL_epochs = n_RL_epochs
        self.n_epochs = n_epochs

        self.IL_losses_mean = float(0)
        self.RL_losses_mean = float(0)
        self.V_losses_mean = float(0)

        # num_targets = self.action_space.shape[0] * (self.future_num_frames - 1)
        # self.policy.pred_net = nn.Linear(self.policy.action_net.in_features, out_features=num_targets).to(self.policy.device)
        # print(self.policy.pred_net)
        
    def _setup_model(self) -> None:
        super()._setup_model()
        num_targets = self.action_space.shape[0] * (self.future_num_frames - 1)
        # num_targets = self.action_space.shape[0] * self.future_num_frames
        self.policy.pred_net = nn.Linear(self.policy.action_net.in_features, out_features=num_targets).to(self.policy.device)
        # self.policy.pred_net = nn.Linear(self.policy.features_extractor._features_dim, out_features=num_targets).to(self.policy.device)

        self.policy.optimizer = self.policy.optimizer_class(self.policy.parameters(), lr=self.lr_schedule(1), **self.policy.optimizer_kwargs)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        IL_losses, RL_losses, pg_losses, value_losses = [], [], [], []
        clip_fractions = []

        continue_training = True

        # ==== train for n_epochs epochs ====
        for epoch in range(self.n_epochs):

            '''# ==== one RL train epoch ====
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                self.V_losses_mean = (self.V_losses_mean * self._n_updates + value_loss.item()) / (self._n_updates+1)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                RL_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss  # PPO
                # RL_loss = self.ent_coef * entropy_loss + self.vf_coef * value_loss  # no policy PPO
                # RL_loss = self.il_coef * il_loss + policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss  # ILPPO
                # RL_loss = self.il_coef * il_loss + 0.0*(policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss)  # IL
                RL_losses.append(RL_loss.item())
                self.RL_losses_mean = (self.RL_losses_mean * self._n_updates + RL_loss.item()) / (self._n_updates+1)
                V_loss = self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # if epoch >= self.n_RL_epochs:
                #     # Optimize only Value net
                #     self.policy.optimizer.zero_grad()
                #     V_loss.backward()
                #     # Clip grad norm
                #     th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                #     self.policy.optimizer.step()
                #     continue

                # Optimization step
                self.policy.optimizer.zero_grad()
                RL_loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            


            # ==== one Pred IL train epoch ====
            for _ in range(int(self.rollout_buffer.buffer_size/self.batch_size)):  # 1 int(self.rollout_buffer.buffer_size/self.batch_size)
                # ===== Imitation learning data for iteration =====
                il_data_buffer = get_data()
                # get dataset observations
                il_obs = {'image': il_data_buffer["image"]}
                # get action_net actions distributions
                il_actions_dist = self.policy.get_distribution(il_obs)
                # get action_net actions
                # il_actions = il_actions_dist.mode()
                il_actions = il_actions_dist.get_actions(deterministic=True)
                pred_actions = self.policy.pred_traj(il_obs)
                traj_actions = th.cat((il_actions, pred_actions), dim=-1)
                # pred_actions = self.policy.pred_traj1(il_obs)
                # traj_actions = pred_actions

                # traj_actions = traj_actions.view(pred_actions.shape[0], self.future_num_frames, -1)

                # get target actions and compute imitation loss
                xy = il_data_buffer["target_positions"]
                yaw = il_data_buffer["target_yaws"]
                # if self.normalize_targets:
                #     xy /= self.xy_scale
                target_actions = th.cat((xy, yaw), dim=-1).view(il_actions.shape[0], -1)
                # get resacle params and rescale the targets
                rescale_action = self.env.get_attr('rescale_action')[0]
                use_kinematic = self.env.get_attr('use_kinematic')[0]
                if rescale_action:
                    if use_kinematic:
                        kin_rescale = self.env.get_attr('kin_rescale')[0]
                        target_actions[..., 0] = target_actions[..., 0] / kin_rescale.steer_scale
                        target_actions[..., 1] = target_actions[..., 1] / kin_rescale.acc_scale
                    else:
                        non_kin_rescale = self.env.get_attr('non_kin_rescale')[0]
                        target_actions[..., 0] = (target_actions[..., 0] - non_kin_rescale.x_mu) / non_kin_rescale.x_scale
                        target_actions[..., 1] = (target_actions[..., 1] - non_kin_rescale.y_mu) / non_kin_rescale.y_scale
                        target_actions[..., 2] = (target_actions[..., 2] - non_kin_rescale.yaw_mu) / non_kin_rescale.yaw_scale
                # target_weights = data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling
                # t = self.criterion(il_actions, target_actions.squeeze(1))
                # [batch_size, num_steps]
                target_weights = (il_data_buffer["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                    target_actions.shape[0], -1
                    )
                pred_loss = th.mean(self.criterion(traj_actions, target_actions) * target_weights)

                IL_loss = self.il_coef * pred_loss  # IL
                IL_losses.append(IL_loss.item())
                self.IL_losses_mean = (self.IL_losses_mean * self._n_updates + IL_loss.item()) / (self._n_updates+1)

                # Optimization step
                self.policy.optimizer.zero_grad()
                IL_loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            '''


            # ==== one training epoch ====
            approx_kl_divs = []
            # if not epoch >= self.n_RL_epochs:  # il and rl
            rollout_datas = list(self.rollout_buffer.get(self.batch_size))
            # Do a complete pass on the rollout buffer
            # for rollout_data in self.rollout_buffer.get(self.batch_size):  # SLOW!!!
            for k in range(int(self.rollout_buffer.buffer_size/self.batch_size)):
                # ===== Imitation learning data for iteration =====
                il_data_buffer = get_data()
                # get dataset observations
                il_obs = {'image': il_data_buffer["image"]}
                # get action_net actions distributions
                il_actions_dist = self.policy.get_distribution(il_obs)
                # get action_net actions
                # il_actions = il_actions_dist.mode()
                il_actions = il_actions_dist.get_actions(deterministic=True)
                pred_actions = self.policy.pred_traj(il_obs)
                traj_actions = th.cat((il_actions, pred_actions), dim=-1)
                # pred_actions = self.policy.pred_traj1(il_obs)
                # traj_actions = pred_actions

                # get target actions and compute imitation loss
                xy = il_data_buffer["target_positions"]
                yaw = il_data_buffer["target_yaws"]
                # if self.normalize_targets:
                #     xy /= self.xy_scale
                target_actions = th.cat((xy, yaw), dim=-1).view(il_actions.shape[0], -1)
                # get resacle params and rescale the targets
                rescale_action = self.env.get_attr('rescale_action')[0]
                use_kinematic = self.env.get_attr('use_kinematic')[0]
                if rescale_action:
                    if use_kinematic:
                        kin_rescale = self.env.get_attr('kin_rescale')[0]
                        target_actions[..., 0] = target_actions[..., 0] / kin_rescale.steer_scale
                        target_actions[..., 1] = target_actions[..., 1] / kin_rescale.acc_scale
                    else:
                        non_kin_rescale = self.env.get_attr('non_kin_rescale')[0]
                        target_actions[..., 0] = (target_actions[..., 0] - non_kin_rescale.x_mu) / non_kin_rescale.x_scale
                        target_actions[..., 1] = (target_actions[..., 1] - non_kin_rescale.y_mu) / non_kin_rescale.y_scale
                        target_actions[..., 2] = (target_actions[..., 2] - non_kin_rescale.yaw_mu) / non_kin_rescale.yaw_scale
                # target_weights = data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling
                # t = self.criterion(il_actions, target_actions.squeeze(1))
                # [batch_size, num_steps]
                target_weights = (il_data_buffer["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                    target_actions.shape[0], -1
                    )
                pred_loss = th.mean(self.criterion(traj_actions, target_actions) * target_weights)
                IL_loss = self.il_coef * pred_loss  # IL prediction
                IL_losses.append(IL_loss.item())
                self.IL_losses_mean = (self.IL_losses_mean * self._n_updates + IL_loss.item()) / (self._n_updates+1)

                # if epoch >= self.n_RL_epochs:  # only il
                #     # Optimization step
                #     self.policy.optimizer.zero_grad()
                #     IL_loss.backward()
                #     # Clip grad norm
                #     th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                #     self.policy.optimizer.step()
                #     if not continue_training:
                #         break        
                #     continue


                rollout_data = rollout_datas[k]
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                self.V_losses_mean = (self.V_losses_mean * self._n_updates + value_loss.item()) / (self._n_updates+1)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                RL_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss  # PPO
                # RL_loss = self.ent_coef * entropy_loss + self.vf_coef * value_loss  # no policy PPO
                # RL_loss = self.il_coef * il_loss + policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss  # ILPPO
                # RL_loss = self.il_coef * il_loss + 0.0*(policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss)  # IL
                RL_losses.append(RL_loss.item())
                self.RL_losses_mean = (self.RL_losses_mean * self._n_updates + RL_loss.item()) / (self._n_updates+1)
                V_loss = self.vf_coef * value_loss

                total_loss = IL_loss + 0.1 * RL_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # if epoch >= self.n_RL_epochs:
                #     # Optimize only Value net
                #     self.policy.optimizer.zero_grad()
                #     V_loss.backward()
                #     # Clip grad norm
                #     th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                #     self.policy.optimizer.step()
                #     continue

                # Optimization step
                self.policy.optimizer.zero_grad()
                # il + rl
                total_loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            
                
            if not continue_training:
                break        
            
            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
            # Logs
            # self.logger.record("train/imitation_loss", np.mean(il_losses))
            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", self.V_losses_mean) # np.mean(value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/IL_loss", IL_loss.item())
            self.logger.record("train/IL_loss_mean", self.IL_losses_mean) # np.mean(IL_losses))
            self.logger.record("train/RL_loss", RL_loss.item())
            self.logger.record("train/RL_loss_mean", self.RL_losses_mean) # np.mean(RL_losses))

            self.logger.record("train/explained_variance", explained_var)
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train/n_updates", self._n_updates)
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

            # Logs
            # self.logger.record("train/imitation_loss", np.mean(il_losses))
            # self.logger.record("train/loss", loss.item())
            # self.logger.record("train/n_updates", self._n_updates)
            
            self.logger.dump(step=self._n_updates)
        

            # ==== save the model ====
            # from pred_ppo_training import MODEL_PATH
            if self._n_updates % 5000 == 0:
                model_path = f"{os.getcwd()}/models_" + os.environ["DATE"] + f"/{self._n_updates}.pt"  # pt zip
                self.save(model_path)

            self._n_updates += 1


    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
