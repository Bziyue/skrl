"""Custom DeFM + ETH-SRU models for SRU-based training.

Policy:
    DeFM(P4) -> image cross-attention -> SRU -> policy head

Value:
    DeFM(P4) -> image cross-attention
             + height projection -> height cross-attention
             -> SRU -> concat time embedding -> value head
"""

from __future__ import annotations

from typing import Literal

import gymnasium
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .custom_models_with_defm import DeFMBackbone, preprocess_depth_independent
from .eth_sru_modules import CrossAttentionFuseModule, LSTM_SRU


def _parse_image_shape(image_shape: tuple[int, ...], model_name: str) -> tuple[int, int, int]:
    if len(image_shape) != 3:
        raise ValueError(f"Unsupported image shape for {model_name}: {image_shape}")
    if image_shape[0] != 4:
        raise ValueError(f"{model_name} expects image shape (4,H,W), got {image_shape}")
    return 4, image_shape[1], image_shape[2]


def _prepare_height_tensor(height: torch.Tensor, model_name: str) -> torch.Tensor:
    if height.ndim == 3:
        return height.unsqueeze(1)
    if height.ndim == 4:
        return height
    raise ValueError(f"{model_name} expects height as (B,H,W) or (B,C,H,W), got {tuple(height.shape)}")


def _space_keys(space_obj) -> set[str]:
    if isinstance(space_obj, spaces.Dict):
        return set(space_obj.spaces.keys())
    if isinstance(space_obj, dict):
        return set(space_obj.keys())
    return set()


def _space_get(space_obj, key: str):
    if isinstance(space_obj, spaces.Dict):
        return space_obj.spaces[key]
    return space_obj[key]


def _unwrap_critic_space(state_space_obj):
    """Unwrap possible {'critic': {...}} layout into critic inner space."""
    keys = _space_keys(state_space_obj)
    if "critic" in keys and "image" not in keys:
        return _space_get(state_space_obj, "critic")
    return state_space_obj


class _SRUMemoryCore(nn.Module):
    """SRU memory wrapper with strict skrl PPO_RNN state handling."""

    def __init__(
        self,
        *,
        input_size: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        sequence_length: int,
        num_envs: int,
    ) -> None:
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length
        self.num_envs = num_envs

        self.memory = LSTM_SRU(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )

    def _init_state(self, batch_size: int, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.memory.init_state(batch_size=batch_size, device=reference.device, dtype=reference.dtype)

    def _prepare_rnn_states(
        self,
        inputs: dict,
        target_batch: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rnn_states = inputs.get("rnn", None)
        # Runner.init_state_dict() doesn't provide recurrent states; bootstrap with zeros.
        if rnn_states is None:
            return self._init_state(target_batch, reference)
        if not isinstance(rnn_states, list) or len(rnn_states) != 2:
            raise ValueError("inputs['rnn'] must be [h, c] for SRU models")

        h, c = rnn_states
        h = h.to(device=reference.device, dtype=reference.dtype)
        c = c.to(device=reference.device, dtype=reference.dtype)

        if h.ndim != 3 or c.ndim != 3:
            raise ValueError(f"RNN states must be 3D, got h={h.shape}, c={c.shape}")
        if h.shape[0] != self.rnn_num_layers or c.shape[0] != self.rnn_num_layers:
            raise ValueError(
                f"RNN layer mismatch: expected {self.rnn_num_layers}, got h={h.shape[0]}, c={c.shape[0]}"
            )
        if h.shape[2] != self.rnn_hidden_size or c.shape[2] != self.rnn_hidden_size:
            raise ValueError(
                f"RNN hidden size mismatch: expected {self.rnn_hidden_size}, got h={h.shape[2]}, c={c.shape[2]}"
            )

        if self.training:
            expected_batch = target_batch * self.sequence_length
            if h.shape[1] != expected_batch or c.shape[1] != expected_batch:
                raise ValueError(
                    f"Training expects RNN batch={expected_batch}, got h={h.shape[1]}, c={c.shape[1]}"
                )
            h = h.view(self.rnn_num_layers, target_batch, self.sequence_length, self.rnn_hidden_size)[:, :, 0, :]
            c = c.view(self.rnn_num_layers, target_batch, self.sequence_length, self.rnn_hidden_size)[:, :, 0, :]
            return h.contiguous(), c.contiguous()

        if h.shape[1] != target_batch or c.shape[1] != target_batch:
            raise ValueError(
                f"Rollout expects RNN batch={target_batch}, got h={h.shape[1]}, c={c.shape[1]}"
            )
        return h, c

    def _extract_done_mask(self, inputs: dict, batch_size: int) -> torch.Tensor | None:
        terminated = inputs.get("terminated", None)
        truncated = inputs.get("truncated", None)
        if terminated is None or truncated is None:
            return None
        done = torch.logical_or(terminated.bool().view(-1), truncated.bool().view(-1))
        if done.numel() != batch_size * self.sequence_length:
            return None
        return done.view(batch_size, self.sequence_length)

    def run_memory(self, features: torch.Tensor, inputs: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
        flat_batch = features.shape[0]
        if self.training:
            if flat_batch % self.sequence_length != 0:
                raise ValueError(
                    f"Batch size {flat_batch} is not divisible by sequence_length {self.sequence_length}"
                )
            batch_size = flat_batch // self.sequence_length
            rnn_input = features.view(batch_size, self.sequence_length, -1)
        else:
            batch_size = flat_batch
            rnn_input = features.view(batch_size, 1, -1)

        h, c = self._prepare_rnn_states(inputs=inputs, target_batch=batch_size, reference=features)

        if self.training:
            done = self._extract_done_mask(inputs=inputs, batch_size=batch_size)
            if done is not None and torch.any(done):
                outputs = []
                indexes = [0] + (done[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [
                    self.sequence_length
                ]
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    chunk_out, (h, c) = self.memory(rnn_input[:, i0:i1, :], state=(h, c))
                    done_at_end = done[:, i1 - 1]
                    if torch.any(done_at_end):
                        h[:, done_at_end, :] = 0
                        c[:, done_at_end, :] = 0
                    outputs.append(chunk_out)
                rnn_output = torch.cat(outputs, dim=1)
            else:
                rnn_output, (h, c) = self.memory(rnn_input, state=(h, c))
        else:
            rnn_output, (h, c) = self.memory(rnn_input, state=(h, c))

        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        return rnn_output, [h, c]

    def get_rnn_specification(self) -> dict:
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.rnn_num_layers, self.num_envs, self.rnn_hidden_size),
                    (self.rnn_num_layers, self.num_envs, self.rnn_hidden_size),
                ],
            }
        }


class DeFMSRUPolicy(GaussianMixin, Model):
    """Gaussian policy using DeFM backbone + cross-attention + LSTM_SRU."""

    def __init__(
        self,
        *,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        clip_actions: bool = False,
        clip_mean_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: Literal["mean", "sum", "prod", "none"] = "sum",
        role: str = "",
        initial_log_std: float = 0.0,
        fixed_log_std: bool = False,
        defm_model_name: str = "defm_resnet18",
        defm_pretrained_path: str | None = None,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        attention_heads: int = 4,
        sequence_length: int = 1,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role=role,
        )

        image_shape = observation_space["image"].shape
        state_dim = observation_space["state"].shape[0]
        action_dim = action_space.shape[0]
        num_cameras, img_height, img_width = _parse_image_shape(image_shape, "DeFMSRUPolicy")

        self.num_cameras = num_cameras
        self.img_height = img_height
        self.img_width = img_width

        self.defm_backbone = DeFMBackbone(
            model_name=defm_model_name,
            pretrained_path=defm_pretrained_path,
            device=device,
            freeze=True,
        )
        self.image_attention = CrossAttentionFuseModule(
            image_dim=self.defm_backbone.p4_channels,
            info_dim=state_dim,
            num_heads=attention_heads,
            spatial_dims=(num_cameras, self.defm_backbone.p4_h, self.defm_backbone.p4_w),
        )
        self.memory_core = _SRUMemoryCore(
            input_size=self.defm_backbone.p4_channels + state_dim,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            sequence_length=sequence_length,
            num_envs=num_envs,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(rnn_hidden_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

        self.log_std_parameter = nn.Parameter(
            torch.full(size=action_space.shape, fill_value=float(initial_log_std), dtype=torch.float32),
            requires_grad=not fixed_log_std,
        )

    def get_specification(self) -> dict:
        return self.memory_core.get_rnn_specification()

    def compute(self, inputs, role=""):
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        raw_depth = observations["image"]
        state = observations["state"]
        n_batch = raw_depth.shape[0]

        with torch.no_grad():
            processed_depth = preprocess_depth_independent(
                raw_depth,
                target_size=(self.img_height, self.img_width),
                device=raw_depth.device,
            )
            p4_features = self.defm_backbone(processed_depth)

        p4_features = p4_features.view(
            n_batch,
            self.num_cameras,
            self.defm_backbone.p4_channels,
            self.defm_backbone.p4_h,
            self.defm_backbone.p4_w,
        )
        p4_features = p4_features.permute(0, 2, 1, 3, 4).contiguous()
        image_features = self.image_attention(img=p4_features, info=state)
        memory_input = torch.cat([image_features, state], dim=-1)

        memory_output, new_states = self.memory_core.run_memory(features=memory_input, inputs=inputs)
        action_mean = self.policy_head(memory_output)
        return action_mean, {"log_std": self.log_std_parameter, "rnn": new_states}


class DeFMSRUValue(DeterministicMixin, Model):
    """Deterministic value model with image/height/time critic flow."""

    def __init__(
        self,
        *,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        clip_actions: bool = False,
        role: str = "",
        defm_model_name: str = "defm_resnet18",
        defm_pretrained_path: str | None = None,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        attention_heads: int = 4,
        time_embed_dim: int = 8,
        sequence_length: int = 1,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)

        self._critic_state_space = _unwrap_critic_space(state_space)
        critic_space_keys = _space_keys(self._critic_state_space)
        for key in ("image", "state", "height", "time"):
            if key not in critic_space_keys:
                raise ValueError(
                    f"DeFMSRUValue requires critic state keys image/state/height/time, got keys={sorted(critic_space_keys)}"
                )

        image_shape = _space_get(self._critic_state_space, "image").shape
        state_dim = _space_get(self._critic_state_space, "state").shape[0]
        height_shape = _space_get(self._critic_state_space, "height").shape
        time_dim = _space_get(self._critic_state_space, "time").shape[0]
        num_cameras, img_height, img_width = _parse_image_shape(image_shape, "DeFMSRUValue")
        if len(height_shape) == 2:
            height_channels = 1
        elif len(height_shape) == 3:
            height_channels = height_shape[0]
        else:
            raise ValueError(f"DeFMSRUValue expects height shape (H,W) or (C,H,W), got {height_shape}")

        self.num_cameras = num_cameras
        self.img_height = img_height
        self.img_width = img_width

        self.defm_backbone = DeFMBackbone(
            model_name=defm_model_name,
            pretrained_path=defm_pretrained_path,
            device=device,
            freeze=True,
        )
        self.image_attention = CrossAttentionFuseModule(
            image_dim=self.defm_backbone.p4_channels,
            info_dim=state_dim,
            num_heads=attention_heads,
            spatial_dims=(num_cameras, self.defm_backbone.p4_h, self.defm_backbone.p4_w),
        )
        self.height_proj = nn.Sequential(
            nn.Conv2d(height_channels, self.defm_backbone.p4_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
        )
        self.height_attention = CrossAttentionFuseModule(
            image_dim=self.defm_backbone.p4_channels,
            info_dim=state_dim,
            num_heads=attention_heads,
            spatial_dims=(1, self.defm_backbone.p4_h, self.defm_backbone.p4_w),
        )
        self.memory_core = _SRUMemoryCore(
            input_size=self.defm_backbone.p4_channels * 2 + state_dim,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            sequence_length=sequence_length,
            num_envs=num_envs,
        )
        self.time_layer = nn.Linear(time_dim, time_embed_dim)

        self.value_head = nn.Sequential(
            nn.Linear(rnn_hidden_size + time_embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def get_specification(self) -> dict:
        return self.memory_core.get_rnn_specification()

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        if isinstance(states, dict) and "critic" in states:
            states = states["critic"]
        raw_depth = states["image"]
        proprio = states["state"]
        height = states["height"]
        time_obs = states["time"].view(states["time"].shape[0], -1)
        n_batch = raw_depth.shape[0]

        with torch.no_grad():
            processed_depth = preprocess_depth_independent(
                raw_depth,
                target_size=(self.img_height, self.img_width),
                device=raw_depth.device,
            )
            p4_features = self.defm_backbone(processed_depth)

        p4_features = p4_features.view(
            n_batch,
            self.num_cameras,
            self.defm_backbone.p4_channels,
            self.defm_backbone.p4_h,
            self.defm_backbone.p4_w,
        )
        p4_features = p4_features.permute(0, 2, 1, 3, 4).contiguous()
        image_features = self.image_attention(img=p4_features, info=proprio)

        height = _prepare_height_tensor(height, "DeFMSRUValue")
        height_features = self.height_proj(height)
        height_features = F.interpolate(
            height_features,
            size=(self.defm_backbone.p4_h, self.defm_backbone.p4_w),
            mode="bilinear",
            align_corners=False,
        )
        height_features = self.height_attention(img=height_features, info=proprio)

        memory_input = torch.cat([image_features, height_features, proprio], dim=-1)
        memory_output, new_states = self.memory_core.run_memory(features=memory_input, inputs=inputs)

        time_embed = self.time_layer(time_obs)
        value_input = torch.cat([memory_output, time_embed], dim=-1)
        value = self.value_head(value_input)
        return value, {"rnn": new_states}
