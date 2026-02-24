"""
Custom models with DeFM (Deep Feature Model) backbone for depth-based RL.

This module provides Policy and Value networks that use DeFM's pretrained ResNet18-BiFPN
backbone for processing depth images. The backbone is frozen and P4 features are extracted.

Input:
    - 4 x 64x32 depth images (processed independently, then fused at feature level)
    - State vector

Output:
    - Policy: Action mean + log_std
    - Value: Scalar value estimation
"""

from __future__ import annotations

from typing import Literal
import pathlib

import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

# DeFM imports
from defm.model_factory import create_defm_model
from defm.utils.utils import preprocess_depth_independent as defm_preprocess_depth_independent


# Get path to DeFM weights directory
DEFM_WEIGHTS_DIR = pathlib.Path(__file__).resolve().parents[6] / "defm" / "weights"


class DeFMBackbone(nn.Module):
    """
    Wrapper around DeFM ResNet18-BiFPN backbone.
    
    - Loads pretrained weights and freezes all parameters
    - Extracts P4 features (stride 16) from BiFPN output
    - Preprocesses raw depth images using DeFM's metric-aware normalization
    
    Input: single-camera DeFM input (N, 3, 64, 32) -> P4 (4x2)
    """
    
    def __init__(
        self,
        model_name: str = "defm_resnet18",
        pretrained_path: str | None = None,
        device: str | torch.device | None = None,
        freeze: bool = True,
    ):
        super().__init__()
        
        self.device = device if device else "cpu"
        
        # Determine weight path
        if pretrained_path is None:
            pretrained_path = str(DEFM_WEIGHTS_DIR / f"{model_name}.pth")
        
        print(f"\n[DeFM Backbone] Loading model: {model_name}")
        print(f"[DeFM Backbone] Weights path: {pretrained_path}")
        
        # Create and load pretrained model
        self.backbone = create_defm_model(
            model_name=model_name,
            pretrained=True,
            pretrained_path=pretrained_path,
        )
        
        # Freeze parameters if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            print("[DeFM Backbone] Parameters frozen")
        
        # Move to device
        self.backbone = self.backbone.to(self.device)
        
        # Compute output dimension using dummy input (single camera resolution)
        dummy_input = torch.zeros(1, 3, 64, 32).to(self.device)
        
        with torch.no_grad():
            out_dict = self.backbone(dummy_input)
            p4_feat = out_dict["dense_bifpn"]["P4"]
            # Store spatial shape info: (channels, H, W)
            self.p4_channels = p4_feat.shape[1]   # 128
            self.p4_h = p4_feat.shape[2]           # 4 (64/16)
            self.p4_w = p4_feat.shape[3]           # 2 (32/16)
            
            print(f"[DeFM Backbone] P4 Feature Shape: {p4_feat.shape}")
            print(f"[DeFM Backbone] P4 Channels: {self.p4_channels}, Spatial: {self.p4_h}x{self.p4_w}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeFM backbone.
        
        Args:
            x: Preprocessed depth tensor of shape (N, 3, 64, 32)

        Returns:
            P4 features of shape (N, 128, 4, 2) - preserving spatial dimensions
        """
        with torch.no_grad():
            out_dict = self.backbone(x)
            return out_dict["dense_bifpn"]["P4"]


def _ensure_multicam_layout(
    depth_images: torch.Tensor,
    num_cameras: int = 4,
) -> torch.Tensor:
    """
    Ensure depth tensor has shape (N, C, H, W), accepting:
      - (N, C, H, W)
      - (N, 1, H, C*W) legacy stitched layout
    """
    if depth_images.ndim != 4:
        raise ValueError(f"depth_images must be 4D, got shape {tuple(depth_images.shape)}")

    if depth_images.shape[1] == num_cameras:
        return depth_images

    # Legacy layout: (N, 1, H, C*W) -> (N, C, H, W)
    if depth_images.shape[1] == 1 and depth_images.shape[3] % num_cameras == 0:
        n_batch, _, height, width_total = depth_images.shape
        cam_width = width_total // num_cameras
        return depth_images.reshape(n_batch, num_cameras, height, cam_width)

    raise ValueError(
        f"Unsupported depth layout {tuple(depth_images.shape)}. "
        f"Expected (N,{num_cameras},H,W) or (N,1,H,{num_cameras}*W)."
    )


def preprocess_depth_independent(
    depth_images: torch.Tensor,
    target_size: tuple[int, int] = (64, 32),
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Preprocess multi-camera depth images independently to avoid pixel-level stitching artifacts.
    """
    depth_images = _ensure_multicam_layout(depth_images, num_cameras=4)
    processed = defm_preprocess_depth_independent(
        depth_images=depth_images,
        target_size=target_size,
        device=device,
    )
    return processed  # (N*4, 3, H, W)


class DeFMPolicy(GaussianMixin, Model):
    """
    Policy network using DeFM ResNet18-BiFPN backbone.
    
    Architecture:
        1. DeFM backbone (frozen) extracts single-view P4 features
        2. Per-camera independent feature compression (no cross-camera spatial mixing)
        3. MLP combines image features + state -> action mean
        4. Learnable log_std parameter for Gaussian policy
    """
    
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
        initial_log_std: float = 0,
        fixed_log_std: bool = False,
        defm_model_name: str = "defm_resnet18",
        defm_pretrained_path: str | None = None,
        **kwargs
    ):
        # Initialize base classes
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
        
        # Print configuration
        print("\n")
        print("======================= DeFMPolicy =======================")
        print("###### [Model Initialization]")
        print("observation_space:", observation_space)
        print("state_space:", state_space)
        print("action_space:", action_space)
        print("device:", device)
        print("\n")
        print("###### [GaussianMixin Initialization]")
        print("clip_actions:", clip_actions)
        print("clip_mean_actions:", clip_mean_actions)
        print("clip_log_std:", clip_log_std)
        print("min_log_std:", min_log_std)
        print("max_log_std:", max_log_std)
        print("reduction:", reduction)
        print("role:", role)
        print("\n")
        print("###### [DeFM Configuration]")
        print("defm_model_name:", defm_model_name)
        print("defm_pretrained_path:", defm_pretrained_path)
        print("initial_log_std:", initial_log_std)
        print("fixed_log_std:", fixed_log_std)
        print("==========================================================\n")
        
        # ------------------- Network Definition -------------------
        
        # 1. Parse input space dimensions
        # observation_space is a Dict with "image" and "state"
        # image shape: (4, 64, 32) -> 4 cameras, 64x32 each
        image_shape = observation_space["image"].shape
        state_shape = observation_space["state"].shape
        action_shape = action_space.shape
        
        if len(image_shape) != 3:
            raise ValueError(f"Unsupported image shape: {image_shape}")

        # Preferred layout: (4, H, W). Also accept legacy stitched layout: (1, H, 4W).
        if image_shape[0] == 4:
            self.num_cameras = 4
            self.img_height = image_shape[1]
            self.img_width = image_shape[2]
        elif image_shape[0] == 1 and image_shape[2] % 4 == 0:
            self.num_cameras = 4
            self.img_height = image_shape[1]
            self.img_width = image_shape[2] // 4
        else:
            raise ValueError(
                f"DeFMPolicy expects image shape (4,H,W) or (1,H,4W), got {image_shape}"
            )
        state_dim = state_shape[0]
        action_dim = action_shape[0]
        
        print(f"[DeFM Policy] Image shape: {image_shape} (cameras, H, W)")
        print(f"[DeFM Policy] State dim: {state_dim}")
        print(f"[DeFM Policy] Action dim: {action_dim}")
        
        # 2. Initialize DeFM backbone (frozen)
        self.defm_backbone = DeFMBackbone(
            model_name=defm_model_name,
            pretrained_path=defm_pretrained_path,
            device=device,
            freeze=True,
        )
        
        # 3. Per-camera feature compression (independent across cameras)
        # (N*4, 128, 4, 2) -> (N*4, 32, 4, 2) -> flatten
        self.feature_compression = nn.Sequential(
            nn.Conv2d(
                in_channels=self.defm_backbone.p4_channels,
                out_channels=32,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        compressed_feat_dim = (
            self.num_cameras * 32 * self.defm_backbone.p4_h * self.defm_backbone.p4_w
        )
        print(f"[DeFM Policy] Compressed feature dim: {compressed_feat_dim}")

        # 4. Define MLP (compressed image features + state -> action)
        self.mlp = nn.Sequential(
            nn.Linear(compressed_feat_dim + state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 5. Learnable log_std parameter
        self.log_std_parameter = nn.Parameter(
            torch.full(size=action_space.shape, fill_value=float(initial_log_std), dtype=torch.float32),
            requires_grad=not fixed_log_std
        )
        
        print(f"[DeFM Policy] MLP input dim: {compressed_feat_dim + state_dim}")
        print(f"[DeFM Policy] MLP output dim: {action_dim}")
    
    def compute(self, inputs, role=""):
        # Unflatten observations
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        
        # Get raw depth images and state
        raw_depth = observations["image"]   # expected: (N, 4, 64, 32)
        state = observations["state"]       # (N, state_dim)
        n_batch = raw_depth.shape[0]

        # Backbone forward under no_grad: frozen params don't need gradients,
        # and skipping activation storage saves significant GPU memory.
        with torch.no_grad():
            processed_depth = preprocess_depth_independent(
                raw_depth,
                target_size=(self.img_height, self.img_width),
                device=raw_depth.device,
            )
            p4_features = self.defm_backbone(processed_depth)  # (N*4, 128, 4, 2)

        # 1) Compress each camera feature independently: (N*4, 128, 4, 2) -> (N*4, 256)
        compressed_features = self.feature_compression(p4_features)
        # 2) Restore camera dimension: (N*4, 256) -> (N, 4, 256)
        compressed_features = compressed_features.reshape(n_batch, self.num_cameras, -1)
        # 3) Flatten 4 cameras into one vector: (N, 4, 256) -> (N, 1024)
        img_features = compressed_features.reshape(n_batch, -1)
        
        # Combine features and compute action
        combined_features = torch.cat([img_features, state], dim=1)
        output = self.mlp(combined_features)
        
        return output, {"log_std": self.log_std_parameter}


class DeFMValue(DeterministicMixin, Model):
    """
    Value network using DeFM ResNet18-BiFPN backbone.
    
    Architecture:
        1. DeFM backbone (frozen) extracts single-view P4 features
        2. Per-camera independent feature compression (no cross-camera spatial mixing)
        3. MLP combines image features + state -> scalar value
    """
    
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
        **kwargs
    ):
        # Initialize base classes
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)
        
        # Print configuration
        print("\n")
        print("======================= DeFMValue ========================")
        print("###### [Model Initialization]")
        print("observation_space:", observation_space)
        print("state_space:", state_space)
        print("action_space:", action_space)
        print("device:", device)
        print("\n")
        print("###### [DeterministicMixin Initialization]")
        print("clip_actions:", clip_actions)
        print("role:", role)
        print("\n")
        print("###### [DeFM Configuration]")
        print("defm_model_name:", defm_model_name)
        print("defm_pretrained_path:", defm_pretrained_path)
        print("==========================================================\n")
        
        # ------------------- Network Definition -------------------
        
        # 1. Parse input space dimensions
        # state_space is a Dict with "image" and "state" (for critic)
        # image shape: (4, 64, 32) -> 4 cameras, 64x32 each
        image_shape = state_space["image"].shape
        state_shape = state_space["state"].shape
        
        if len(image_shape) != 3:
            raise ValueError(f"Unsupported image shape: {image_shape}")

        # Preferred layout: (4, H, W). Also accept legacy stitched layout: (1, H, 4W).
        if image_shape[0] == 4:
            self.num_cameras = 4
            self.img_height = image_shape[1]
            self.img_width = image_shape[2]
        elif image_shape[0] == 1 and image_shape[2] % 4 == 0:
            self.num_cameras = 4
            self.img_height = image_shape[1]
            self.img_width = image_shape[2] // 4
        else:
            raise ValueError(
                f"DeFMValue expects image shape (4,H,W) or (1,H,4W), got {image_shape}"
            )
        state_dim = state_shape[0]
        
        print(f"[DeFM Value] Image shape: {image_shape} (cameras, H, W)")
        print(f"[DeFM Value] State dim: {state_dim}")
        
        # 2. Initialize DeFM backbone (frozen)
        self.defm_backbone = DeFMBackbone(
            model_name=defm_model_name,
            pretrained_path=defm_pretrained_path,
            device=device,
            freeze=True,
        )
        
        # 3. Per-camera feature compression (independent across cameras)
        self.feature_compression = nn.Sequential(
            nn.Conv2d(
                in_channels=self.defm_backbone.p4_channels,
                out_channels=32,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        compressed_feat_dim = (
            self.num_cameras * 32 * self.defm_backbone.p4_h * self.defm_backbone.p4_w
        )
        print(f"[DeFM Value] Compressed feature dim: {compressed_feat_dim}")

        # 4. Define MLP (compressed image features + state -> value)
        self.mlp = nn.Sequential(
            nn.Linear(compressed_feat_dim + state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output scalar value
        )
        
        print(f"[DeFM Value] MLP input dim: {compressed_feat_dim + state_dim}")
        print(f"[DeFM Value] MLP output dim: 1")
    
    def compute(self, inputs, role=""):
        # Unflatten states (critic uses state_space)
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        
        raw_depth = states["image"]   # expected: (N, 4, 64, 32)
        state = states["state"]       # (N, state_dim)
        n_batch = raw_depth.shape[0]

        # Backbone forward under no_grad to save GPU memory
        with torch.no_grad():
            processed_depth = preprocess_depth_independent(
                raw_depth,
                target_size=(self.img_height, self.img_width),
                device=raw_depth.device,
            )
            p4_features = self.defm_backbone(processed_depth)  # (N*4, 128, 4, 2)

        compressed_features = self.feature_compression(p4_features)
        compressed_features = compressed_features.reshape(n_batch, self.num_cameras, -1)
        img_features = compressed_features.reshape(n_batch, -1)
        
        # Combine features and compute value
        combined_features = torch.cat([img_features, state], dim=1)
        output = self.mlp(combined_features)
        
        return output, {}
