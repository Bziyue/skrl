"""
Custom models with DeFM (Deep Feature Model) backbone for depth-based RL.

This module provides Policy and Value networks that use DeFM's pretrained ResNet18-BiFPN
backbone for processing depth images. The backbone is frozen and P4 features are extracted.

Input:
    - 4 x 64x64 depth images (arranged as 2x2 grid -> 128x128)
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
from defm.utils.utils import preprocess_depth_batch


# Get path to DeFM weights directory
DEFM_WEIGHTS_DIR = pathlib.Path(__file__).resolve().parents[6] / "defm" / "weights"


class DeFMBackbone(nn.Module):
    """
    Wrapper around DeFM ResNet18-BiFPN backbone.
    
    - Loads pretrained weights and freezes all parameters
    - Extracts P4 features (stride 16) from BiFPN output
    - Preprocesses raw depth images using DeFM's metric-aware normalization
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
        
        # Compute output dimension using dummy input
        # Input: 4 x 128x128 depth images -> 256x256 grid
        dummy_input = torch.zeros(1, 3, 256, 256).to(self.device)
        
        with torch.no_grad():
            out_dict = self.backbone(dummy_input)
            p4_feat = out_dict["dense_bifpn"]["P4"]
            # Store spatial shape info: (channels, H, W)
            self.p4_channels = p4_feat.shape[1]   # 128
            self.p4_h = p4_feat.shape[2]           # 16
            self.p4_w = p4_feat.shape[3]           # 16
            
            print(f"[DeFM Backbone] P4 Feature Shape: {p4_feat.shape}")
            print(f"[DeFM Backbone] P4 Channels: {self.p4_channels}, Spatial: {self.p4_h}x{self.p4_w}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeFM backbone.
        
        Args:
            x: Preprocessed depth tensor of shape (N, 3, 256, 256)
        
        Returns:
            P4 features of shape (N, 128, 16, 16) - preserving spatial dimensions
        """
        with torch.no_grad():
            out_dict = self.backbone(x)
            return out_dict["dense_bifpn"]["P4"]


def preprocess_depth_grid(
    depth_images: torch.Tensor,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Preprocess 4 depth images into DeFM format.
    
    Takes 4 x 128x128 raw depth images, arranges them into a 2x2 grid (256x256),
    then applies DeFM's metric-aware 3-channel normalization.
    
    Args:
        depth_images: Raw depth tensor of shape (N, 4, 128, 128) 
                      where 4 cameras are: [front, right, back, left]
        device: Target device
    
    Returns:
        Preprocessed tensor of shape (N, 3, 256, 256)
    """
    N = depth_images.shape[0]
    
    # Arrange 4 cameras into 2x2 grid
    # Layout:  front  | right
    #          -------+-------
    #          back   | left
    front = depth_images[:, 0]  # (N, 128, 128)
    right = depth_images[:, 1]
    back = depth_images[:, 2]
    left = depth_images[:, 3]
    
    # Create 256x256 grid
    top_row = torch.cat([front, right], dim=2)     # (N, 128, 256)
    bottom_row = torch.cat([back, left], dim=2)    # (N, 128, 256)
    grid = torch.cat([top_row, bottom_row], dim=1) # (N, 256, 256)
    
    # Apply DeFM preprocessing (metric-aware 3-channel normalization)
    # Input: (N, 256, 256) raw metric depth
    # Output: (N, 3, 256, 256) normalized
    processed = preprocess_depth_batch(
        grid,
        target_size=256,
        cnn_padding=False,
        device=device,
    )
    
    return processed


class DeFMPolicy(GaussianMixin, Model):
    """
    Policy network using DeFM ResNet18-BiFPN backbone.
    
    Architecture:
        1. DeFM backbone (frozen) extracts P4 features from depth grid
        2. MLP combines image features + state -> action mean
        3. Learnable log_std parameter for Gaussian policy
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
        # image shape: (4, 128, 128) -> 4 cameras, 128x128 each
        image_shape = observation_space["image"].shape
        state_shape = observation_space["state"].shape
        action_shape = action_space.shape
        
        self.num_cameras = image_shape[0]
        self.img_height = image_shape[1]
        self.img_width = image_shape[2]
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
        
        # 3. Spatial adapter: 1x1 conv to compress P4 channels (128 -> 8)
        #    Keeps spatial resolution (16x16), then flatten -> 8*16*16 = 2048
        p4_ch = self.defm_backbone.p4_channels  # 128
        adapter_out_ch = 8
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(p4_ch, adapter_out_ch, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        adapter_out_dim = adapter_out_ch * self.defm_backbone.p4_h * self.defm_backbone.p4_w  # 8*16*16=2048
        
        print(f"[DeFM Policy] Spatial adapter: {p4_ch}ch -> {adapter_out_ch}ch, "
              f"flatten -> {adapter_out_dim}")
        
        # 4. Define MLP (adapter features + state -> action)
        self.mlp = nn.Sequential(
            nn.Linear(adapter_out_dim + state_dim, 512),
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
        
        print(f"[DeFM Policy] MLP input dim: {adapter_out_dim + state_dim}")
        print(f"[DeFM Policy] MLP output dim: {action_dim}")
    
    def compute(self, inputs, role=""):
        # Unflatten observations
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        
        # Get raw depth images and state
        raw_depth = observations["image"]   # (N, 4, 128, 128)
        state = observations["state"]       # (N, state_dim)
        
        # Backbone forward under no_grad: frozen params don't need gradients,
        # and skipping activation storage saves significant GPU memory.
        with torch.no_grad():
            processed_depth = preprocess_depth_grid(raw_depth, device=raw_depth.device)
            p4_features = self.defm_backbone(processed_depth)  # (N, 128, 16, 16)
        
        # Compress via spatial adapter (N, 128, 16, 16) -> (N, 2048)
        img_features = self.spatial_adapter(p4_features)
        
        # Combine features and compute action
        combined_features = torch.cat([img_features, state], dim=1)
        output = self.mlp(combined_features)
        
        return output, {"log_std": self.log_std_parameter}


class DeFMValue(DeterministicMixin, Model):
    """
    Value network using DeFM ResNet18-BiFPN backbone.
    
    Architecture:
        1. DeFM backbone (frozen) extracts P4 features from depth grid
        2. MLP combines image features + state -> scalar value
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
        image_shape = state_space["image"].shape
        state_shape = state_space["state"].shape
        
        self.num_cameras = image_shape[0]
        self.img_height = image_shape[1]
        self.img_width = image_shape[2]
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
        
        # 3. Spatial adapter: 1x1 conv to compress P4 channels (128 -> 8)
        #    Keeps spatial resolution (16x16), then flatten -> 8*16*16 = 2048
        p4_ch = self.defm_backbone.p4_channels  # 128
        adapter_out_ch = 8
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(p4_ch, adapter_out_ch, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        adapter_out_dim = adapter_out_ch * self.defm_backbone.p4_h * self.defm_backbone.p4_w  # 8*16*16=2048
        
        print(f"[DeFM Value] Spatial adapter: {p4_ch}ch -> {adapter_out_ch}ch, "
              f"flatten -> {adapter_out_dim}")
        
        # 4. Define MLP (adapter features + state -> value)
        self.mlp = nn.Sequential(
            nn.Linear(adapter_out_dim + state_dim, 512),
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
        
        print(f"[DeFM Value] MLP input dim: {adapter_out_dim + state_dim}")
        print(f"[DeFM Value] MLP output dim: 1")
    
    def compute(self, inputs, role=""):
        # Unflatten states (critic uses state_space)
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        
        raw_depth = states["image"]   # (N, 4, 128, 128)
        state = states["state"]       # (N, state_dim)
        
        # Backbone forward under no_grad to save GPU memory
        with torch.no_grad():
            processed_depth = preprocess_depth_grid(raw_depth, device=raw_depth.device)
            p4_features = self.defm_backbone(processed_depth)  # (N, 128, 16, 16)
        
        # Compress via spatial adapter (N, 128, 16, 16) -> (N, 2048)
        img_features = self.spatial_adapter(p4_features)
        
        # Combine features and compute value
        combined_features = torch.cat([img_features, state], dim=1)
        output = self.mlp(combined_features)
        
        return output, {}
