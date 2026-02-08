# ==============================================================================
# DINO-SAE: Autoencoder Module
#
# This script implements the Autoencoder component for the DINO-SAE project.
#
# Acknowledgements:
# A significant portion of this code is heavily referenced from the EfficientViT
# GitHub repository.
#
# Reference URL: 
# https://github.com/dc-ai-projects/DC-Gen 
# https://github.com/mit-han-lab/efficientvit?tab=readme-ov-file
# ==============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Any, Optional
from omegaconf import ListConfig, OmegaConf
from efficientvit.models.nn.ops import EfficientViTBlock

def val2tuple(x: Any, n: int) -> tuple[Any, ...]:
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,) * n

def get_same_padding(kernel_size: int) -> int:
    return (kernel_size - 1) // 2

def build_act(act_func: Optional[str], inplace: bool = False) -> Optional[nn.Module]:
    if act_func is None:
        return None
    if act_func.lower() == "relu":
        return nn.ReLU(inplace=inplace)
    if act_func.lower() == "silu":
        return nn.SiLU(inplace=inplace)
    if act_func.lower() == "tanh":
        return nn.Tanh()
    if act_func.lower() == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation function: {act_func}")

def build_norm(norm: Optional[str], num_features: Optional[int] = None) -> Optional[nn.Module]:
    if norm is None:
        return None
    if norm.lower() == "bn2d":
        return nn.BatchNorm2d(num_features)
    if norm.lower() == "trms2d":
        return RMSNorm2d(num_features) 
    raise ValueError(f"Unsupported normalization: {norm}")

class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class OpSequential(nn.Module):
    def __init__(self, op_list: list[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, groups=1, use_bias=False, norm=None, padding_mode='zeros', act_func=None):
        super(ConvLayer, self).__init__()
        padding = get_same_padding(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=padding, groups=groups, bias=use_bias, padding_mode=padding_mode)
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act: x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x = self.conv(x)
        return x
    

class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.factor)
        return x

class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x

    
class RMSNorm1d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The calculation is done in float32 to avoid precision issues.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RMSNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over C, H, W for each item in the batch
        variance = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

# ====================================================================================
# Builder functions from dc_ae.py
# ====================================================================================

def build_block(block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]) -> nn.Module:
    if block_type == "ResBlock":
        assert in_channels == out_channels
        main_block = ResBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, use_bias=(True, False), norm=(None, norm), act_func=(act, None))
        return ResidualBlock(main_block, IdentityLayer())
    elif block_type == "EViT_GLU":
        if EfficientViTBlock is None:
            raise ImportError("`efficientvit` is not installed. Please install it to use the 'EViT_GLU' block type.")
        assert in_channels == out_channels
        
        # If norm is trms2d, use our new PyTorch-native RMSNorm2d to avoid Triton.
        # Our environment couldn't handle Triton package.
        # Otherwise, pass the norm string to the block.
        if norm == "trms2d":
            norm_layer = RMSNorm2d(in_channels)
        else:
            norm_layer = norm

        return EfficientViTBlock(in_channels, norm=norm_layer, act_func=act, local_module="GLUMBConv", scales=())
    raise ValueError(f"block_type {block_type} is not supported")

def build_stage_main(width: int, depth: int, block_type: str, norm: str, act: str, input_width: int) -> list[nn.Module]:
    stage = []
    for d in range(depth):
        block = build_block(block_type=block_type, in_channels=width if d > 0 else input_width, out_channels=width, norm=norm, act=act)
        stage.append(block)
    return stage

def build_upsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2)
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    
    if shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels=in_channels, out_channels=out_channels, factor=2)
        return ResidualBlock(block, shortcut_block)
    elif shortcut == "InterpolateConv":
        shortcut_block = InterpolateConvUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
        return ResidualBlock(block, shortcut_block)
    elif shortcut is None:
        return block
    raise ValueError(f"shortcut {shortcut} is not supported for upsample")

def build_decoder_project_in_block(in_channels: int, out_channels: int, shortcut: Optional[str]):
    block = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        use_bias=True,
        norm=None,
        act_func=None,
    )
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    elif shortcut == "InterpolateConv":
        shortcut_block = InterpolateConvUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
        return ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for decoder project in")
    return block


def build_decoder_project_out_block(in_channels: int, out_channels: int, factor: int, upsample_block_type: str, norm: Optional[str], act: Optional[str]):
    layers: list[nn.Module] = [build_norm(norm, in_channels), build_act(act)]
    if factor == 1:
        layers.append(ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, use_bias=True, norm=None, act_func=None))
    elif factor == 2:
        layers.append(build_upsample_block(block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None))
    else:
        raise ValueError(f"upsample factor {factor} is not supported for decoder project out")
    return OpSequential(layers)

# ====================================================================================
# DCAE Decoder Implementation (from dc_ae.py)
# ====================================================================================

@dataclass
class DecoderConfig:
    in_channels: int
    latent_channels: int
    in_shortcut: Optional[str] = "duplicating"
    width_list: tuple[int, ...] = (128, 256, 512, 512)
    depth_list: tuple[int, ...] = (2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: str = "ConvPixelShuffle"
    upsample_match_channel: bool = True
    upsample_shortcut: str = "duplicating"
    out_norm: str = "trms2d"
    out_act: str = "relu"

class DCAE_Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages

        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, (list, ListConfig)) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (
            isinstance(cfg.norm, (list, ListConfig)) and len(cfg.norm) == num_stages
        )
        assert isinstance(cfg.act, str) or (
            isinstance(cfg.act, (list, ListConfig)) and len(cfg.act) == num_stages
        )

        self.project_in = build_decoder_project_in_block(in_channels=cfg.latent_channels, out_channels=cfg.width_list[-1], shortcut=cfg.in_shortcut)

        stages_list: list[OpSequential] = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                print(cfg.upsample_block_type, cfg.width_list[stage_id + 1], width)
                upsample_block = build_upsample_block(
                    block_type=cfg.upsample_block_type,
                    in_channels=cfg.width_list[stage_id + 1],
                    out_channels=width if cfg.upsample_match_channel else cfg.width_list[stage_id + 1],
                    shortcut=cfg.upsample_shortcut,
                )
                stage.append(upsample_block)

            input_width_for_stage = width if cfg.upsample_match_channel else cfg.width_list[min(stage_id + 1, num_stages - 1)]
            
            block_type = cfg.block_type[stage_id] if not isinstance(cfg.block_type, str) else cfg.block_type
            norm = cfg.norm[stage_id] if not isinstance(cfg.norm, str) else cfg.norm
            act = cfg.act[stage_id] if not isinstance(cfg.act, str) else cfg.act
            
            stage.extend(build_stage_main(width=width, depth=depth, block_type=block_type, norm=norm, act=act, input_width=input_width_for_stage))
            stages_list.insert(0, OpSequential(stage))
        stages_list.reverse()
        self.stages = nn.ModuleList(stages_list)

        self.project_out = build_decoder_project_out_block(
            in_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            out_channels=cfg.in_channels,
            factor=1 if cfg.depth_list[0] > 0 else 2,
            upsample_block_type=cfg.upsample_block_type,
            norm=cfg.out_norm,
            act=cfg.out_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) == 0: continue
            x = stage(x)
        x = self.project_out(x)
        return x

# ====================================================================================
# Top-level model for our project
# ====================================================================================

class CnnPatchEmbed(nn.Module):
    """
    A CNN-based patch embedding module to replace the standard Conv2d patch embedding
    in Vision Transformers. This allows capturing richer local details at the earliest stage.
    The output is a 4D tensor, as expected by the DINO model's internal processing.
    """
    def __init__(self, embed_dim: int, in_chans: int = 3):
        super().__init__()
        # This CNN downsamples the image by a factor of 16 (2**4)
        # and projects it to the `embed_dim`.
        norm_type = 'trms2d'
        self.cnn = nn.Sequential(
            ConvLayer(in_chans, 64, kernel_size=7, stride=2, norm=norm_type, act_func='silu'),      # 2x downsample
            ConvLayer(64, 128, kernel_size=3, stride=2, norm=norm_type, act_func='silu'),     # 4x downsample
            ConvLayer(128, 256, kernel_size=3, stride=2, norm=norm_type, act_func='silu'),     # 8x downsample
            ConvLayer(256, embed_dim, kernel_size=3, stride=2, norm=None, act_func=None), # 16x downsample
        )
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                k = 1.0 / fan_in
                limit = math.sqrt(k)
                nn.init.uniform_(m.weight, -limit, limit)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -limit, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output a 4D tensor (B, C, H, W)
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1)
        return x

class DINOSphericalAutoencoder(nn.Module):
    """
    An autoencoder that uses a DINO model with an enhanced CNN-based patch embedder.
    """
    def __init__(self, dino_cfg: OmegaConf, decoder_cfg: DecoderConfig, train_cnn_embedder: bool = True):
        super().__init__()
        print("Initializing DINOSphericalAutoencoder (CNN Patch Embedding)...")
        
        # 1. Load the pre-trained DINO model
        self.dino_model = torch.hub.load(
            dino_cfg.repo_path, dino_cfg.model_name,
            source='local', force_reload=True
        )

        self.dino_model.load_state_dict(torch.load(dino_cfg.weights_path, map_location="cpu"))
        
        # 2. Create our new CNN-based patch embedder
        dino_embedding_dim = self.dino_model.embed_dim
        print(f"[DEBUG] Fetched dino_model.embed_dim: {dino_embedding_dim}")
        new_patch_embed = CnnPatchEmbed(embed_dim=dino_embedding_dim)
        
        # 3. Replace the original patch_embed layer
        self.dino_model.patch_embed = new_patch_embed
        print(f"Replaced DINO's patch_embed layer with a new CNN-based embedder.")

        # 4. Freeze parameters selectively
        for name, param in self.dino_model.named_parameters():
            if 'patch_embed' in name and train_cnn_embedder:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.dino_model.patch_embed.reset_parameters()
        self.decoder = DCAE_Decoder(decoder_cfg)
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoded_out = self.encode(image)
        patch_tokens = encoded_out['patch_tokens']

        B, N, C = patch_tokens.shape
        H = W = int(N**0.5)
        decoder_input = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, C, H, W)


        # 3. Decode
        reconstructed_image = self.decode(decoder_input)
        
        return reconstructed_image, patch_tokens
    
    def encode(self, image: torch.Tensor):
        dino_output = self.dino_model.forward_features(image)
        
        return {
            'patch_tokens': dino_output['x_norm_patchtokens'], # (B, N, C)
            'cls_token': dino_output['x_norm_clstoken']        # (B, C) for linear probing
        }
    
    def decode(self, decoder_input):
        image = self.decoder(decoder_input)
        return image
