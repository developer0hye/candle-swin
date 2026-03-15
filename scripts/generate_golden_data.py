"""Generate golden test data for candle-swin by running PyTorch Swin V1.

Saves input tensors, model weights, and expected outputs as safetensors files.
Each test case captures a specific module's forward pass for cross-validation.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.expanduser("~/Documents/Projects/BiRefNet"))

# Disable SDPA to use the manual attention path (compatible with our Candle impl)
import config as birefnet_config
birefnet_config.Config.SDPA_enabled = False
_original_init = birefnet_config.Config.__init__
def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.SDPA_enabled = False
birefnet_config.Config.__init__ = _patched_init

from models.backbones.swin_v1 import (
    PatchEmbed,
    PatchMerging,
    WindowAttention,
    SwinTransformerBlock,
    SwinTransformer,
    window_partition,
    window_reverse,
)

torch.manual_seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test-data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name: str, tensors: dict[str, torch.Tensor]):
    # Ensure all tensors are contiguous float32
    out = {}
    for k, v in tensors.items():
        if v.dtype in (torch.float16, torch.bfloat16):
            v = v.float()
        out[k] = v.contiguous()
    path = os.path.join(OUTPUT_DIR, f"{name}.safetensors")
    save_file(out, path)
    print(f"Saved {path} ({len(out)} tensors)")


def generate_patch_embed():
    """Test PatchEmbed with patch_size=4, in_channels=3, embed_dim=96, with norm."""
    pe = PatchEmbed(patch_size=4, in_channels=3, embed_dim=96, norm_layer=nn.LayerNorm)
    pe.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = pe(x)

    state = pe.state_dict()
    save("patch_embed", {
        "input": x,
        "output": y,
        **{f"param.{k}": v for k, v in state.items()},
    })


def generate_patch_merging():
    """Test PatchMerging with dim=96, H=16, W=16."""
    pm = PatchMerging(dim=96, norm_layer=nn.LayerNorm)
    pm.eval()

    H, W = 16, 16
    x = torch.randn(1, H * W, 96)
    with torch.no_grad():
        y = pm(x, H, W)

    state = pm.state_dict()
    save("patch_merging", {
        "input": x,
        "output": y,
        "H": torch.tensor(H, dtype=torch.int64),
        "W": torch.tensor(W, dtype=torch.int64),
        **{f"param.{k}": v for k, v in state.items()},
    })


def generate_window_attention():
    """Test WindowAttention with dim=96, window_size=7, num_heads=3."""
    wa = WindowAttention(
        dim=96,
        window_size=(7, 7),
        num_heads=3,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    wa.eval()

    N = 7 * 7  # 49
    num_windows = 4
    x = torch.randn(num_windows, N, 96)
    with torch.no_grad():
        y = wa(x, mask=None)

    state = wa.state_dict()
    save("window_attention_no_mask", {
        "input": x,
        "output": y,
        **{f"param.{k}": v for k, v in state.items()},
    })


def generate_window_attention_with_mask():
    """Test WindowAttention with an attention mask (shifted window)."""
    wa = WindowAttention(
        dim=96,
        window_size=(7, 7),
        num_heads=3,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    wa.eval()

    N = 49
    num_windows = 4
    batch = 2
    B_ = batch * num_windows

    x = torch.randn(B_, N, 96)

    # Build a realistic mask (some positions blocked)
    mask = torch.zeros(num_windows, N, N)
    mask[1, :24, 24:] = float("-inf")
    mask[1, 24:, :24] = float("-inf")
    mask[2, :24, 24:] = float("-inf")
    mask[2, 24:, :24] = float("-inf")

    with torch.no_grad():
        y = wa(x, mask=mask)

    state = wa.state_dict()
    save("window_attention_with_mask", {
        "input": x,
        "output": y,
        "mask": mask,
        **{f"param.{k}": v for k, v in state.items()},
    })


def generate_swin_block():
    """Test SwinTransformerBlock without shift (W-MSA)."""
    block = SwinTransformerBlock(
        dim=96,
        num_heads=3,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )
    block.eval()

    H, W = 14, 14
    x = torch.randn(1, H * W, 96)
    block.H = H
    block.W = W

    # Need attn_mask for the block.forward
    mask_matrix = None  # no shift -> no mask needed

    with torch.no_grad():
        y = block(x, mask_matrix)

    state = block.state_dict()
    save("swin_block_no_shift", {
        "input": x,
        "output": y,
        "H": torch.tensor(H, dtype=torch.int64),
        "W": torch.tensor(W, dtype=torch.int64),
        **{f"param.{k}": v for k, v in state.items()},
    })


def generate_swin_block_shifted():
    """Test SwinTransformerBlock with shift (SW-MSA)."""
    ws = 7
    shift = ws // 2
    block = SwinTransformerBlock(
        dim=96,
        num_heads=3,
        window_size=ws,
        shift_size=shift,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )
    block.eval()

    H, W = 14, 14
    x = torch.randn(1, H * W, 96)
    block.H = H
    block.W = W

    # Build attention mask for shifted window (same logic as BasicLayer)
    Hp = int(torch.ceil(torch.tensor(H) / ws).to(torch.int64) * ws)
    Wp = int(torch.ceil(torch.tensor(W) / ws).to(torch.int64) * ws)
    img_mask = torch.zeros((1, Hp, Wp, 1))
    h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, ws)
    mask_windows = mask_windows.view(-1, ws * ws)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(
        attn_mask == 0, float(0.0)
    )

    with torch.no_grad():
        y = block(x, attn_mask)

    state = block.state_dict()
    save("swin_block_shifted", {
        "input": x,
        "output": y,
        "attn_mask": attn_mask,
        "H": torch.tensor(H, dtype=torch.int64),
        "W": torch.tensor(W, dtype=torch.int64),
        **{f"param.{k}": v for k, v in state.items()},
    })


def generate_swin_tiny_full():
    """Test full SwinTransformer tiny with small input (64x64 for speed)."""
    model = SwinTransformer(
        patch_size=4,
        in_channels=3,
        embed_dim=96,
        depths=[2, 2, 2, 2],  # Reduced depths for test speed
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
    )
    model.eval()

    x = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        outs = model(x)

    state = model.state_dict()
    tensors = {
        "input": x,
        **{f"output_{i}": out for i, out in enumerate(outs)},
        **{f"param.{k}": v for k, v in state.items()},
    }
    save("swin_tiny_full", tensors)


if __name__ == "__main__":
    generate_patch_embed()
    generate_patch_merging()
    generate_window_attention()
    generate_window_attention_with_mask()
    generate_swin_block()
    generate_swin_block_shifted()
    generate_swin_tiny_full()
    print("\nAll golden test data generated successfully!")
