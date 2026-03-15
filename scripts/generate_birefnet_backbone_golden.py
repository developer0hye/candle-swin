"""Generate golden test data using BiRefNet's pretrained Swin-L backbone.

Loads the official BiRefNet model from HuggingFace, runs a real image through
just the Swin backbone, and saves weights + input + 4-stage outputs.
"""

import sys
import os
import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, os.path.expanduser("~/Documents/Projects/BiRefNet"))

# Disable SDPA for reproducibility with manual attention
import config as birefnet_config
_original_init = birefnet_config.Config.__init__
def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.SDPA_enabled = False
birefnet_config.Config.__init__ = _patched_init

from models.backbones.swin_v1 import SwinTransformer, swin_v1_l

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test-data")

BIREFNET_WEIGHTS = (
    "/tmp/birefnet_cache/models--ZhengPeng7--BiRefNet/snapshots/"
    "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/model.safetensors"
)


def load_swin_backbone_weights():
    """Load Swin-L backbone weights from BiRefNet's pretrained model."""
    from safetensors import safe_open

    model = swin_v1_l()
    model_dict = model.state_dict()

    f = safe_open(BIREFNET_WEIGHTS, framework="pt")
    loaded = {}
    skipped_buffers = []
    for key in f.keys():
        if not key.startswith("bb."):
            continue
        param_name = key[3:]  # strip "bb."
        tensor = f.get_tensor(key).float()  # fp16 -> fp32

        if param_name in model_dict:
            if tensor.shape == model_dict[param_name].shape:
                loaded[param_name] = tensor
            else:
                print(f"Shape mismatch: {param_name} model={model_dict[param_name].shape} file={tensor.shape}")
        else:
            skipped_buffers.append(param_name)

    print(f"Loaded {len(loaded)}/{len(model_dict)} params, skipped {len(skipped_buffers)} buffers")
    model_dict.update(loaded)
    model.load_state_dict(model_dict, strict=False)
    model.eval()
    return model


def generate():
    model = load_swin_backbone_weights()

    torch.manual_seed(42)

    # Use a small input for manageable test-data size
    # BiRefNet default is 1024x1024, but backbone works at any resolution
    # Use 384x384 (Swin-L was pretrained at 384)
    x = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        outputs = model(x)

    print(f"Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"Output {i}: {out.shape}")

    # Save model weights with candle-compatible key names
    state = model.state_dict()
    tensors = {"input": x}
    for i, out in enumerate(outputs):
        tensors[f"output_{i}"] = out

    # Save params (skip relative_position_index buffers - they're computed in Candle)
    for k, v in state.items():
        if "relative_position_index" in k:
            continue
        tensors[f"param.{k}"] = v.float().contiguous()

    save_file(tensors, os.path.join(OUTPUT_DIR, "birefnet_swin_backbone.safetensors"))
    print(f"\nSaved birefnet_swin_backbone.safetensors ({len(tensors)} tensors)")


if __name__ == "__main__":
    generate()
