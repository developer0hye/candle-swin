# candle-swin

Swin Transformer V1 implementation for [Hugging Face Candle](https://github.com/huggingface/candle).

Pure Rust, no custom kernels — works on all Candle backends (CPU, CUDA, Metal, WASM).

## Model Variants

| Variant | embed_dim | depths | num_heads | window_size | Output channels |
|---------|-----------|--------|-----------|-------------|-----------------|
| Tiny | 96 | [2,2,6,2] | [3,6,12,24] | 7 | [96, 192, 384, 768] |
| Small | 96 | [2,2,18,2] | [3,6,12,24] | 7 | [96, 192, 384, 768] |
| Base | 128 | [2,2,18,2] | [4,8,16,32] | 12 | [128, 256, 512, 1024] |
| Large | 192 | [2,2,18,2] | [6,12,24,48] | 12 | [192, 384, 768, 1536] |

## Usage

```rust
use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use candle_swin::{SwinTransformer, swin_transformer::SwinTransformerConfig};

let device = &Device::Cpu;
let varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

let config = SwinTransformerConfig::large();
let model = SwinTransformer::new(&config, vb).unwrap();

// Input: [B, 3, H, W]
let input = candle_core::Tensor::randn(0f32, 1.0, (1, 3, 384, 384), device).unwrap();
let outputs = model.forward(&input).unwrap();
// outputs: Vec<Tensor> with 4 multi-scale feature maps
```

## Reference

- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

## License

Apache-2.0
