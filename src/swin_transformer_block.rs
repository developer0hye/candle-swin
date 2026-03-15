use candle_core::{Module, Result, Tensor};
use candle_nn::{Dropout, LayerNorm, Linear, VarBuilder};

use crate::window_attention::WindowAttention;

/// Partitions a `[B, H, W, C]` tensor into windows of `[num_windows*B, ws, ws, C]`.
pub fn window_partition(x: &Tensor, window_size: usize) -> Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;
    let nh: usize = h / window_size;
    let nw: usize = w / window_size;
    let x: Tensor = x.reshape((b, nh, window_size, nw, window_size, c))?;
    let x: Tensor = x.permute((0, 1, 3, 2, 4, 5))?; // [B, nh, nw, ws, ws, C]
    x.reshape((b * nh * nw, window_size, window_size, c))
}

/// Reverses window_partition: `[num_windows*B, ws, ws, C]` -> `[B, H, W, C]`.
pub fn window_reverse(windows: &Tensor, window_size: usize, h: usize, w: usize) -> Result<Tensor> {
    let c: usize = windows.dims()[3];
    let nh: usize = h / window_size;
    let nw: usize = w / window_size;
    let b: usize = windows.dims()[0] / (nh * nw);
    let x: Tensor = windows.reshape((b, nh, nw, window_size, window_size, c))?;
    let x: Tensor = x.permute((0, 1, 3, 2, 4, 5))?; // [B, nh, ws, nw, ws, C]
    x.reshape((b, h, w, c))
}

/// MLP block: Linear -> GELU -> Dropout -> Linear -> Dropout.
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    drop: Dropout,
}

impl Mlp {
    fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        drop: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1: Linear = candle_nn::linear(in_features, hidden_features, vb.pp("fc1"))?;
        let fc2: Linear = candle_nn::linear(hidden_features, out_features, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            drop: Dropout::new(drop),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x: Tensor = self.fc1.forward(x)?;
        let x: Tensor = x.gelu_erf()?;
        let x: Tensor = self.drop.forward(&x, false)?;
        let x: Tensor = self.fc2.forward(&x)?;
        self.drop.forward(&x, false)
    }
}

/// Swin Transformer Block.
///
/// Performs Layer Norm -> (Shifted) Window Attention -> residual -> Layer Norm -> MLP -> residual.
/// DropPath is omitted for inference (acts as identity).
///
/// Reference: Swin Transformer (https://arxiv.org/abs/2103.14030), Section 3.2
pub struct SwinTransformerBlock {
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    window_size: usize,
    shift_size: usize,
}

impl SwinTransformerBlock {
    pub fn new(
        dim: usize,
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        drop: f32,
        attn_drop: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1: LayerNorm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn: WindowAttention = WindowAttention::new(
            dim,
            (window_size, window_size),
            num_heads,
            qkv_bias,
            attn_drop,
            drop,
            vb.pp("attn"),
        )?;
        let norm2: LayerNorm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let mlp_hidden_dim: usize = (dim as f64 * mlp_ratio) as usize;
        let mlp: Mlp = Mlp::new(dim, mlp_hidden_dim, dim, drop, vb.pp("mlp"))?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
            shift_size,
        })
    }

    /// Forward pass.
    ///
    /// `x`: `[B, H*W, C]`
    /// `h`, `w`: spatial resolution
    /// `attn_mask`: optional `[num_windows, ws*ws, ws*ws]`
    ///
    /// Returns: `[B, H*W, C]`
    pub fn forward(
        &self,
        x: &Tensor,
        h: usize,
        w: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, _l, c) = x.dims3()?;
        let shortcut: Tensor = x.clone();

        let x: Tensor = self.norm1.forward(x)?;
        let x: Tensor = x.reshape((b, h, w, c))?;

        // Pad to multiples of window_size
        let pad_r: usize = (self.window_size - w % self.window_size) % self.window_size;
        let pad_b: usize = (self.window_size - h % self.window_size) % self.window_size;
        let x: Tensor = if pad_r > 0 || pad_b > 0 {
            // Pad [B, H, W, C] along H and W dims
            let x: Tensor = if pad_r > 0 {
                let zeros: Tensor = Tensor::zeros((b, h, pad_r, c), x.dtype(), x.device())?;
                Tensor::cat(&[&x, &zeros], 2)?
            } else {
                x
            };
            if pad_b > 0 {
                let w_padded: usize = w + pad_r;
                let zeros: Tensor = Tensor::zeros((b, pad_b, w_padded, c), x.dtype(), x.device())?;
                Tensor::cat(&[&x, &zeros], 1)?
            } else {
                x
            }
        } else {
            x
        };

        let (_, hp, wp, _) = x.dims4()?;

        // Cyclic shift
        let (shifted_x, use_mask): (Tensor, bool) = if self.shift_size > 0 {
            // torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
            let shifted: Tensor = x
                .roll(-(self.shift_size as i32), 1)?
                .roll(-(self.shift_size as i32), 2)?;
            (shifted, true)
        } else {
            (x, false)
        };

        // Partition into windows: [nW*B, ws, ws, C]
        let x_windows: Tensor = window_partition(&shifted_x, self.window_size)?;
        let n: usize = self.window_size * self.window_size;
        let x_windows: Tensor = x_windows.reshape(((), n, c))?; // [nW*B, ws*ws, C]

        // Window attention
        let mask: Option<&Tensor> = if use_mask { attn_mask } else { None };
        let attn_windows: Tensor = self.attn.forward(&x_windows, mask)?;

        // Merge windows back
        let attn_windows: Tensor =
            attn_windows.reshape(((), self.window_size, self.window_size, c))?;
        let shifted_x: Tensor = window_reverse(&attn_windows, self.window_size, hp, wp)?;

        // Reverse cyclic shift
        let x: Tensor = if self.shift_size > 0 {
            shifted_x
                .roll(self.shift_size as i32, 1)?
                .roll(self.shift_size as i32, 2)?
        } else {
            shifted_x
        };

        // Remove padding
        let x: Tensor = if pad_r > 0 || pad_b > 0 {
            x.narrow(1, 0, h)?.narrow(2, 0, w)?
        } else {
            x
        };

        let x: Tensor = x.reshape((b, h * w, c))?;

        // Residual + FFN
        let x: Tensor = (shortcut + x)?;
        let x_norm2: Tensor = self.norm2.forward(&x)?;
        let x_mlp: Tensor = self.mlp.forward(&x_norm2)?;
        let x: Tensor = (x + x_mlp)?;

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_window_partition_reverse_roundtrip() -> Result<()> {
        let device = &Device::Cpu;
        let (b, h, w, c) = (2, 14, 14, 96);
        let ws: usize = 7;
        let input = Tensor::randn(0f32, 1.0, (b, h, w, c), device)?;

        let windows = window_partition(&input, ws)?;
        assert_eq!(windows.dims(), &[b * 4, ws, ws, c]); // 14/7=2 -> 2*2=4 windows per image

        let restored = window_reverse(&windows, ws, h, w)?;
        assert_eq!(restored.dims(), &[b, h, w, c]);

        // Values should be preserved
        let diff: f64 = (input - restored)?.abs()?.sum_all()?.to_scalar::<f32>()? as f64;
        assert!(diff < 1e-6);
        Ok(())
    }

    #[test]
    fn test_swin_block_output_shape() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let dim: usize = 96;
        let h: usize = 56;
        let w: usize = 56;

        let block = SwinTransformerBlock::new(dim, 3, 7, 0, 4.0, true, 0.0, 0.0, vb)?;

        let input = Tensor::randn(0f32, 1.0, (1, h * w, dim), device)?;
        let output = block.forward(&input, h, w, None)?;

        assert_eq!(output.dims(), &[1, h * w, dim]);
        Ok(())
    }

    #[test]
    fn test_swin_block_with_shift() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let dim: usize = 96;
        let h: usize = 56;
        let w: usize = 56;
        let ws: usize = 7;
        let shift: usize = ws / 2; // 3

        let block = SwinTransformerBlock::new(dim, 3, ws, shift, 4.0, true, 0.0, 0.0, vb)?;

        let input = Tensor::randn(0f32, 1.0, (1, h * w, dim), device)?;

        // Need to provide attention mask for shifted window
        let n_windows_h: usize = h / ws;
        let n_windows_w: usize = w / ws;
        let n_windows: usize = n_windows_h * n_windows_w;
        let n: usize = ws * ws;
        let mask = Tensor::zeros((n_windows, n, n), DType::F32, device)?;

        let output = block.forward(&input, h, w, Some(&mask))?;
        assert_eq!(output.dims(), &[1, h * w, dim]);
        Ok(())
    }
}
