use candle_core::{D, DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Dropout, Linear, VarBuilder};

/// Window-based Multi-Head Self-Attention (W-MSA / SW-MSA).
///
/// Computes self-attention within local windows, with learnable relative position bias.
/// Supports both regular and shifted window attention via an optional attention mask.
///
/// Reference: Swin Transformer (https://arxiv.org/abs/2103.14030), Section 3.2
pub struct WindowAttention {
    qkv: Linear,
    proj: Linear,
    attn_drop: Dropout,
    proj_drop: Dropout,
    num_heads: usize,
    scale: f64,
    /// Learnable relative position bias table: `[(2*Wh-1)*(2*Ww-1), num_heads]`
    relative_position_bias_table: Tensor,
    /// Precomputed index: `[Wh*Ww, Wh*Ww]` — registered buffer, not a parameter
    relative_position_index: Tensor,
    window_size: (usize, usize),
}

impl WindowAttention {
    pub fn new(
        dim: usize,
        window_size: (usize, usize),
        num_heads: usize,
        qkv_bias: bool,
        attn_drop: f32,
        proj_drop: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim: usize = dim / num_heads;
        let scale: f64 = (head_dim as f64).powf(-0.5);

        let qkv: Linear = if qkv_bias {
            candle_nn::linear(dim, dim * 3, vb.pp("qkv"))?
        } else {
            candle_nn::linear_no_bias(dim, dim * 3, vb.pp("qkv"))?
        };
        let proj: Linear = candle_nn::linear(dim, dim, vb.pp("proj"))?;

        // Relative position bias table: [(2*Wh-1)*(2*Ww-1), num_heads]
        let table_size: usize = (2 * window_size.0 - 1) * (2 * window_size.1 - 1);
        let relative_position_bias_table: Tensor =
            vb.get((table_size, num_heads), "relative_position_bias_table")?;

        // Compute relative position index (non-learnable buffer)
        let relative_position_index: Tensor =
            Self::build_relative_position_index(window_size, vb.device())?;

        Ok(Self {
            qkv,
            proj,
            attn_drop: Dropout::new(attn_drop),
            proj_drop: Dropout::new(proj_drop),
            num_heads,
            scale,
            relative_position_bias_table,
            relative_position_index,
            window_size,
        })
    }

    /// Build the relative position index lookup table.
    ///
    /// Returns `[Wh*Ww, Wh*Ww]` i64 tensor mapping each (query, key) pair
    /// to a row in `relative_position_bias_table`.
    fn build_relative_position_index(
        window_size: (usize, usize),
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let wh: usize = window_size.0;
        let ww: usize = window_size.1;
        let _n: usize = wh * ww;

        // coords: [2, Wh, Ww]
        let coords_h: Tensor = Tensor::arange(0i64, wh as i64, device)?;
        let coords_w: Tensor = Tensor::arange(0i64, ww as i64, device)?;

        // Flatten to [Wh*Ww] each, then compute pairwise differences
        // coords_h_flat[i] = i / ww, coords_w_flat[i] = i % ww (for row-major)
        let coords_h_flat: Tensor = coords_h.unsqueeze(1)?.expand((wh, ww))?.flatten_all()?; // [n]
        let coords_w_flat: Tensor = coords_w.unsqueeze(0)?.expand((wh, ww))?.flatten_all()?; // [n]

        // relative_coords_h[i, j] = coords_h[i] - coords_h[j]
        let rel_h: Tensor = coords_h_flat
            .unsqueeze(1)?
            .broadcast_sub(&coords_h_flat.unsqueeze(0)?)?; // [n, n]
        let rel_w: Tensor = coords_w_flat
            .unsqueeze(1)?
            .broadcast_sub(&coords_w_flat.unsqueeze(0)?)?; // [n, n]

        // Shift to start from 0
        let rel_h: Tensor = (rel_h + (wh as f64 - 1.0))?;
        let rel_w: Tensor = (rel_w + (ww as f64 - 1.0))?;

        // rel_h * (2*Ww - 1) + rel_w -> flat index into bias table
        let index: Tensor = (rel_h * (2.0 * ww as f64 - 1.0))?.add(&rel_w)?;

        Ok(index)
    }

    /// Forward pass.
    ///
    /// `x`: `[num_windows*B, N, C]` where `N = Wh * Ww`
    /// `mask`: optional `[num_windows, N, N]` attention mask (0 / -inf values)
    ///
    /// Returns: `[num_windows*B, N, C]`
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_windows, n, c) = x.dims3()?;
        let head_dim: usize = c / self.num_heads;

        // QKV projection: [B_, N, 3*C] -> [B_, N, 3, num_heads, head_dim] -> [3, B_, num_heads, N, head_dim]
        let qkv: Tensor = self
            .qkv
            .forward(x)?
            .reshape((b_windows, n, 3, self.num_heads, head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q: Tensor = qkv.i(0)?.contiguous()?; // [B_, num_heads, N, head_dim]
        let k: Tensor = qkv.i(1)?.contiguous()?;
        let v: Tensor = qkv.i(2)?.contiguous()?;

        // Scaled dot-product: Q @ K^T * scale
        let q: Tensor = (q * self.scale)?;
        let attn: Tensor = q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?; // [B_, num_heads, N, N]

        // Add relative position bias
        // index: [N, N] (i64) -> flatten -> index_select from table -> reshape to [N, N, num_heads] -> permute to [num_heads, N, N]
        let index_flat: Tensor = self
            .relative_position_index
            .flatten_all()?
            .to_dtype(DType::U32)?;
        let bias: Tensor = self
            .relative_position_bias_table
            .index_select(&index_flat, 0)?
            .reshape((n, n, self.num_heads))?
            .permute((2, 0, 1))?
            .unsqueeze(0)?; // [1, num_heads, N, N]
        let bias: Tensor = bias.to_dtype(attn.dtype())?;
        let attn: Tensor = attn.broadcast_add(&bias)?;

        // Apply attention mask if present
        let attn: Tensor = if let Some(mask) = mask {
            // mask: [nW, N, N] -> expand for batch
            // attn: [B_, num_heads, N, N] where B_ = batch * nW
            let n_windows: usize = mask.dims()[0];
            let batch: usize = b_windows / n_windows;
            let attn: Tensor = attn.reshape((batch, n_windows, self.num_heads, n, n))?;
            let mask: Tensor = mask.unsqueeze(0)?.unsqueeze(2)?; // [1, nW, 1, N, N]
            let attn: Tensor = attn.broadcast_add(&mask)?;
            attn.reshape((b_windows, self.num_heads, n, n))?
        } else {
            attn
        };

        // Softmax + dropout
        let attn: Tensor = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn: Tensor = self.attn_drop.forward(&attn, false)?;

        // Attn @ V -> [B_, num_heads, N, head_dim] -> transpose -> [B_, N, C]
        let x: Tensor = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b_windows, n, c))?;

        // Output projection
        let x: Tensor = self.proj.forward(&x)?;
        let x: Tensor = self.proj_drop.forward(&x, false)?;

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_window_attention_output_shape() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let dim: usize = 96;
        let window_size: (usize, usize) = (7, 7);
        let num_heads: usize = 3;
        let n: usize = window_size.0 * window_size.1; // 49

        let attn = WindowAttention::new(dim, window_size, num_heads, true, 0.0, 0.0, vb)?;

        let num_windows: usize = 8;
        let input = Tensor::randn(0f32, 1.0, (num_windows, n, dim), device)?;
        let output = attn.forward(&input, None)?;

        assert_eq!(output.dims(), &[num_windows, n, dim]);
        Ok(())
    }

    #[test]
    fn test_window_attention_with_mask() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let dim: usize = 96;
        let window_size: (usize, usize) = (7, 7);
        let num_heads: usize = 3;
        let n: usize = 49;

        let attn = WindowAttention::new(dim, window_size, num_heads, true, 0.0, 0.0, vb)?;

        let num_windows: usize = 4;
        let batch: usize = 2;
        let b_total: usize = batch * num_windows;
        let input = Tensor::randn(0f32, 1.0, (b_total, n, dim), device)?;
        let mask = Tensor::zeros((num_windows, n, n), DType::F32, device)?;
        let output = attn.forward(&input, Some(&mask))?;

        assert_eq!(output.dims(), &[b_total, n, dim]);
        Ok(())
    }

    #[test]
    fn test_relative_position_index_range() -> Result<()> {
        let device = &Device::Cpu;
        let window_size: (usize, usize) = (7, 7);
        let index = WindowAttention::build_relative_position_index(window_size, device)?;

        assert_eq!(index.dims(), &[49, 49]);

        // All values should be in [0, (2*7-1)*(2*7-1) - 1] = [0, 168]
        let max_val: f64 = index.max(0)?.max(0)?.to_scalar::<i64>()? as f64;
        let min_val: f64 = index.min(0)?.min(0)?.to_scalar::<i64>()? as f64;
        assert!(min_val >= 0.0);
        assert!(max_val <= 168.0);
        Ok(())
    }
}
