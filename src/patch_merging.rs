use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

/// Patch Merging Layer.
///
/// Downsamples spatial resolution by 2x via stride-2 subsampling, then concatenates
/// 4 sub-patches and projects from `4*dim` to `2*dim` with a linear layer.
///
/// Input: `[B, H*W, C]`
/// Output: `[B, (H/2)*(W/2), 2*C]`
///
/// Reference: Swin Transformer (https://arxiv.org/abs/2103.14030), Section 3.1
pub struct PatchMerging {
    reduction: Linear,
    norm: LayerNorm,
}

impl PatchMerging {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let reduction = candle_nn::linear_no_bias(4 * dim, 2 * dim, vb.pp("reduction"))?;
        let norm = candle_nn::layer_norm(4 * dim, 1e-5, vb.pp("norm"))?;
        Ok(Self { reduction, norm })
    }

    /// Forward pass.
    ///
    /// `x`: `[B, H*W, C]`
    /// `h`, `w`: spatial dimensions
    ///
    /// Returns `(output, new_h, new_w)` where output is `[B, (H/2)*(W/2), 2*C]`.
    pub fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<(Tensor, usize, usize)> {
        let (b, _l, c) = x.dims3()?;
        let x: Tensor = x.reshape((b, h, w, c))?;

        // Pad if odd dimensions
        let x: Tensor = if h % 2 == 1 || w % 2 == 1 {
            let pad_h: usize = h % 2;
            let pad_w: usize = w % 2;
            // Pad along W (dim 2) and H (dim 1), channels-last layout [B, H, W, C]
            let x: Tensor = if pad_w > 0 {
                let zeros: Tensor = Tensor::zeros((b, h, pad_w, c), x.dtype(), x.device())?;
                Tensor::cat(&[&x, &zeros], 2)?
            } else {
                x
            };
            if pad_h > 0 {
                let new_w: usize = w + pad_w;
                let zeros: Tensor = Tensor::zeros((b, pad_h, new_w, c), x.dtype(), x.device())?;
                Tensor::cat(&[&x, &zeros], 1)?
            } else {
                x
            }
        } else {
            x
        };

        let (_, h_padded, w_padded, _) = x.dims4()?;
        let new_h: usize = h_padded / 2;
        let new_w: usize = w_padded / 2;

        // Stride-2 subsampling: x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], etc.
        // Reshape [B, H, W, C] -> [B, H/2, 2, W/2, 2, C], then index dim 2 and dim 4
        let x_reshaped: Tensor = x.reshape((b, new_h, 2, new_w, 2, c))?;
        // x0 = x[:, 0::2, 0::2, :] = x_reshaped[:, :, 0, :, 0, :]
        let x0: Tensor = x_reshaped
            .narrow(2, 0, 1)?
            .narrow(4, 0, 1)?
            .reshape((b, new_h, new_w, c))?;
        // x1 = x[:, 1::2, 0::2, :] = x_reshaped[:, :, 1, :, 0, :]
        let x1: Tensor = x_reshaped
            .narrow(2, 1, 1)?
            .narrow(4, 0, 1)?
            .reshape((b, new_h, new_w, c))?;
        // x2 = x[:, 0::2, 1::2, :] = x_reshaped[:, :, 0, :, 1, :]
        let x2: Tensor = x_reshaped
            .narrow(2, 0, 1)?
            .narrow(4, 1, 1)?
            .reshape((b, new_h, new_w, c))?;
        // x3 = x[:, 1::2, 1::2, :] = x_reshaped[:, :, 1, :, 1, :]
        let x3: Tensor = x_reshaped
            .narrow(2, 1, 1)?
            .narrow(4, 1, 1)?
            .reshape((b, new_h, new_w, c))?;
        let x: Tensor = Tensor::cat(&[&x0, &x1, &x2, &x3], 3)?; // [B, new_h, new_w, 4*C]
        let x: Tensor = x.reshape((b, new_h * new_w, 4 * c))?;

        let x: Tensor = self.norm.forward(&x)?;
        let x: Tensor = self.reduction.forward(&x)?;

        Ok((x, new_h, new_w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_patch_merging_output_shape() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let dim: usize = 96;
        let pm = PatchMerging::new(dim, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 56 * 56, dim), device)?;
        let (output, new_h, new_w) = pm.forward(&input, 56, 56)?;

        assert_eq!(output.dims(), &[1, 28 * 28, 2 * dim]);
        assert_eq!(new_h, 28);
        assert_eq!(new_w, 28);
        Ok(())
    }

    #[test]
    fn test_patch_merging_odd_dimensions() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let dim: usize = 96;
        let pm = PatchMerging::new(dim, vb)?;
        // Odd spatial dimension: 57x57 -> pads to 58x58 -> output 29x29
        let input = Tensor::randn(0f32, 1.0, (1, 57 * 57, dim), device)?;
        let (output, new_h, new_w) = pm.forward(&input, 57, 57)?;

        assert_eq!(new_h, 29);
        assert_eq!(new_w, 29);
        assert_eq!(output.dims(), &[1, 29 * 29, 2 * dim]);
        Ok(())
    }
}
