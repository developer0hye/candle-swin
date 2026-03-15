use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, VarBuilder};

/// Image to Patch Embedding.
///
/// Projects input image `[B, C, H, W]` into patch tokens using a non-overlapping
/// convolution with `kernel_size = stride = patch_size`, then optionally applies LayerNorm.
///
/// Reference: Swin Transformer (https://arxiv.org/abs/2103.14030), Section 3.1
pub struct PatchEmbed {
    proj: Conv2d,
    norm: Option<LayerNorm>,
    patch_size: (usize, usize),
}

impl PatchEmbed {
    pub fn new(
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        use_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_channels, embed_dim, patch_size, config, vb.pp("proj"))?;
        let norm = if use_norm {
            Some(candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("norm"))?)
        } else {
            None
        };
        Ok(Self {
            proj,
            norm,
            patch_size: (patch_size, patch_size),
        })
    }

    /// Forward pass.
    ///
    /// Input: `[B, C, H, W]`
    /// Output: `[B, embed_dim, Wh, Ww]` where `Wh = ceil(H / patch_size)`, `Ww = ceil(W / patch_size)`
    ///
    /// Pads input if spatial dimensions are not divisible by patch_size.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = x.dims4()?;

        // Pad to multiples of patch_size
        let pad_h: usize = (self.patch_size.0 - h % self.patch_size.0) % self.patch_size.0;
        let pad_w: usize = (self.patch_size.1 - w % self.patch_size.1) % self.patch_size.1;
        let x: Tensor = if pad_h > 0 || pad_w > 0 {
            x.pad_with_zeros(3, 0, pad_w)?.pad_with_zeros(2, 0, pad_h)?
        } else {
            x.clone()
        };

        let x: Tensor = self.proj.forward(&x)?;

        let x: Tensor = if let Some(ref norm) = self.norm {
            let (b, c, wh, ww) = x.dims4()?;
            // [B, C, Wh, Ww] -> [B, Wh*Ww, C] -> LayerNorm -> [B, C, Wh, Ww]
            let x: Tensor = x.flatten_from(2)?.transpose(1, 2)?;
            let x: Tensor = norm.forward(&x)?;
            x.transpose(1, 2)?.reshape((b, c, wh, ww))?
        } else {
            x
        };

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_patch_embed_output_shape() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let patch_embed = PatchEmbed::new(4, 3, 96, true, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 3, 224, 224), device)?;
        let output = patch_embed.forward(&input)?;

        assert_eq!(output.dims(), &[1, 96, 56, 56]);
        Ok(())
    }

    #[test]
    fn test_patch_embed_non_divisible_input() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let patch_embed = PatchEmbed::new(4, 3, 96, true, vb)?;
        // 225 is not divisible by 4 -> should pad to 228, output = 57
        let input = Tensor::randn(0f32, 1.0, (1, 3, 225, 225), device)?;
        let output = patch_embed.forward(&input)?;

        assert_eq!(output.dims(), &[1, 96, 57, 57]);
        Ok(())
    }

    #[test]
    fn test_patch_embed_without_norm() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let patch_embed = PatchEmbed::new(4, 3, 96, false, vb)?;
        let input = Tensor::randn(0f32, 1.0, (1, 3, 224, 224), device)?;
        let output = patch_embed.forward(&input)?;

        assert_eq!(output.dims(), &[1, 96, 56, 56]);
        Ok(())
    }
}
