use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Dropout, LayerNorm, VarBuilder};

use crate::patch_embed::PatchEmbed;
use crate::patch_merging::PatchMerging;
use crate::swin_transformer_block::SwinTransformerBlock;

/// A single Swin Transformer stage containing multiple SwinTransformerBlocks
/// and an optional PatchMerging downsample layer.
struct BasicLayer {
    blocks: Vec<SwinTransformerBlock>,
    downsample: Option<PatchMerging>,
    window_size: usize,
    shift_size: usize,
}

impl BasicLayer {
    fn new(
        dim: usize,
        depth: usize,
        num_heads: usize,
        window_size: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        drop: f32,
        attn_drop: f32,
        use_downsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let shift_size: usize = window_size / 2;
        let mut blocks: Vec<SwinTransformerBlock> = Vec::with_capacity(depth);
        for i in 0..depth {
            let block_shift: usize = if i % 2 == 0 { 0 } else { shift_size };
            blocks.push(SwinTransformerBlock::new(
                dim,
                num_heads,
                window_size,
                block_shift,
                mlp_ratio,
                qkv_bias,
                drop,
                attn_drop,
                vb.pp(format!("blocks.{}", i)),
            )?);
        }
        let downsample: Option<PatchMerging> = if use_downsample {
            Some(PatchMerging::new(dim, vb.pp("downsample"))?)
        } else {
            None
        };
        Ok(Self {
            blocks,
            downsample,
            window_size,
            shift_size,
        })
    }

    /// Build the attention mask for shifted window attention.
    ///
    /// The mask ensures tokens from different spatial regions (after cyclic shift)
    /// don't attend to each other. Values are 0.0 (attend) or -100.0 (block).
    fn build_attn_mask(
        &self,
        h: usize,
        w: usize,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let ws: usize = self.window_size;
        let ss: usize = self.shift_size;

        // Compute padded dimensions
        let hp: usize = ((h + ws - 1) / ws) * ws;
        let wp: usize = ((w + ws - 1) / ws) * ws;

        // Build region index image [1, Hp, Wp, 1]
        // Each region gets a unique integer label based on slice membership
        let mut img_mask_data: Vec<f32> = vec![0.0; hp * wp];
        let h_slices: [(usize, usize); 3] = [
            (0, hp.saturating_sub(ws)),
            (hp.saturating_sub(ws), hp.saturating_sub(ss)),
            (hp.saturating_sub(ss), hp),
        ];
        let w_slices: [(usize, usize); 3] = [
            (0, wp.saturating_sub(ws)),
            (wp.saturating_sub(ws), wp.saturating_sub(ss)),
            (wp.saturating_sub(ss), wp),
        ];
        let mut cnt: f32 = 0.0;
        for &(h_start, h_end) in &h_slices {
            for &(w_start, w_end) in &w_slices {
                for row in h_start..h_end {
                    for col in w_start..w_end {
                        img_mask_data[row * wp + col] = cnt;
                    }
                }
                cnt += 1.0;
            }
        }

        let img_mask: Tensor = Tensor::from_vec(img_mask_data, (1, hp, wp, 1), device)?;

        // Partition into windows: [nW, ws, ws, 1]
        let mask_windows: Tensor = crate::swin_transformer_block::window_partition(&img_mask, ws)?;
        let n: usize = ws * ws;
        let n_windows: usize = mask_windows.dims()[0];
        let mask_windows: Tensor = mask_windows.reshape((n_windows, n))?;

        // Pairwise difference: same region -> 0, different region -> nonzero
        let attn_mask: Tensor = mask_windows
            .unsqueeze(1)?
            .broadcast_sub(&mask_windows.unsqueeze(2)?)?; // [nW, N, N]

        // Replace nonzero with -100.0, zero with 0.0
        let is_nonzero: Tensor = attn_mask.ne(0.0f32)?;
        let neg_100: Tensor = Tensor::full(-100.0f32, attn_mask.shape(), device)?;
        let zeros: Tensor = Tensor::zeros(attn_mask.shape(), DType::F32, device)?;
        let attn_mask: Tensor = is_nonzero.where_cond(&neg_100, &zeros)?;
        let attn_mask: Tensor = attn_mask.to_dtype(dtype)?;

        Ok(attn_mask)
    }

    /// Forward pass through the layer.
    ///
    /// Returns `(x_out, h, w, x_down, new_h, new_w)`:
    /// - `x_out`: output before downsampling (for skip connections)
    /// - `x_down`, `new_h`, `new_w`: output after downsampling (or same as x_out if no downsample)
    fn forward(
        &self,
        x: &Tensor,
        h: usize,
        w: usize,
    ) -> Result<(Tensor, usize, usize, Tensor, usize, usize)> {
        let attn_mask: Tensor = self.build_attn_mask(h, w, x.dtype(), x.device())?;

        let mut x: Tensor = x.clone();
        for block in &self.blocks {
            x = block.forward(&x, h, w, Some(&attn_mask))?;
        }

        let x_out: Tensor = x.clone();
        if let Some(ref downsample) = self.downsample {
            let (x_down, new_h, new_w) = downsample.forward(&x, h, w)?;
            Ok((x_out, h, w, x_down, new_h, new_w))
        } else {
            Ok((x_out, h, w, x, h, w))
        }
    }
}

/// Swin Transformer V1 backbone.
///
/// Hierarchical vision transformer using shifted windows. Produces multi-scale
/// feature maps for downstream tasks (segmentation, detection, etc.).
///
/// Reference: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
/// https://arxiv.org/abs/2103.14030
pub struct SwinTransformer {
    patch_embed: PatchEmbed,
    pos_drop: Dropout,
    layers: Vec<BasicLayer>,
    norms: Vec<LayerNorm>,
    out_indices: Vec<usize>,
    num_features: Vec<usize>,
    embed_dim: usize,
}

/// Configuration for SwinTransformer model variants.
pub struct SwinTransformerConfig {
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub depths: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub window_size: usize,
    pub mlp_ratio: f64,
    pub qkv_bias: bool,
    pub drop_rate: f32,
    pub attn_drop_rate: f32,
    pub out_indices: Vec<usize>,
}

impl SwinTransformerConfig {
    /// Swin-Tiny: embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7
    pub fn tiny() -> Self {
        Self {
            patch_size: 4,
            in_channels: 3,
            embed_dim: 96,
            depths: vec![2, 2, 6, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            out_indices: vec![0, 1, 2, 3],
        }
    }

    /// Swin-Small: embed_dim=96, depths=[2,2,18,2], num_heads=[3,6,12,24], window_size=7
    pub fn small() -> Self {
        Self {
            depths: vec![2, 2, 18, 2],
            ..Self::tiny()
        }
    }

    /// Swin-Base: embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=12
    pub fn base() -> Self {
        Self {
            embed_dim: 128,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![4, 8, 16, 32],
            window_size: 12,
            ..Self::tiny()
        }
    }

    /// Swin-Large: embed_dim=192, depths=[2,2,18,2], num_heads=[6,12,24,48], window_size=12
    /// This is the default backbone for BiRefNet.
    pub fn large() -> Self {
        Self {
            embed_dim: 192,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![6, 12, 24, 48],
            window_size: 12,
            ..Self::tiny()
        }
    }
}

impl SwinTransformer {
    pub fn new(config: &SwinTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let num_layers: usize = config.depths.len();

        let patch_embed: PatchEmbed = PatchEmbed::new(
            config.patch_size,
            config.in_channels,
            config.embed_dim,
            true, // patch_norm
            vb.pp("patch_embed"),
        )?;

        let pos_drop: Dropout = Dropout::new(config.drop_rate);

        let num_features: Vec<usize> = (0..num_layers)
            .map(|i| config.embed_dim * (1 << i))
            .collect();

        let mut layers: Vec<BasicLayer> = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let dim: usize = num_features[i];
            let use_downsample: bool = i < num_layers - 1;
            layers.push(BasicLayer::new(
                dim,
                config.depths[i],
                config.num_heads[i],
                config.window_size,
                config.mlp_ratio,
                config.qkv_bias,
                config.drop_rate,
                config.attn_drop_rate,
                use_downsample,
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        let mut norms: Vec<LayerNorm> = Vec::with_capacity(config.out_indices.len());
        for &i in &config.out_indices {
            norms.push(candle_nn::layer_norm(
                num_features[i],
                1e-5,
                vb.pp(format!("norm{}", i)),
            )?);
        }

        Ok(Self {
            patch_embed,
            pos_drop,
            layers,
            norms,
            out_indices: config.out_indices.clone(),
            num_features,
            embed_dim: config.embed_dim,
        })
    }

    /// Forward pass.
    ///
    /// Input: `[B, 3, H, W]`
    /// Output: `Vec<Tensor>` — multi-scale feature maps `[B, C_i, H_i, W_i]` for each output stage.
    ///
    /// For Swin-Large with 224x224 input:
    /// - Stage 0: `[B, 192, 56, 56]`
    /// - Stage 1: `[B, 384, 28, 28]`
    /// - Stage 2: `[B, 768, 14, 14]`
    /// - Stage 3: `[B, 1536, 7, 7]`
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let x: Tensor = self.patch_embed.forward(x)?;
        let (_, _, wh, ww) = x.dims4()?;

        // [B, C, Wh, Ww] -> [B, Wh*Ww, C]
        let x: Tensor = x.flatten_from(2)?.transpose(1, 2)?;
        let x: Tensor = self.pos_drop.forward(&x, false)?;

        let mut outs: Vec<Tensor> = Vec::with_capacity(self.out_indices.len());
        let mut current_x: Tensor = x;
        let mut current_h: usize = wh;
        let mut current_w: usize = ww;

        for (i, layer) in self.layers.iter().enumerate() {
            let (x_out, h, w, x_down, new_h, new_w) =
                layer.forward(&current_x, current_h, current_w)?;

            if let Some(norm_idx) = self.out_indices.iter().position(|&idx| idx == i) {
                let norm: &LayerNorm = &self.norms[norm_idx];
                let x_normed: Tensor = norm.forward(&x_out)?;
                // [B, H*W, C] -> [B, H, W, C] -> [B, C, H, W]
                let out: Tensor = x_normed
                    .reshape(((), h, w, self.num_features[i]))?
                    .permute((0, 3, 1, 2))?
                    .contiguous()?;
                outs.push(out);
            }

            current_x = x_down;
            current_h = new_h;
            current_w = new_w;
        }

        Ok(outs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_swin_tiny_output_shapes() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let config = SwinTransformerConfig::tiny();
        let model = SwinTransformer::new(&config, vb)?;

        let input = Tensor::randn(0f32, 1.0, (1, 3, 224, 224), device)?;
        let outputs: Vec<Tensor> = model.forward(&input)?;

        assert_eq!(outputs.len(), 4);
        assert_eq!(outputs[0].dims(), &[1, 96, 56, 56]);
        assert_eq!(outputs[1].dims(), &[1, 192, 28, 28]);
        assert_eq!(outputs[2].dims(), &[1, 384, 14, 14]);
        assert_eq!(outputs[3].dims(), &[1, 768, 7, 7]);
        Ok(())
    }

    #[test]
    fn test_swin_large_output_shapes() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let config = SwinTransformerConfig::large();
        let model = SwinTransformer::new(&config, vb)?;

        let input = Tensor::randn(0f32, 1.0, (1, 3, 224, 224), device)?;
        let outputs: Vec<Tensor> = model.forward(&input)?;

        assert_eq!(outputs.len(), 4);
        assert_eq!(outputs[0].dims(), &[1, 192, 56, 56]);
        assert_eq!(outputs[1].dims(), &[1, 384, 28, 28]);
        assert_eq!(outputs[2].dims(), &[1, 768, 14, 14]);
        assert_eq!(outputs[3].dims(), &[1, 1536, 7, 7]);
        Ok(())
    }

    #[test]
    fn test_swin_tiny_non_standard_input() -> Result<()> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let config = SwinTransformerConfig::tiny();
        let model = SwinTransformer::new(&config, vb)?;

        // Non-standard size: 384x384
        let input = Tensor::randn(0f32, 1.0, (1, 3, 384, 384), device)?;
        let outputs: Vec<Tensor> = model.forward(&input)?;

        assert_eq!(outputs.len(), 4);
        assert_eq!(outputs[0].dims(), &[1, 96, 96, 96]);
        assert_eq!(outputs[1].dims(), &[1, 192, 48, 48]);
        assert_eq!(outputs[2].dims(), &[1, 384, 24, 24]);
        assert_eq!(outputs[3].dims(), &[1, 768, 12, 12]);
        Ok(())
    }
}
