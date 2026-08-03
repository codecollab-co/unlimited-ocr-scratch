//! SAM-ViT-B image encoder (the first half of Unlimited-OCR's DeepEncoder).
//!
//! Clean-room reimplementation matching Meta's Segment Anything ViT-B backbone,
//! as configured in Unlimited-OCR's config.json (vision_config.width.sam_vit_b):
//! width=768, layers=12, heads=12, window_size=14, global_attn_indexes=[2,5,8,11],
//! downsample_channels=[512, 1024], img_size=1024, patch_size=16.
//!
//! Pipeline: patchify (1024x1024 -> 64x64 grid of 768-dim tokens) -> 12 transformer
//! blocks (windowed attention with relative position bias, except 4 global-attention
//! blocks) -> neck (1x1 conv -> 3x3 conv, down to 256 channels) -> two stride-2 convs
//! (256->512->1024) giving a final 16x16x1024 feature map. Module/parameter names
//! match the reference state_dict layout so real weights can be loaded directly.

use candle_core::{Result, Tensor, D};
use candle_nn::{conv2d, conv2d_no_bias, layer_norm, linear, ops, Conv2d, Conv2dConfig, Init, LayerNorm, Linear, Module, VarBuilder};

struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    fn new(patch_size: usize, in_chans: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { stride: patch_size, ..Default::default() };
        let proj = conv2d(in_chans, embed_dim, patch_size, cfg, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    /// [B, 3, H, W] -> [B, H/patch, W/patch, C]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)?.permute((0, 2, 3, 1))?.contiguous()
    }
}

/// Bicubic resize isn't in candle; unused for this model's fixed 1024x1024/patch16
/// config (the trained grid size always matches the input grid size), so this is
/// only a same-size passthrough. Would need a real interpolation impl if the model
/// were ever run at a resolution other than what it was trained on.
fn interpolate_abs_pos(pos_embed: &Tensor, tgt_size: usize) -> Result<Tensor> {
    if pos_embed.dim(1)? == tgt_size {
        return Ok(pos_embed.clone());
    }
    candle_core::bail!("abs-pos interpolation for a non-native grid size is not implemented (candle has no bicubic upsample)");
}

/// [B, H, W, C] -> [B * num_windows, window_size, window_size, C], zero-padding H/W up to a window multiple.
fn window_partition(x: &Tensor, window_size: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = x.dims4()?;
    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;
    let x = if pad_h > 0 || pad_w > 0 {
        x.pad_with_zeros(1, 0, pad_h)?.pad_with_zeros(2, 0, pad_w)?
    } else {
        x.clone()
    };
    let (hp, wp) = (h + pad_h, w + pad_w);
    let x = x.reshape((b, hp / window_size, window_size, wp / window_size, window_size, c))?;
    let windows = x
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b * (hp / window_size) * (wp / window_size), window_size, window_size, c))?;
    Ok((windows, (hp, wp)))
}

/// Inverse of window_partition: reassemble windows and strip padding back to (h, w).
fn window_unpartition(windows: &Tensor, window_size: usize, pad_hw: (usize, usize), hw: (usize, usize)) -> Result<Tensor> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let c = windows.dim(D::Minus1)?;
    let b = windows.dim(0)? / (hp * wp / window_size / window_size);
    let x = windows.reshape((b, hp / window_size, wp / window_size, window_size, window_size, c))?;
    let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b, hp, wp, c))?;
    if hp > h || wp > w {
        x.narrow(1, 0, h)?.narrow(2, 0, w)?.contiguous()
    } else {
        Ok(x)
    }
}

/// Look up the relative position embedding for every (query, key) offset pair.
/// Assumes q_size == k_size (always true here — plain self-attention), so unlike
/// the reference we never need the resize-table branch (rel_pos is always sized
/// exactly 2*size-1 for the size it was created at).
fn get_rel_pos(size: usize, rel_pos: &Tensor) -> Result<Tensor> {
    let head_dim = rel_pos.dim(1)?;
    let device = rel_pos.device();
    let mut idx = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            idx.push((i as i64 - j as i64 + (size as i64 - 1)) as u32);
        }
    }
    let idx = Tensor::from_vec(idx, size * size, device)?;
    rel_pos.index_select(&idx, 0)?.reshape((size, size, head_dim))
}

/// Additive attention bias from separable height/width relative position embeddings
/// (cheaper than a full 2D relative table: O(h*w) params instead of O(h^2*w^2)).
/// `q` is already flattened to (batch*heads, h*w, head_dim).
fn decomposed_rel_pos_bias(q: &Tensor, rel_pos_h: &Tensor, rel_pos_w: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let b = q.dim(0)?;
    let r_q = q.reshape((b, h, w, head_dim))?;

    let rh = get_rel_pos(h, rel_pos_h)?; // (h, h, head_dim)
    let rw = get_rel_pos(w, rel_pos_w)?; // (w, w, head_dim)

    let r_q_h = r_q.permute((1, 0, 2, 3))?.contiguous()?.reshape((h, b * w, head_dim))?;
    let rh_t = rh.transpose(1, 2)?.contiguous()?; // (h, head_dim, h)
    let rel_h = r_q_h.matmul(&rh_t)?.reshape((h, b, w, h))?.permute((1, 0, 2, 3))?; // (b, h, w, h)

    let r_q_w = r_q.permute((2, 0, 1, 3))?.contiguous()?.reshape((w, b * h, head_dim))?;
    let rw_t = rw.transpose(1, 2)?.contiguous()?; // (w, head_dim, w)
    let rel_w = r_q_w.matmul(&rw_t)?.reshape((w, b, h, w))?.permute((1, 2, 0, 3))?; // (b, h, w, w)

    let bias = rel_h.unsqueeze(4)?.broadcast_add(&rel_w.unsqueeze(3)?)?; // (b, h, w, h, w)
    bias.reshape((b, h * w, h * w))
}

struct WindowAttention {
    num_heads: usize,
    head_dim: usize,
    qkv: Linear,
    proj: Linear,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
}

impl WindowAttention {
    fn new(dim: usize, num_heads: usize, grid: (usize, usize), vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = linear(dim, dim, vb.pp("proj"))?;
        let rel_pos_h = vb.get_with_hints((2 * grid.0 - 1, head_dim), "rel_pos_h", Init::Const(0.))?;
        let rel_pos_w = vb.get_with_hints((2 * grid.1 - 1, head_dim), "rel_pos_w", Init::Const(0.))?;
        Ok(Self { num_heads, head_dim, qkv, proj, rel_pos_h, rel_pos_w })
    }

    /// [B, H, W, C] -> [B, H, W, C]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, _) = x.dims4()?;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, h * w, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?
            .reshape((3, b * self.num_heads, h * w, self.head_dim))?;
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;

        let bias = decomposed_rel_pos_bias(&q, &self.rel_pos_h, &self.rel_pos_w, h, w)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn = ops::softmax(&(attn + bias)?, D::Minus1)?;
        let out = attn.matmul(&v)?; // (b*heads, h*w, head_dim)

        let out = out
            .reshape((b, self.num_heads, h, w, self.head_dim))?
            .permute((0, 2, 3, 1, 4))?
            .contiguous()?
            .reshape((b, h, w, self.num_heads * self.head_dim))?;
        self.proj.forward(&out)
    }
}

struct Block {
    window_size: usize,
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp_lin1: Linear,
    mlp_lin2: Linear,
}

impl Block {
    fn new(dim: usize, num_heads: usize, mlp_ratio: f64, window_size: usize, grid: (usize, usize), vb: VarBuilder) -> Result<Self> {
        let attn_grid = if window_size > 0 { (window_size, window_size) } else { grid };
        let hidden = (dim as f64 * mlp_ratio) as usize;
        Ok(Self {
            window_size,
            norm1: layer_norm(dim, 1e-6, vb.pp("norm1"))?,
            attn: WindowAttention::new(dim, num_heads, attn_grid, vb.pp("attn"))?,
            norm2: layer_norm(dim, 1e-6, vb.pp("norm2"))?,
            mlp_lin1: linear(dim, hidden, vb.pp("mlp.lin1"))?,
            mlp_lin2: linear(hidden, dim, vb.pp("mlp.lin2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();
        let x = self.norm1.forward(x)?;
        let (x, pad_hw, hw) = if self.window_size > 0 {
            let hw = (x.dim(1)?, x.dim(2)?);
            let (windows, pad_hw) = window_partition(&x, self.window_size)?;
            (windows, Some(pad_hw), Some(hw))
        } else {
            (x, None, None)
        };
        let x = self.attn.forward(&x)?;
        let x = match (pad_hw, hw) {
            (Some(pad_hw), Some(hw)) => window_unpartition(&x, self.window_size, pad_hw, hw)?,
            _ => x,
        };
        let x = (shortcut + x)?;
        let mlp_out = self.mlp_lin2.forward(&self.mlp_lin1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        x + mlp_out
    }
}

/// LayerNorm over the channel dim of a [B, C, H, W] tensor (ConvNeXt-style).
struct LayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm2d {
    fn new(num_channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get_with_hints(num_channels, "weight", Init::Const(1.))?,
            bias: vb.get_with_hints(num_channels, "bias", Init::Const(0.))?,
            eps: 1e-6,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let u = x.mean_keepdim(1)?;
        let xm = x.broadcast_sub(&u)?;
        let s = xm.sqr()?.mean_keepdim(1)?;
        let x = xm.broadcast_div(&(s + self.eps)?.sqrt()?)?;
        let (c,) = self.weight.dims1().map(|d| (d,))?;
        x.broadcast_mul(&self.weight.reshape((c, 1, 1))?)?
            .broadcast_add(&self.bias.reshape((c, 1, 1))?)
    }
}

pub struct SamVitB {
    patch_embed: PatchEmbed,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    neck_conv1: Conv2d,
    neck_norm1: LayerNorm2d,
    neck_conv2: Conv2d,
    neck_norm2: LayerNorm2d,
    net_2: Conv2d,
    net_3: Conv2d,
}

impl SamVitB {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Self::with_config(1024, 16, 768, 12, 12, 4.0, 14, &[2, 5, 8, 11], 256, (512, 1024), vb)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        img_size: usize,
        patch_size: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f64,
        window_size: usize,
        global_attn_indexes: &[usize],
        neck_chans: usize,
        downsample_channels: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let grid = img_size / patch_size; // 64
        let patch_embed = PatchEmbed::new(patch_size, 3, embed_dim, vb.pp("patch_embed"))?;
        let pos_embed = vb.get_with_hints((1, grid, grid, embed_dim), "pos_embed", Init::Const(0.))?;

        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let w = if global_attn_indexes.contains(&i) { 0 } else { window_size };
            blocks.push(Block::new(embed_dim, num_heads, mlp_ratio, w, (grid, grid), vb.pp(format!("blocks.{i}")))?);
        }

        let neck_conv1 = conv2d_no_bias(embed_dim, neck_chans, 1, Conv2dConfig::default(), vb.pp("neck.0"))?;
        let neck_norm1 = LayerNorm2d::new(neck_chans, vb.pp("neck.1"))?;
        let cfg3x3 = Conv2dConfig { padding: 1, ..Default::default() };
        let neck_conv2 = conv2d_no_bias(neck_chans, neck_chans, 3, cfg3x3, vb.pp("neck.2"))?;
        let neck_norm2 = LayerNorm2d::new(neck_chans, vb.pp("neck.3"))?;

        let (c1, c2) = downsample_channels;
        let stride2 = Conv2dConfig { padding: 1, stride: 2, ..Default::default() };
        let net_2 = conv2d_no_bias(neck_chans, c1, 3, stride2, vb.pp("net_2"))?;
        let net_3 = conv2d_no_bias(c1, c2, 3, stride2, vb.pp("net_3"))?;

        Ok(Self { patch_embed, pos_embed, blocks, neck_conv1, neck_norm1, neck_conv2, neck_norm2, net_2, net_3 })
    }

    /// [B, 3, 1024, 1024] -> [B, 1024, 16, 16]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(x)?;
        let pos = interpolate_abs_pos(&self.pos_embed, x.dim(1)?)?;
        x = x.broadcast_add(&pos)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = x.permute((0, 3, 1, 2))?.contiguous()?; // B,H,W,C -> B,C,H,W
        let x = self.neck_norm1.forward(&self.neck_conv1.forward(&x)?)?;
        let x = self.neck_norm2.forward(&self.neck_conv2.forward(&x)?)?;
        let x = self.net_2.forward(&x)?;
        self.net_3.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn window_partition_unpartition_round_trip() -> Result<()> {
        let device = Device::Cpu;
        // Non-multiple-of-window-size grid to exercise the padding path too.
        let x = Tensor::randn(0f32, 1f32, (2, 20, 23, 5), &device)?;
        let (windows, pad_hw) = window_partition(&x, 7)?;
        let restored = window_unpartition(&windows, 7, pad_hw, (20, 23))?;
        let diff: f32 = (restored - &x)?.abs()?.max_all()?.to_scalar()?;
        assert!(diff < 1e-6, "window partition/unpartition round trip mismatch: {diff}");
        Ok(())
    }

    #[test]
    fn forward_shape_and_global_attention_placement() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // Tiny config so the CPU test runs fast: grid=4 (img 8 / patch 2), depth=4, one window block + one global.
        let model = SamVitB::with_config(8, 2, 16, 4, 4, 4.0, 2, &[1, 3], 8, (12, 16), vb)?;

        assert_eq!(model.blocks[0].window_size, 2, "block 0 should use windowed attention");
        assert_eq!(model.blocks[1].window_size, 0, "block 1 is in global_attn_indexes");
        assert_eq!(model.blocks[2].window_size, 2, "block 2 should use windowed attention");
        assert_eq!(model.blocks[3].window_size, 0, "block 3 is in global_attn_indexes");

        let x = Tensor::randn(0f32, 1f32, (1, 3, 8, 8), &device)?;
        let out = model.forward(&x)?;
        // grid=4 -> neck keeps 4x4 -> net_2 stride2 -> 2x2 -> net_3 stride2 -> 1x1
        assert_eq!(out.dims(), &[1, 16, 1, 1]);
        Ok(())
    }
}
