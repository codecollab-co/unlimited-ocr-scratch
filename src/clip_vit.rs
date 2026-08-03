//! CLIP-ViT-L/14 encoder (the second half of Unlimited-OCR's DeepEncoder).
//!
//! Clean-room reimplementation matching config.json's vision_config.width.clip-l-14-224:
//! width=1024, layers=24, heads=16, patch_size=14, image_size=224.
//!
//! Unlike a standalone CLIP encoder, this one never patchifies pixels itself: the
//! reference (`modeling_unlimitedocr.py`, ~line 499-501) always calls
//! `vision_model(patches, local_features_1)` where `local_features_1` is SAM-ViT-B's
//! own output (B, 1024, 16, 16). CLIP's own conv patch embedding is dead code on this
//! path — it just prepends a CLS token to SAM's 256 tokens, adds position embeddings,
//! and runs 24 layers of standard *global* (non-windowed) self-attention on top. This
//! is the "cascading window attention ViT and global attention one" from the paper's
//! section 2.2.1: SAM supplies cheap high-res local features, CLIP adds global context.

use candle_core::{Result, Tensor, D};
use candle_nn::{layer_norm, linear, ops, LayerNorm, Linear, Module, VarBuilder};

fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    (x * ops::sigmoid(&(x * 1.702)?)?)?.to_dtype(x.dtype())
}

struct ClipAttention {
    num_heads: usize,
    head_dim: usize,
    qkv_proj: Linear,
    out_proj: Linear,
}

impl ClipAttention {
    fn new(hidden: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            num_heads,
            head_dim: hidden / num_heads,
            qkv_proj: linear(hidden, hidden * 3, vb.pp("qkv_proj"))?,
            out_proj: linear(hidden, hidden, vb.pp("out_proj"))?,
        })
    }

    /// Standard global (unmasked) multi-head self-attention — no windowing, no relative
    /// position bias, unlike SAM's blocks.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qkv = self
            .qkv_proj
            .forward(x)?
            .reshape((b, t, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn = ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?; // (B, heads, T, head_dim)

        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

struct ClipBlock {
    layer_norm1: LayerNorm,
    self_attn: ClipAttention,
    layer_norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl ClipBlock {
    fn new(hidden: usize, num_heads: usize, ffn_hidden: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            layer_norm1: layer_norm(hidden, eps, vb.pp("layer_norm1"))?,
            self_attn: ClipAttention::new(hidden, num_heads, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(hidden, eps, vb.pp("layer_norm2"))?,
            fc1: linear(hidden, ffn_hidden, vb.pp("mlp.fc1"))?,
            fc2: linear(ffn_hidden, hidden, vb.pp("mlp.fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = self.self_attn.forward(&self.layer_norm1.forward(x)?)?;
        let h = (x + residual)?;
        let mlp_out = self.fc2.forward(&quick_gelu(&self.fc1.forward(&self.layer_norm2.forward(&h)?)?)?)?;
        h + mlp_out
    }
}

pub struct ClipVitL {
    class_embedding: Tensor,      // (hidden,)
    position_embedding: Tensor,   // (num_positions, hidden) — always used at native size, see module docs
    pre_layrnorm: LayerNorm,      // name matches the reference's typo'd attribute for weight-key compatibility
    blocks: Vec<ClipBlock>,
}

impl ClipVitL {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Self::with_config(1024, 24, 16, 4096, 224, 14, 1e-5, vb)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        hidden: usize,
        depth: usize,
        num_heads: usize,
        ffn_hidden: usize,
        image_size: usize,
        patch_size: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_positions = (image_size / patch_size).pow(2) + 1;
        let class_embedding = vb.get_with_hints(hidden, "embeddings.class_embedding", candle_nn::Init::Const(0.))?;
        let position_embedding = vb.get_with_hints(
            (num_positions, hidden),
            "embeddings.position_embedding.weight",
            candle_nn::Init::Const(0.),
        )?;
        let pre_layrnorm = layer_norm(hidden, eps, vb.pp("pre_layrnorm"))?;
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            blocks.push(ClipBlock::new(hidden, num_heads, ffn_hidden, eps, vb.pp(format!("transformer.layers.{i}")))?);
        }
        Ok(Self { class_embedding, position_embedding, pre_layrnorm, blocks })
    }

    /// patch_embeds: (B, hidden, grid, grid) — SAM-ViT-B's output, used directly as CLIP's
    /// patch tokens. Returns (B, grid*grid + 1, hidden) including the CLS token at index 0.
    pub fn forward(&self, patch_embeds: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = patch_embeds.dims4()?;
        let patches = patch_embeds.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?; // (B, HW, C)
        let cls = self.class_embedding.reshape((1, 1, c))?.broadcast_as((b, 1, c))?;
        let embeddings = Tensor::cat(&[&cls, &patches], 1)?; // (B, HW+1, C)
        let embeddings = embeddings.broadcast_add(&self.position_embedding.reshape((1, h * w + 1, c))?)?;

        let mut x = self.pre_layrnorm.forward(&embeddings)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn forward_shape_and_cls_token_present() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // Tiny config: hidden=8, depth=2, heads=2, grid 4x4 (image 8 / patch 2) -> 16 patches + CLS.
        let model = ClipVitL::with_config(8, 2, 2, 16, 8, 2, 1e-5, vb)?;

        let sam_features = Tensor::randn(0f32, 1f32, (2, 8, 4, 4), &device)?;
        let out = model.forward(&sam_features)?;
        assert_eq!(out.dims(), &[2, 17, 8], "expected CLS + 16 patch tokens");
        Ok(())
    }
}
