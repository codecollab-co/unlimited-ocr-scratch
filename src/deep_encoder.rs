//! DeepEncoder: the full vision pipeline feeding Unlimited-OCR's decoder.
//!
//! Wiring matches `modeling_unlimitedocr.py` (~line 499-504):
//!   local_features_1 = sam_model(patches)                                  # (B, 1024, 16, 16)
//!   local_features_2 = vision_model(patches, local_features_1)             # (B, 257, 1024), CLS + 256 patches
//!   local_features = cat([local_features_2[:, 1:], local_features_1.flatten(2).permute(0,2,1)], dim=-1)
//!   local_features = projector(local_features)                            # (B, 256, 2048) -> (B, 256, 1280)
//!
//! i.e. SAM supplies cheap high-res local features (last dim of the concat), CLIP
//! adds global semantic context on top of those same 256 spatial positions (its CLS
//! token is dropped — it's not part of the per-patch token stream fed to the decoder).
//! This is the paper's 16x compression: a 1024x1024 image, patchified at 16px by SAM
//! (64x64 = 4096 tokens), is spatially downsampled by SAM's neck to 16x16 = 256 tokens
//! before CLIP or the projector ever see it.

use candle_core::{Result, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::clip_vit::ClipVitL;
use crate::sam_vit::SamVitB;

pub struct DeepEncoder {
    sam: SamVitB,
    clip: ClipVitL,
    projector: Linear, // 2048 -> 1280, MlpProjector's "linear" variant
}

impl DeepEncoder {
    /// vb must be the checkpoint's ROOT VarBuilder — the real weight file nests these
    /// three under "model." (verified against model.safetensors.index.json), matching
    /// MoeDecoder::new's own "model.embed_tokens" / "model.layers.N" / "model.norm" paths.
    pub fn new(n_embed: usize, vb: VarBuilder) -> Result<Self> {
        let sam = SamVitB::new(vb.pp("model.sam_model"))?;
        let clip = ClipVitL::new(vb.pp("model.vision_model"))?;
        let concat_dim = 1024 + 1024; // SAM out_chans=256 -> net_3=1024, concat with CLIP hidden=1024
        let projector = linear(concat_dim, n_embed, vb.pp("model.projector.layers"))?;
        Ok(Self { sam, clip, projector })
    }

    /// Wire a pre-built SAM/CLIP pair together — lets tests use tiny configs (already
    /// exercised individually by sam_vit's and clip_vit's own tests) without paying for
    /// a full-resolution forward pass just to check the concat/projector wiring.
    pub fn from_parts(sam: SamVitB, clip: ClipVitL, sam_chans: usize, clip_hidden: usize, n_embed: usize, vb: VarBuilder) -> Result<Self> {
        let projector = linear(sam_chans + clip_hidden, n_embed, vb.pp("projector.layers"))?;
        Ok(Self { sam, clip, projector })
    }

    /// pixel_values: (B, 3, 1024, 1024) -> (B, 256, n_embed) vision tokens ready for the decoder.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let sam_features = self.sam.forward(pixel_values)?; // (B, 1024, 16, 16)
        let clip_features = self.clip.forward(&sam_features)?; // (B, 257, 1024)

        let clip_patches = clip_features.narrow(1, 1, clip_features.dim(1)? - 1)?; // drop CLS -> (B, 256, 1024)
        let (b, c, h, w) = sam_features.dims4()?;
        let sam_patches = sam_features.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?; // (B, 256, 1024)

        let concat = Tensor::cat(&[&clip_patches, &sam_patches], D::Minus1)?; // (B, 256, 2048)
        self.projector.forward(&concat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    /// Verifies the wiring (SAM output grid -> CLIP patch count, concat dim, projector
    /// output dim) at a tiny scale. SAM's and CLIP's own internals (windowing, relative
    /// position bias, global attention) are already covered by sam_vit's and clip_vit's
    /// tests — a real 1024x1024/12-layer/64x64-grid forward pass takes minutes in an
    /// unoptimized debug build and would test the same wiring, just slower.
    #[test]
    fn forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // SAM: img=16/patch=2 -> grid 8x8, then two stride-2 downsamples (neck's net_2/net_3) -> grid 2x2, 8 channels.
        let sam = SamVitB::with_config(16, 2, 8, 2, 2, 2.0, 2, &[1], 4, (6, 8), vb.pp("sam_model"))?;
        // CLIP: img=4/patch=2 -> grid 2x2 (matches SAM's output grid), hidden=8 (matches SAM's output channels).
        let clip = ClipVitL::with_config(8, 2, 2, 16, 4, 2, 1e-5, vb.pp("vision_model"))?;
        let encoder = DeepEncoder::from_parts(sam, clip, 8, 8, 10, vb)?;

        let pixel_values = Tensor::randn(0f32, 1f32, (1, 3, 16, 16), &device)?;
        let out = encoder.forward(&pixel_values)?;
        assert_eq!(out.dims(), &[1, 4, 10], "expected 4 vision tokens (SAM's 2x2 grid) of width n_embed");
        Ok(())
    }
}
