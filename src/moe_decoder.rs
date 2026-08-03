//! DeepSeek-V2-style MoE decoder, using Stage 1's R-SWA as every layer's attention.
//!
//! Matches config.json: hidden_size=1280, num_hidden_layers=12, num_attention_heads=10,
//! num_key_value_heads=10 (plain MHA, no GQA — use_mla=false so there's no latent
//! attention either), v_head_dim=128, vocab_size=129280. Layer 0 is a dense MLP
//! (intermediate_size=6848); layers 1-11 are MoE (n_routed_experts=64, n_shared_experts=2,
//! num_experts_per_tok=6, moe_intermediate_size=896, topk_method="greedy", scoring
//! "softmax", norm_topk_prob=false, routed_scaling_factor=1.0 — see
//! configuration_deepseek_v2.py defaults). `sliding_window=128` is R-SWA's `n`.
//!
//! Simplification: each MoE layer here evaluates all n_routed_experts densely and masks
//! non-selected ones to zero weight, rather than gathering/dispatching only the top-k
//! per token. Mathematically identical (a zero-weighted expert contributes nothing to
//! the sum) — just wasteful compute. A real serving engine (SGLang, this project's own
//! Stage 5 target) dispatches sparsely for throughput; this reimplementation prioritizes
//! being obviously correct over being fast.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Init, Linear, Module, VarBuilder};

use crate::r_swa::{RSWAAttention, RSWACache};

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(hidden: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(hidden, "weight", Init::Const(1.))?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let variance = x32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        self.weight.to_dtype(DType::F32)?.broadcast_mul(&x_norm)?.to_dtype(dtype)
    }
}

/// SwiGLU-style dense MLP: down_proj(silu(gate_proj(x)) * up_proj(x)).
struct DenseMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DenseMlp {
    fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gated * up)?)
    }
}

/// Softmax gate + greedy top-k expert selection (config: topk_method="greedy",
/// scoring_func="softmax", norm_topk_prob=false, routed_scaling_factor=1.0 — so
/// selected weights are used as-is, no renormalization to sum-to-1).
struct MoeGate {
    weight: Tensor, // (n_routed_experts, hidden)
    top_k: usize,
    n_routed_experts: usize,
}

impl MoeGate {
    fn new(hidden: usize, n_routed_experts: usize, top_k: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints((n_routed_experts, hidden), "weight", Init::Const(0.))?;
        Ok(Self { weight, top_k, n_routed_experts })
    }

    /// hidden_states: (N, H) flattened tokens. Returns (N, n_routed_experts): the
    /// softmax gate score for each token's chosen top-k experts, 0 for the rest.
    fn dispatch_weights(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let logits = hidden_states.to_dtype(DType::F32)?.matmul(&self.weight.to_dtype(DType::F32)?.t()?)?;
        let scores = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let scores_host: Vec<Vec<f32>> = scores.to_vec2()?;

        let mut dispatch = vec![0f32; scores_host.len() * self.n_routed_experts];
        for (i, row) in scores_host.iter().enumerate() {
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            for &e in idx.iter().take(self.top_k) {
                dispatch[i * self.n_routed_experts + e] = row[e];
            }
        }
        Tensor::from_vec(dispatch, (scores_host.len(), self.n_routed_experts), hidden_states.device())
    }
}

struct DeepseekMoe {
    gate: MoeGate,
    experts: Vec<DenseMlp>,
    shared_experts: DenseMlp,
}

impl DeepseekMoe {
    #[allow(clippy::too_many_arguments)]
    fn new(hidden: usize, moe_intermediate: usize, n_routed_experts: usize, n_shared_experts: usize, top_k: usize, vb: VarBuilder) -> Result<Self> {
        let gate = MoeGate::new(hidden, n_routed_experts, top_k, vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(n_routed_experts);
        for i in 0..n_routed_experts {
            experts.push(DenseMlp::new(hidden, moe_intermediate, vb.pp(format!("experts.{i}")))?);
        }
        let shared_experts = DenseMlp::new(hidden, moe_intermediate * n_shared_experts, vb.pp("shared_experts"))?;
        Ok(Self { gate, experts, shared_experts })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, h) = x.dims3()?;
        let flat = x.reshape((b * t, h))?;
        let dispatch = self.gate.dispatch_weights(&flat)?; // (N, E)

        let mut y = Tensor::zeros((b * t, h), x.dtype(), x.device())?;
        for (e, expert) in self.experts.iter().enumerate() {
            let w = dispatch.narrow(1, e, 1)?; // (N, 1)
            y = (y + expert.forward(&flat)?.broadcast_mul(&w)?)?;
        }
        let y = (y + self.shared_experts.forward(&flat)?)?;
        y.reshape((b, t, h))
    }
}

enum Mlp {
    Dense(DenseMlp),
    Moe(DeepseekMoe),
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Mlp::Dense(m) => m.forward(x),
            Mlp::Moe(m) => m.forward(x),
        }
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: RSWAAttention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn prefill(&self, x: &Tensor, position_ids: &Tensor, cache: &mut RSWACache) -> Result<Tensor> {
        let attn_out = self.self_attn.prefill(&self.input_layernorm.forward(x)?, position_ids, cache)?;
        let h = (x + attn_out)?;
        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h)?)?;
        h + mlp_out
    }

    fn decode_step(&self, x: &Tensor, position_ids: &Tensor, cache: &mut RSWACache) -> Result<Tensor> {
        let attn_out = self.self_attn.decode_step(&self.input_layernorm.forward(x)?, position_ids, cache)?;
        let h = (x + attn_out)?;
        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h)?)?;
        h + mlp_out
    }
}

pub struct MoeDecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub dense_intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f64,
}

impl MoeDecoderConfig {
    /// The real Unlimited-OCR config (config.json's language_config).
    pub fn real() -> Self {
        Self {
            vocab_size: 129280,
            hidden_size: 1280,
            num_hidden_layers: 12,
            num_attention_heads: 10,
            head_dim: 128,
            dense_intermediate_size: 6848,
            moe_intermediate_size: 896,
            n_routed_experts: 64,
            n_shared_experts: 2,
            num_experts_per_tok: 6,
            first_k_dense_replace: 1,
            sliding_window: 128,
            rms_norm_eps: 1e-6,
        }
    }
}

pub struct MoeDecoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    cfg_heads: usize,
    cfg_head_dim: usize,
    cfg_window: usize,
}

impl MoeDecoder {
    pub fn new(cfg: &MoeDecoderConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("model.layers.{i}"));
            let mlp = if i >= cfg.first_k_dense_replace {
                Mlp::Moe(DeepseekMoe::new(
                    cfg.hidden_size,
                    cfg.moe_intermediate_size,
                    cfg.n_routed_experts,
                    cfg.n_shared_experts,
                    cfg.num_experts_per_tok,
                    layer_vb.pp("mlp"),
                )?)
            } else {
                Mlp::Dense(DenseMlp::new(cfg.hidden_size, cfg.dense_intermediate_size, layer_vb.pp("mlp"))?)
            };
            layers.push(DecoderLayer {
                input_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, layer_vb.pp("input_layernorm"))?,
                self_attn: RSWAAttention::new(cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim, layer_vb.pp("self_attn"))?,
                post_attention_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, layer_vb.pp("post_attention_layernorm"))?,
                mlp,
            });
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;

        Ok(Self { embed_tokens, layers, norm, lm_head, cfg_heads: cfg.num_attention_heads, cfg_head_dim: cfg.head_dim, cfg_window: cfg.sliding_window })
    }

    pub fn new_caches(&self, batch: usize, dtype: DType, device: &candle_core::Device) -> Result<Vec<RSWACache>> {
        (0..self.layers.len())
            .map(|_| RSWACache::new(batch, self.cfg_heads, self.cfg_head_dim, self.cfg_window, dtype, device))
            .collect()
    }

    /// input_ids: (B, T) -> logits: (B, T, vocab_size). Sets up each layer's R-SWA prefix cache.
    pub fn prefill(&self, input_ids: &Tensor, position_ids: &Tensor, caches: &mut [RSWACache]) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            x = layer.prefill(&x, position_ids, cache)?;
        }
        self.lm_head.forward(&self.norm.forward(&x)?)
    }

    /// input_ids: (B, 1) -> logits: (B, 1, vocab_size). One R-SWA decode step per layer.
    pub fn decode_step(&self, input_ids: &Tensor, position_ids: &Tensor, caches: &mut [RSWACache]) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            x = layer.decode_step(&x, position_ids, cache)?;
        }
        self.lm_head.forward(&self.norm.forward(&x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn tiny_config() -> MoeDecoderConfig {
        MoeDecoderConfig {
            vocab_size: 20,
            hidden_size: 8,
            num_hidden_layers: 3, // layer 0 dense, layers 1-2 MoE
            num_attention_heads: 2,
            head_dim: 4,
            dense_intermediate_size: 6,
            moe_intermediate_size: 5,
            n_routed_experts: 6,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            first_k_dense_replace: 1,
            sliding_window: 3,
            rms_norm_eps: 1e-6,
        }
    }

    #[test]
    fn moe_gate_selects_exactly_top_k_experts_per_token() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let gate = MoeGate::new(8, 6, 2, vb)?;

        let tokens = Tensor::randn(0f32, 1f32, (5, 8), &device)?;
        let dispatch = gate.dispatch_weights(&tokens)?;
        let rows: Vec<Vec<f32>> = dispatch.to_vec2()?;
        for row in rows {
            let nonzero = row.iter().filter(|&&w| w != 0.0).count();
            assert_eq!(nonzero, 2, "each token should route to exactly top_k=2 experts, got {nonzero}");
            let total: f32 = row.iter().sum();
            assert!(total > 0.0, "selected experts should carry positive softmax weight");
        }
        Ok(())
    }

    #[test]
    fn full_stack_prefill_then_decode_produces_expected_logit_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let decoder = MoeDecoder::new(&cfg, vb)?;

        let batch = 1;
        let prefix_len = 5;
        let mut caches = decoder.new_caches(batch, DType::F32, &device)?;

        let ids: Vec<u32> = (0..prefix_len as u32).map(|i| i % cfg.vocab_size as u32).collect();
        let input_ids = Tensor::from_vec(ids, (batch, prefix_len), &device)?;
        let position_ids = Tensor::arange(0u32, prefix_len as u32, &device)?.unsqueeze(0)?;
        let logits = decoder.prefill(&input_ids, &position_ids, &mut caches)?;
        assert_eq!(logits.dims(), &[batch, prefix_len, cfg.vocab_size]);

        // Decode past the sliding window (window=3) to exercise the ring buffer end-to-end.
        for step in 0..6 {
            let next_id = Tensor::from_vec(vec![(step % cfg.vocab_size) as u32], (batch, 1), &device)?;
            let pos = Tensor::new(&[[(prefix_len + step) as u32]], &device)?;
            let logits = decoder.decode_step(&next_id, &pos, &mut caches)?;
            assert_eq!(logits.dims(), &[batch, 1, cfg.vocab_size]);
        }

        for cache in &caches {
            assert_eq!(cache.total_len()?, prefix_len + cfg.sliding_window, "cache should be capped at L_m + window after 6 decode steps past a window of 3");
        }
        Ok(())
    }
}
