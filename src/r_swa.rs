//! Reference Sliding Window Attention (R-SWA).
//!
//! Clean-room reimplementation of Unlimited-OCR's core contribution
//! (arXiv:2606.23050, eq. 1-9). Every decode token attends to:
//!   - the fixed prefix P = {1..L_m} (visual + prompt tokens, cached once, never evicted)
//!   - a causal sliding window D_n(t) of the last `window` decode tokens
//!
//! KV cache size is therefore bounded by L_m + window regardless of how many
//! tokens have been generated (eq. 6), unlike vanilla MHA where it grows as L_m + T.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{ops, Linear, Module, VarBuilder};

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

pub struct RotaryEmbedding {
    inv_freq: Tensor, // (head_dim/2,)
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, base: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / (base as f32).powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half, device)?;
        Ok(Self { inv_freq })
    }

    /// position_ids: (B, T) -> cos, sin: each (B, T, head_dim)
    pub fn forward(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t) = position_ids.dims2()?;
        let half = self.inv_freq.dim(0)?;
        let pos = position_ids.to_dtype(DType::F32)?.reshape((b, t, 1))?;
        let inv_freq = self.inv_freq.reshape((1, 1, half))?;
        let freqs = pos.broadcast_mul(&inv_freq)?; // (B, T, half)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // (B, T, head_dim)
        Ok((emb.cos()?, emb.sin()?))
    }
}

fn apply_rotary(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(1)?; // (B, 1, T, D) broadcasts over heads
    let sin = sin.unsqueeze(1)?;
    let q_out = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_out = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_out, k_out))
}

/// Per-layer KV cache implementing the R-SWA eviction policy.
///
/// Prefix KV is set once (prefill) and kept forever. Decode KV lives in a
/// fixed-size ring buffer of width `window`: the first `window` decode
/// tokens fill it linearly, then each new token overwrites the oldest slot.
pub struct RSWACache {
    window: usize,
    prefix_k: Option<Tensor>,
    prefix_v: Option<Tensor>,
    decode_k: Tensor,
    decode_v: Tensor,
    valid_len: usize,
    ring_pos: usize,
}

impl RSWACache {
    pub fn new(batch: usize, heads: usize, head_dim: usize, window: usize, dtype: DType, device: &Device) -> Result<Self> {
        let decode_k = Tensor::zeros((batch, heads, window, head_dim), dtype, device)?;
        let decode_v = decode_k.clone();
        Ok(Self { window, prefix_k: None, prefix_v: None, decode_k, decode_v, valid_len: 0, ring_pos: 0 })
    }

    pub fn set_prefix(&mut self, k: Tensor, v: Tensor) {
        self.prefix_k = Some(k);
        self.prefix_v = Some(v);
    }

    /// k, v: (B, H, 1, D) — a single new decode token.
    pub fn append_decode(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (b, h, _, d) = k.dims4()?;
        let slot = if self.valid_len < self.window { self.valid_len } else { self.ring_pos };
        let ranges = [0..b, 0..h, slot..slot + 1, 0..d];
        self.decode_k = self.decode_k.slice_assign(&ranges, k)?;
        self.decode_v = self.decode_v.slice_assign(&ranges, v)?;
        if self.valid_len < self.window {
            self.valid_len += 1;
        } else {
            self.ring_pos = (self.ring_pos + 1) % self.window;
        }
        Ok(())
    }

    /// The full set of keys/values a new decode token may attend to (P ∪ D_n(t)).
    pub fn active_kv(&self) -> Result<(Tensor, Tensor)> {
        let dk = self.decode_k.narrow(2, 0, self.valid_len)?;
        let dv = self.decode_v.narrow(2, 0, self.valid_len)?;
        let k = Tensor::cat(&[self.prefix_k.as_ref().unwrap(), &dk], 2)?;
        let v = Tensor::cat(&[self.prefix_v.as_ref().unwrap(), &dv], 2)?;
        Ok((k, v))
    }

    pub fn total_len(&self) -> Result<usize> {
        Ok(self.prefix_k.as_ref().unwrap().dim(2)? + self.valid_len)
    }
}

pub struct RSWAAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
}

impl RSWAAttention {
    pub fn new(hidden_size: usize, num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = num_heads * head_dim;
        Ok(Self {
            num_heads,
            head_dim,
            q_proj: candle_nn::linear_no_bias(hidden_size, inner, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(hidden_size, inner, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(hidden_size, inner, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(inner, hidden_size, vb.pp("o_proj"))?,
            rotary_emb: RotaryEmbedding::new(head_dim, 10000.0, vb.device())?,
        })
    }

    fn qkv(&self, hidden_states: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, _) = hidden_states.dims3()?;
        let shape = (b, t, self.num_heads, self.head_dim);
        let q = self.q_proj.forward(hidden_states)?.reshape(shape)?.transpose(1, 2)?;
        let k = self.k_proj.forward(hidden_states)?.reshape(shape)?.transpose(1, 2)?;
        let v = self.v_proj.forward(hidden_states)?.reshape(shape)?.transpose(1, 2)?;
        let (cos, sin) = self.rotary_emb.forward(position_ids)?;
        let (q, k) = apply_rotary(&q.contiguous()?, &k.contiguous()?, &cos, &sin)?;
        Ok((q, k, v.contiguous()?))
    }

    /// Standard causal self-attention over the prefix (visual + prompt tokens).
    pub fn prefill(&self, hidden_states: &Tensor, position_ids: &Tensor, cache: &mut RSWACache) -> Result<Tensor> {
        let (q, k, v) = self.qkv(hidden_states, position_ids)?;
        cache.set_prefix(k.clone(), v.clone());

        let (b, t, _) = hidden_states.dims3()?;
        let device = hidden_states.device();
        let mut mask_data = vec![0f32; t * t];
        for i in 0..t {
            for j in (i + 1)..t {
                mask_data[i * t + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(mask_data, (1, 1, t, t), device)?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let weights = ops::softmax(&weights.broadcast_add(&mask)?, D::Minus1)?;
        let out = weights.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    /// One decode token: attend over prefix ∪ sliding window, then update the ring buffer.
    pub fn decode_step(&self, hidden_states: &Tensor, position_ids: &Tensor, cache: &mut RSWACache) -> Result<Tensor> {
        let (q, k, v) = self.qkv(hidden_states, position_ids)?;
        cache.append_decode(&k, &v)?;
        let (active_k, active_v) = cache.active_kv()?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let weights = (q.matmul(&active_k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let weights = ops::softmax(&weights, D::Minus1)?;
        let out = weights.matmul(&active_v)?;
        let b = hidden_states.dim(0)?;
        let out = out.transpose(1, 2)?.reshape((b, 1, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    pub fn qkv_pub(&self, hidden_states: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        self.qkv(hidden_states, position_ids)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn o_proj(&self) -> &Linear {
        &self.o_proj
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::D;
    use candle_nn::VarMap;

    /// Recompute q/k/v for the ENTIRE sequence so far, then mask attention to
    /// exactly N(t) = prefix ∪ last `window` decode tokens (eq. 1-2). O(t) per
    /// call by design — this is the "obviously correct" reference, not the fast path.
    fn brute_force_decode_step(
        attn: &RSWAAttention,
        all_hidden: &Tensor,
        prefix_len: usize,
        window: usize,
    ) -> Result<Tensor> {
        let t = all_hidden.dim(1)?;
        let device = all_hidden.device();
        let position_ids = Tensor::arange(0u32, t as u32, device)?.unsqueeze(0)?;
        let (q, k, v) = attn.qkv_pub(all_hidden, &position_ids)?;

        let decode_len = t - prefix_len;
        let window_start = prefix_len + decode_len.saturating_sub(window);
        let mut mask_data = vec![f32::NEG_INFINITY; t];
        for j in 0..prefix_len {
            mask_data[j] = 0.0;
        }
        for j in window_start..t {
            mask_data[j] = 0.0;
        }
        let mask = Tensor::from_vec(mask_data, (1, 1, 1, t), device)?;

        let q_last = q.narrow(2, t - 1, 1)?;
        let scale = (attn.head_dim() as f64).powf(-0.5);
        let weights = (q_last.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let weights = ops::softmax(&weights.broadcast_add(&mask)?, D::Minus1)?;
        let out = weights.matmul(&v)?;
        let b = all_hidden.dim(0)?;
        let heads_x_dim = out.dim(1)? * out.dim(3)?;
        let out = out.transpose(1, 2)?.reshape((b, 1, heads_x_dim))?;
        attn.o_proj().forward(&out)
    }

    #[test]
    fn ring_buffer_matches_brute_force_and_caps_cache_size() -> Result<()> {
        let device = Device::Cpu;
        let (batch, heads, head_dim, hidden) = (1usize, 2usize, 8usize, 16usize);
        let window = 4usize;
        let prefix_len = 5usize;
        let decode_steps = 12usize;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = RSWAAttention::new(hidden, heads, head_dim, vb)?;
        let mut cache = RSWACache::new(batch, heads, head_dim, window, DType::F32, &device)?;

        let prefix_hidden = Tensor::randn(0f32, 1f32, (batch, prefix_len, hidden), &device)?;
        let prefix_positions = Tensor::arange(0u32, prefix_len as u32, &device)?.unsqueeze(0)?;
        attn.prefill(&prefix_hidden, &prefix_positions, &mut cache)?;
        assert_eq!(cache.total_len()?, prefix_len);

        let mut all_hidden = prefix_hidden.clone();
        for step in 1..=decode_steps {
            let new_token = Tensor::randn(0f32, 1f32, (batch, 1, hidden), &device)?;
            all_hidden = Tensor::cat(&[&all_hidden, &new_token], 1)?;
            let pos = Tensor::new(&[[(prefix_len + step - 1) as u32]], &device)?;

            let fast_out = attn.decode_step(&new_token, &pos, &mut cache)?;
            let expected_len = prefix_len + step.min(window);
            assert_eq!(cache.total_len()?, expected_len, "step {step}: cache size mismatch");

            let ref_out = brute_force_decode_step(&attn, &all_hidden, prefix_len, window)?;
            let diff: f32 = (fast_out - ref_out)?.abs()?.max_all()?.to_scalar()?;
            assert!(diff < 1e-4, "step {step}: ring-buffer output diverged from brute-force by {diff}");
        }

        assert_eq!(cache.total_len()?, prefix_len + window, "cache must stay capped at L_m + n after warmup");
        Ok(())
    }
}
