# unlimited-ocr-scratch

A clean-room, from-scratch **Rust** reimplementation of Baidu's **Unlimited-OCR**
architecture — R-SWA (Reference Sliding Window Attention), the SAM-ViT-B + CLIP-ViT-L
DeepEncoder, and the DeepSeek-V2-style MoE decoder — built as a learning project, then
verified by loading the real published weights and confirming plausible output.

## Paper

> **Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing**
> Youyang Yin, Huanhuan Liu, YY, et al. — Baidu Inc., 2026.
> arXiv: [2606.23050](https://arxiv.org/abs/2606.23050)

```bibtex
@misc{yin2026unlimitedocrworks,
      title={Unlimited OCR Works},
      author={Youyang Yin and Huanhuan Liu and YY and Qunyi Xie and Chaorun Liu and Shiqi Yang and Shaohua Wang and Zhanlong Liu and Hao Zou and Jinyue Chen and Shu Wei and Jingjing Wu and Mingxin Huang and Zhen Wu and Guibin Wang and Tengyu Du and Lei Jia},
      year={2026},
      eprint={2606.23050},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.23050},
}
```

Baidu's original reference implementation and published weights:
[github.com/baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) ·
[huggingface.co/baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) (MIT licensed).
This repository's code is independently written from reading that source and the paper
(see the [Stages](#stages) section below for what was studied) — no code is copied from
the reference. This repo is [MIT licensed](LICENSE) itself; model weights loaded by this
code (not included in this repository) remain subject to Baidu's own license terms for
`baidu/Unlimited-OCR` on Hugging Face.

Unlimited-OCR itself builds on **DeepSeek-OCR** (arXiv:[2510.18234](https://arxiv.org/abs/2510.18234),
Wei, Sun & Li — for the DeepEncoder and MoE decoder it extends) and
[**Segment Anything**](https://arxiv.org/abs/2304.02643) (Kirillov et al. — SAM-ViT-B).

## Language choice

Note on language choice: porting inference glue code to Rust does not change OCR
*accuracy* — that's fixed by the weights/architecture, not the host language. This
project is Rust because it's a from-scratch learning build, not because Rust makes
the model "better" at OCR. Built with [candle](https://github.com/huggingface/candle),
Hugging Face's pure-Rust ML framework (CPU/CUDA/Metal backends, native safetensors
loading) — no Python anywhere in this project.

Run tests: `cargo test` (from this directory). Every stage's correctness check lives
as a `#[cfg(test)] mod tests` block in its own source file — no separate test runner.

## Confirmed hyperparameters (from config.json)

Decoder (DeepSeek-V2 style, MLA disabled -> plain MHA):
- hidden_size=1280, num_hidden_layers=12, num_attention_heads=10, num_key_value_heads=10, v_head_dim=128
- layer 0 dense (intermediate_size=6848); layers 1-11 MoE (n_routed_experts=64, n_shared_experts=2,
  num_experts_per_tok=6, moe_intermediate_size=896, topk_method="greedy")
- vocab_size=129280, max_position_embeddings=32768
- sliding_window=128  <- this is `n` (the R-SWA decode window) from the paper

Vision (DeepEncoder = SAM-ViT-B -> CLIP-ViT-L -> linear projector):
- SAM-ViT-B: width=768, layers=12, heads=12, global_attn_indexes=[2,5,8,11], downsample_channels=[512,1024]
- CLIP-L/14-224: width=1024, layers=24, heads=16, patch_size=14, image_size=224
- Projector: concat(SAM 1024-d, CLIP 1024-d) = 2048-d -> linear -> 1280-d (matches decoder hidden_size)
- 16x token compression at the SAM/CLIP bridge (paper section 3.3)

## R-SWA (the paper's core contribution — eq. 1-9)

`N(t) = P ∪ D_n(t)` where `P = {1..L_m}` (prefix: visual+prompt tokens, fixed, never evicted)
and `D_n(t)` is a causal sliding window of width `n=128` over the decode region.
KV cache size is bounded: `C_R-SWA(T) = L_m + min(n, T) ≤ L_m + n` (constant), vs. vanilla MHA's
`C_MHA(T) = L_m + T` (unbounded growth). Reference implementation:
`SlidingWindowLlamaAttention` in `modeling_deepseekv2.py:1232-1377` — prefix cached normally on
prefill, then a fixed-width ring buffer overwrites decode-side KV slots in place once full.

## Stages

1. **R-SWA attention** (`src/r_swa.rs`) — the novel mechanism, no dependency on the rest.
   Test: constant cache size post-warmup + numerical match (`<1e-4`) vs. brute-force windowed attention.
2. **SAM-ViT-B encoder** (`src/sam_vit.rs`) — window+global attention ViT, decomposed relative
   position bias, neck + downsampling convs. Tests: window partition/unpartition round-trip,
   forward shape, global-attention block placement.
3. **CLIP-ViT-L encoder + projector** (`src/clip_vit.rs`, `src/deep_encoder.rs`) — completes DeepEncoder.
4. **MoE decoder** (`src/moe_decoder.rs`) — 12-layer DeepSeek-V2-style transformer using Stage 1's R-SWA.
5. **Full model + real weight loading + plausibility check** (`src/bin/verify_weights.rs`) — load
   the real `baidu/Unlimited-OCR` safetensors (6.67GB) into the from-scratch `DeepEncoder` and
   `MoeDecoder` via candle's `VarBuilder::from_mmaped_safetensors`, run one real forward pass
   through each, confirm finite/correctly-shaped/non-degenerate output. Not a bit-exact diff
   against the Python reference — no Python is used anywhere in this project — see the note below.

Each stage lands with its own `#[test]`s before moving to the next — no stage is "done" until
`cargo test` passes for it.

## Status — all 5 stages complete

- [x] Stage 1 — `src/r_swa.rs`
- [x] Stage 2 — `src/sam_vit.rs`
- [x] Stage 3 — `src/clip_vit.rs`, `src/deep_encoder.rs`
- [x] Stage 4 — `src/moe_decoder.rs`
- [x] Stage 5 — `src/bin/verify_weights.rs`

`cargo test`: 7/7 passing (0.03s).

`cargo run --release --bin verify_weights` (weights at `weights/model.safetensors`, gitignored,
not checked in): every state_dict key the reimplementation requested was found in the real
checkpoint with a matching shape (confirms the module tree exactly mirrors the reference's), and
every forward pass produced finite, correctly-shaped, non-degenerate output:

```
vision_tokens:         shape=[1, 256, 1280]   min=-1.64  max=0.81  mean=-0.004
prefill_logits:        shape=[1, 8, 129280]   min=-22.4  max=26.4  mean=-3.00
decode_step_0_logits:  shape=[1, 1, 129280]   min=-21.9  max=27.9  mean=1.77
decode_step_1_logits:  shape=[1, 1, 129280]   min=-14.1  max=11.6  mean=-2.32
decode_step_2_logits:  shape=[1, 1, 129280]   min=-14.1  max=10.3  mean=-2.09
```

**Scope note on "verified":** this confirms the reimplementation loads and runs the real weights
correctly at the module level (shapes, no NaN/Inf, plausible activation ranges) — not a token-level
match against the original model's generated text. That would need the multimodal input-embedding
splicing (locating the `<image>` placeholder token, injecting `vision_tokens` in its place, tokenizer/
chat-template handling) and, per the "verify approach" decision made mid-project, was explicitly
scoped out in favor of staying 100% Python-free. A future pass could add that splicing logic and a
real end-to-end `infer()` if exact-generation parity becomes the goal.

Note on test scale: full-config forward passes (real 1024x1024 image, SAM's 64x64 grid,
CLIP's 24 layers) are correct but slow in an unoptimized debug build — minutes, not
seconds — because candle's plain `matmul`/`softmax` aren't fused attention kernels.
Tests instead use tiny synthetic configs (`SamVitB::with_config`, `ClipVitL::with_config`)
that exercise the same code paths (windowing, global attention, relative position bias,
the SAM->CLIP concat wiring) in milliseconds. Real-scale correctness gets checked once,
for real, in Stage 5 — against the actual reference weights, in `--release` mode.
