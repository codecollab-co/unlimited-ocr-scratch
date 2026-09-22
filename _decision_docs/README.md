# Research references

Diligo's ML/OCR core is built on the clean-room Rust reimplementation of Baidu's
Unlimited-OCR architecture, studied and built in the sibling `unlimited-ocr-scratch`
project at the `ocr` workspace root.

- **Paper**: *Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing* —
  Yin et al., Baidu Inc., 2026. [arXiv:2606.23050](https://arxiv.org/abs/2606.23050)
- **Reference implementation**: [github.com/baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR),
  [huggingface.co/baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
- **Lineage**: DeepSeek-OCR ([arXiv:2510.18234](https://arxiv.org/abs/2510.18234)),
  Segment Anything ([arXiv:2304.02643](https://arxiv.org/abs/2304.02643))
- **Rust reimplementation + real-weight verification**: `../../unlimited-ocr-scratch/`
  (sibling project) — [github.com/codecollab-co/unlimited-ocr-scratch](https://github.com/codecollab-co/unlimited-ocr-scratch)

Deeper source material — the extracted paper text, the cloned HF model repo — lives in
`../../_research_refs/` at the `ocr` workspace root and isn't duplicated here, to avoid
carrying two copies of the same large cloned repos. Point here if you need the citation
or the lineage; go up a level if you need the actual source files.
