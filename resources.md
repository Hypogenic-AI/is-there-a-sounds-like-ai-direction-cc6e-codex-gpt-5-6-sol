# Resources Catalog

## Summary

The workspace contains 12 validated papers, four required datasets (three complete and one documented coverage sample because the full release is 16.7 GB), and six official code repositories. The resources support a causal steering study with HAP-E for identification, MAGE/RAID for independent evaluation, and published assistant/persona methods for confound control.

## Papers

| Title | Authors | Year | File | Key information |
|---|---|---:|---|---|
| Linear Probing Provides Robust and Efficient Detection of MGT | Quaremba et al. | 2026 | `papers/2608.24780_linear_probing_mgt.pdf` | Shared, continuous, OOD-transferable linear MGT readout |
| SV-Detect | Vishnyakov & Gaintseva | 2026 | `papers/2606.07313_sv_detect.pdf` | Layer-wise logistic directions and interpretation; readout only |
| Feature-Level Insights into ATD with SAEs | Kuznetsov et al. | 2025 | `papers/2503.03601_sae_artificial_text_detection.pdf` | Interpretable formality/repetition/complexity features and feature steering |
| The Assistant Axis | Lu et al. | 2026 | `papers/2601.10387_assistant_axis.pdf` | Causal assistant-persona axis and activation capping |
| Inspection and Control of Self-Generated-Text Recognition | Ackerman & Panickssery | 2025 | `papers/2410.02064_self_generated_text_recognition.pdf` | Closest causal precedent; controls self-authorship judgment/perception |
| Steering Language Models With Activation Engineering | Turner et al. | 2023 | `papers/2308.10248_activation_engineering.pdf` | Contrastive activation addition |
| Representation Engineering | Zou et al. | 2023 | `papers/2310.01405_representation_engineering.pdf` | General reading/control framework |
| Effectiveness of Style Vectors: Human Evaluation | Diallo et al. | 2026 | `papers/2601.21505_style_vectors_human_eval.pdf` | Human-rated steering/quality tradeoff |
| RAID | Dugan et al. | 2024 | `papers/2405.07940_raid.pdf` | Robust detector benchmark and attacks |
| How Close is ChatGPT to Human Experts? | Guo et al. | 2023 | `papers/2301.07597_hc3.pdf` | HC3 corpus and detector baselines |
| Do LLMs Write Like Humans? | Reinhart et al. | 2025 | `papers/2410.16107_hape_style_variation.pdf` | Content-paired rhetorical analysis; instruct/base contrast |
| MAGE | Li et al. | 2024 | `papers/2305.13242_mage.pdf` | Broad OOD MGT benchmark |

See `papers/README.md` for full descriptions and `artifacts/fulltexts/` for searchable extraction. Core-paper chunk manifests are under `papers/pages/`.

## Datasets

| Name | Source | Local size/status | Task | Location | Notes |
|---|---|---|---|---|---|
| HC3 | Hugging Face `Hello-SimpleAI/HC3` | Complete `all.jsonl`, 73.7 MB | paired QA authorship | `datasets/HC3/` | 24,322 questions; 85,449 total answers |
| RAID | Hugging Face `liamdugan/raid` | 4,800-row local sample; full source 16.7 GB | robust MGT detection | `datasets/raid/` | 11 machine models, 12 attack values in local coverage sample; full instructions provided |
| HAP-E | Hugging Face `browndw/human-ai-parallel-corpus` | Complete, 114 MB | content-paired human/AI continuation | `datasets/human-ai-parallel-corpus/` | 8,290 aligned IDs × 8 files; primary identification data |
| MAGE | Hugging Face `yaful/MAGE` | Complete, 554 MB | in-domain and OOD detection | `datasets/MAGE/` | 436,606 released rows across five splits |

`datasets/.gitignore` excludes all large payloads while retaining `datasets/README.md`, the reproducible downloader, and small examples. Validation found no missing values in HAP-E or MAGE and only expected human-row null metadata in the RAID sample.

## Code repositories

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| mgt_probes | https://github.com/gerritq/mgt_probes | LLP/CLP linear probing | `code/mgt_probes/` | Best probe baseline; add generation hook |
| sv-detect | https://github.com/Atmyre/sv-detect | Activation extraction, layer directions, OOD analysis | `code/sv-detect/` | Best direction-estimation starting point |
| assistant-axis | https://github.com/safety-research/assistant-axis | Persona axes and activation capping | `code/assistant-axis/` | Required confound/control implementation |
| representation-engineering | https://github.com/andyzoujm/representation-engineering | Generic representation control | `code/representation-engineering/` | Reusable intervention plumbing |
| raid | https://github.com/liamdugan/raid | Detector evaluation and attacks | `code/raid/` | Official robust evaluation; large checkout |
| chatgpt-comparison-detection | https://github.com/Hello-SimpleAI/chatgpt-comparison-detection | HC3 detector/linguistic baselines | `code/chatgpt-comparison-detection/` | Independent legacy baseline |

All Python source trees compiled successfully. Full execution was not attempted because the official repositories specify mutually incompatible CUDA/PyTorch/Transformers stacks and require model weights/GPU memory. Exact revisions and entry points are in `code/README.md`.

## Resource gathering notes

### Search strategy

The required paper-finder service was attempted first in diligent JSON mode but returned HTTP 500. Manual fallback searched arXiv and official project pages for residual-stream machine-text probes, activation steering, style vectors, self-authorship recognition, persona directions, and the named datasets. Citation chasing from the four core papers identified ActAdd, RepE, self-recognition, HAP-E, RAID, HC3, and MAGE. Only primary papers, official dataset cards, and official repositories were used for technical claims.

### Selection criteria

Resources were selected for direct causal relevance, a clean control or alternative explanation, established OOD evaluation, content pairing, or reusable official code. The search emphasized quality over count. `planning.md` scores eight plausible experimental directions and fixes the top three under the direction budget.

### Challenges and workarounds

- Paper-finder failed with an internal server error; manual primary-source search was used and logged.
- `pdftotext` was unavailable; the mandated chunker plus `pypdf` extracted every core chunk. Complex PDF form warnings did not prevent section review.
- `uv add` failed because Hatchling cannot build an empty package; the required fresh `.venv` was retained, packages were installed with the mandated `uv pip` fallback, and dependencies were recorded in root `pyproject.toml`.
- Full RAID is too large for a prudent default download. The local sample uses evenly spaced dataset-server blocks; full download commands are documented.
- GPU repositories pin conflicting environments. They were syntax-checked and documented without polluting the root environment.

## Experiment design recommendation

1. **Primary data:** HAP-E, split by `doc_id`, using human chunk 2 versus within-model continuations. Use MAGE for independent detector training and RAID for adversarial/OOD stress testing.
2. **Primary model:** an open 8B instruct model with its base counterpart. Estimate per-layer mean-difference and logistic directions, then run signed activation-addition sweeps.
3. **Causal outcome:** change in independent detector score with bootstrap intervals clustered by prompt. Require monotonicity and specificity over matched random/shuffled directions.
4. **Preservation outcomes:** semantic similarity or task accuracy, length, perplexity, repetition, grammar, and blind quality/coherence ratings.
5. **Distinctness tests:** cosine overlap and matched-norm orthogonalization against a same-model Assistant Axis, plus formality, verbosity, sentiment, domain, and length nuisance directions.
6. **Code to reuse:** activation/direction code from `sv-detect`, probe baselines from `mgt_probes`, intervention hooks from RepE, persona controls from `assistant-axis`, and evaluation/attacks from RAID.

The decisive interpretation rule is conservative: detector movement from the raw vector alone is insufficient. A distinct causal AI-ness direction requires preserved content/quality and an effect that remains after persona and style residualization.

## Experiment execution (2026-10-03)

The recommendation was executed on Qwen2.5-7B base and instruct checkpoints with HAP-E paired continuations. The completed run includes five-layer readout, signed generation-time activation addition, joint-span Assistant/formality residualization, equal-norm random and shuffled-label controls, a prompt baseline, two independent detectors, perplexity/semantic/repetition checks, paired bootstrap/permutation statistics, and a reduced base-model replication. Actual outputs are in `results/`, plots in `figures/`, implementation in `src/`, and interpretation in `REPORT.md`.

The planned OpenRouter judge (`openai/gpt-5.6-luna`, verified in the live catalog) could not execute because the supplied key had already exceeded its daily workspace limit. The run used the independently trained public OpenAI RoBERTa detector as the second authorship measure and reports this substitution as a limitation; no scores were simulated.

## Experiment execution (2026-10-03)

The recommendation was executed on Qwen2.5-7B base and instruct checkpoints with HAP-E paired continuations. The completed run includes five-layer readout, signed generation-time activation addition, Assistant/formality residualization, equal-norm random and shuffled-label controls, a prompt baseline, two independent detectors, perplexity/semantic/repetition checks, paired bootstrap/permutation statistics, and a reduced base-model replication. Actual outputs are in `results/`, plots in `figures/`, implementation in `src/`, and interpretation in `REPORT.md`.

The planned OpenRouter judge (`openai/gpt-5.6-luna`, verified in the live catalog) could not execute because the supplied key had already exceeded its daily workspace limit. The run used the independently trained public OpenAI RoBERTa detector as the second authorship measure and reports this substitution as a limitation; no scores were simulated.
