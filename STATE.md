# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-03T17:27:53.661214Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/is-there-a-sounds-like-ai-direction-cc6e-codex-gpt-5-6-sol`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
Phase `resource_finder` completed 2026-10-03. Gathered and validated 12 PDFs, four required datasets, and six official repositories; wrote `literature_review.md`, `resources.md`, `planning.md`, and per-directory READMEs. Core specified papers were split into 43 three-page chunks and fully reviewed; searchable text is in `artifacts/fulltexts/`.

Key finding: human/AI text is robustly linearly decodable, but existing MGT probe papers do not causally steer free generation. The closest causal work changes self-authorship judgments, while SAE and Assistant Axis results show strong formality/verbosity/persona confounds. HAP-E is the primary paired identification dataset; MAGE and RAID are independent OOD/stress tests.

Direction budget fixed in `planning.md`: (1) signed causal steering with supervised human→AI directions, (2) Assistant/style nuisance residualization, and (3) layer/model transfer with paired counterfactual validation. Rejected directions and reasons are recorded there.

Next phase: implement on an open 8B instruct/base pair using HAP-E document-level splits; reuse `code/sv-detect`, `code/mgt_probes`, RepE hooks, and `code/assistant-axis`; evaluate with independent MAGE/RAID-trained detectors plus content/coherence metrics. Full RAID was not downloaded because its source is 16.7 GB; a reproducible 4,800-row coverage sample and full-download instructions are in `datasets/`. GPU execution remains unresolved and requires a compatible CUDA host/model weights.

Failures/workarounds: paper-finder returned HTTP 500, so primary-source manual search was used; `uv add` hit Hatchling empty-package detection, so dependencies were installed in the fresh `.venv` via `uv pip` and recorded in `pyproject.toml`; `pdftotext` was absent, so `pypdf` extracted each chunk.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
Phases 1–6 complete. Planning and direction ranking are in `planning.md`; environment and EDA evidence are in `pyproject.toml`, `results/data_audit.csv`, and `figures/hape_lengths.png`. Real Qwen2.5-7B base/instruct runs produced five-layer train/validation/test activations, fitted vectors, 292 causal/control generations, two independent detector scores, quality metrics, paired statistics, error analysis, and figures. Primary artifacts: `results/qwen25_7b_{instruct,base}/`, `results/evaluated_generations.parquet`, `results/endpoint_effects.csv`, `results/dose_response_stats.csv`, `results/error_analysis.md`, and `figures/`.

Key result: content-paired human/AI authorship is almost perfectly readable in both checkpoints (selected-layer test AUROC 0.9997), and the direction has low cosine with Assistant/formality axes. Causal steering is null at tested layer/doses: instruct +2 minus −2 MAGE-detector Δ=+0.008, 95% CI [−0.077,+0.095]; public-RoBERTa Δ=−0.077 [−0.246,+0.057]. The jointly orthogonal residual and all controls are also null; base replication is null. Quality proxies do not explain the raw null. Evidence is high for within-design readout and low-to-moderate for the causal null because n=12–16 detects only large paired effects.

Documented deviations/failures: PyTorch 2.14 required an unavailable compiler, so the isolated environment was pinned to 2.7.x; the live-catalog-verified OpenRouter judge returned HTTP 403 because the supplied key was already over its daily limit, so a validated public RoBERTa detector became the second measure; validation caught and corrected a truncated prompt-baseline instruction and correlated-axis sequential residualization. No scores were simulated.

Final documentation: `REPORT.md`, `README.md`, updated `resources.md`, and commented code under `src/`. `results/validation.json` records all checks passing; cached statistical outputs reproduced with identical hash `496f0e60b20cf5b5742f2952cce23b7001f3ccb2334ad847d2d9cae394c93c26`. No experiment process remains running. Next step, if pursued, is a powered multi-layer study with 64–100 prompts and successful blind judging.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
