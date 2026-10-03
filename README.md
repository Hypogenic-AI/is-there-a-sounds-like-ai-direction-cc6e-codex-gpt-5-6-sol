# Residual-stream “AI-ness” direction

This project tests whether a linear direction separating human- from machine-written text in Qwen2.5-7B activations also causally controls how AI-like the model's generations appear. It uses content-paired HAP-E texts, signed activation steering, base/instruct checkpoints, nuisance-axis residualization, two independent detectors, and quality controls.

## Key findings

- Human versus machine text was almost perfectly readable in both Qwen2.5-7B checkpoints (held-out AUROC 0.9997).
- The readout direction was nearly orthogonal to measured Assistant and formality axes.
- Signed steering did **not** produce a reliable independent-detector shift; detector signs disagreed and all corrected tests were null.
- The residualized direction and random, shuffled, formality, and Assistant controls were also null.
- A plain “write like a human” prompt did not reliably lower detector scores and substantially degraded content/repetition metrics.

The result is a qualified negative: strong decodability does not imply causal control at the tested layer and doses. See [REPORT.md](REPORT.md) for full methods, statistics, figures, and limitations.

## Reproduce

```bash
uv venv
source .venv/bin/activate
uv sync --no-install-project

# With model weights under artifacts/models/:
python src/eda.py
python src/experiment.py --model artifacts/models/Qwen2.5-7B-Instruct --tag qwen25_7b_instruct --instruct --full-conditions --n-train 240 --n-val 80 --n-test 80 --n-prompts 16 --batch-size 4 --max-new-tokens 80
python src/experiment.py --model artifacts/models/Qwen2.5-7B --tag qwen25_7b_base --n-train 240 --n-val 80 --n-test 80 --n-prompts 12 --batch-size 4 --max-new-tokens 80
python src/evaluate.py --tags qwen25_7b_instruct qwen25_7b_base --skip-judge
python src/add_perplexity.py
python -m src.analyze
```

The run used the documented `uv pip` fallback because the research root is not itself an installable package.

## Structure

- `planning.md` — motivation, hypotheses, direction ranking, and decision rules
- `src/` — EDA, activation extraction/steering, evaluation, and analysis
- `results/` — activations, vectors, raw generations, scores, and statistics
- `figures/` — publication-ready plots
- `REPORT.md` — primary research report
- `literature_review.md`, `resources.md`, `papers/`, `code/`, `datasets/` — gathered resources
