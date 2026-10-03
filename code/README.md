# Cloned Repositories

All repositories were shallow-cloned and their Python sources passed `compileall`. This validates syntax, not GPU execution. Current revisions are recorded below for reproducibility.

## mgt_probes

- **URL / revision:** https://github.com/gerritq/mgt_probes, `8875297`
- **Purpose:** official LLP/CLP implementation for arXiv:2608.24780.
- **Key files:** `src/probes/probe_main.py`, `src/inference.py`, `probe.sh`.
- **Reuse:** train per-layer or concatenated logistic readout directions. The experiment runner should add generation-time hooks, because the released code is detector-oriented.
- **Requirements:** Python 3.11, PyTorch 2.5.1 CUDA build, Transformers, scikit-learn, and detector dependencies.

## sv-detect

- **URL / revision:** https://github.com/Atmyre/sv-detect, `d8d6541`
- **Purpose:** official layer-wise activation extraction, logistic/mean/PCA direction construction, projections, OOD evaluation, and logit-lens interpretation.
- **Key files:** `src/extract/extract_activations.py`, `compute_steering_vectors.py`, `nb_pipeline.py`, `src/analysis/raid_eval.py`, `src/interpret/`.
- **Reuse:** best starting point for supervised human-vs-AI directions and nuisance/style analyses. Its “steering vectors” are used only as readout features in the paper, so causal activation addition is new work.
- **Requirements:** Python 3.11+, PyTorch 2.10+/CUDA 12.8, Transformers 4.57+, scikit-learn 1.8+.

## assistant-axis

- **URL / revision:** https://github.com/safety-research/assistant-axis, `a989619`
- **Purpose:** compute persona vectors/Assistant Axis and intervene with activation capping.
- **Key files:** `assistant_axis/axis.py`, `steering.py`, `pipeline/`, `notebooks/steer.ipynb`.
- **Reuse:** measure cosine/projection overlap with AI-ness and construct a persona-orthogonal direction; use capping or matched projection as a control.
- **Requirements:** large open-weight models (27B–70B in the paper), PyTorch/Transformers/vLLM, substantial GPU memory. Precomputed axes are linked in the README.

## representation-engineering

- **URL / revision:** https://github.com/andyzoujm/representation-engineering, `5455d8a`
- **Purpose:** generic representation-reading and control pipelines.
- **Key files:** `repe/rep_readers.py`, `rep_control_pipeline.py`, `rep_control_reading_vec.py`.
- **Reuse:** intervention plumbing and PCA/mean-difference baselines; adapt to the selected 8B-scale model.

## raid

- **URL / revision:** https://github.com/liamdugan/raid, `f450f4b`
- **Purpose:** official benchmark loading, detector execution, evaluation, and adversarial attacks.
- **Key files:** `raid/detect.py`, `raid/evaluate.py`, `generation/adversarial/`, CLI wrappers.
- **Reuse:** independent detector robustness evaluation, especially attacks and generator/domain breakdowns.
- **Caveat:** the checkout is about 2.6 GB because the upstream repository contains leaderboard artifacts. It is kept intact as the official implementation.

## chatgpt-comparison-detection

- **URL / revision:** https://github.com/Hello-SimpleAI/chatgpt-comparison-detection, `1f8c15c`
- **Purpose:** official HC3 detector and linguistic-analysis baselines.
- **Key files:** `detect/ml_train.py`, `detect/dl_train.py`, `linguistic_analysis/`.
- **Reuse:** independent RoBERTa/linguistic baselines. Syntax validation emits only legacy-regex warnings.

## Execution recommendation

Do not install every repository into the shared environment at once: their pinned PyTorch, NumPy, and Transformers versions conflict. Start with the root `.venv`, add one 8B-capable CUDA stack chosen for the experiment host, and vendor only the small functions needed from `mgt_probes`, `sv-detect`, and `assistant-axis`.
