# Downloaded Datasets

Data files are intentionally excluded from git. Run the downloader from the repository root with:

```bash
source .venv/bin/activate
python datasets/download_datasets.py
```

Pass `--raid-full` only when roughly 20 GB of free space and substantial download time are available. The default reproduces the 4,800-row RAID coverage sample used for resource validation.

## HC3

- **Source:** `Hello-SimpleAI/HC3`; CC-BY-SA-4.0 subject to stricter licenses of upstream sources.
- **Local location:** `datasets/HC3/all.jsonl` (73.7 MB).
- **Shape:** 24,322 question records; 58,546 human answers and 26,903 ChatGPT answers across Reddit ELI5, open QA, Wikipedia CS/AI, finance, and medicine.
- **Schema:** `question`, lists of `human_answers` and `chatgpt_answers`, `index`, `source`; no corruption found.
- **Use:** simple paired QA pilot and legacy-detector comparison. It is less clean for causal generation than HAP-E because answer counts and lengths differ.

Load with:

```python
import pandas as pd
hc3 = pd.read_json("datasets/HC3/all.jsonl", lines=True)
```

## RAID

- **Source:** `liamdugan/raid`; official ACL 2024 benchmark repository.
- **Full source size:** 16.7 GB in original CSV files; 4.31M labeled train/extra rows plus a 672K-row hidden-label test configuration on the dataset server.
- **Local location:** `datasets/raid/train_sample.parquet` (4,800 rows) plus the upstream README.
- **Sample coverage:** 11 machine generators plus human; 12 attack values; abstracts, news, books, and poetry. The evenly spaced sample is deliberately a validation/evaluation convenience, not a statistically balanced training set (only 100 human rows).
- **Schema:** IDs, generator, decoding, repetition penalty, attack, domain, title, prompt, and generation. Null decoding/prompt fields occur exactly on human rows and are expected.
- **Use:** adversarial and generator-shift stress tests after training on HAP-E/MAGE. For publication results, use the complete benchmark or construct a balanced subset from it.

Full download alternative:

```python
from datasets import load_dataset
raid = load_dataset("liamdugan/raid")
raid.save_to_disk("datasets/raid_full")
```

## Human-AI Parallel English Corpus (HAP-E)

- **Source:** `browndw/human-ai-parallel-corpus`; MIT.
- **Local location:** `datasets/human-ai-parallel-corpus/text_data/` (all eight parquet files, 114 MB total repository payload).
- **Shape:** 8,290 aligned document IDs in every file; two human chunks and outputs from GPT-4o, GPT-4o-mini, Llama-3 8B/70B base and instruct variants. Each file has `doc_id`, `text`, and no missing values.
- **Domains:** academic, blogs, fiction, news, spoken transcripts, and TV/movie scripts.
- **Use:** primary direction-estimation dataset. Human chunk 1 is the shared prompt; human chunk 2 and each model continuation are content-aligned outcomes, making this the strongest resource for controlling topic, length, and domain. Exclude chunk 1 from the human-vs-AI label task.

Load and align with:

```python
import pandas as pd
human = pd.read_parquet("datasets/human-ai-parallel-corpus/text_data/hape-text_human-chunk-2.parquet")
gpt4o = pd.read_parquet("datasets/human-ai-parallel-corpus/text_data/hape-text_gpt-4o-2024-08-06.parquet")
paired = human.merge(gpt4o, on="doc_id", suffixes=("_human", "_ai"))
```

## MAGE

- **Source:** `yaful/MAGE`; Apache-2.0 dataset card.
- **Local location:** `datasets/MAGE/` (all released CSV splits, about 554 MB).
- **Shape:** train 319,071; validation 56,792; test 56,819; GPT OOD 1,562; paraphrased GPT OOD 2,362. Columns are `text`, `label`, `src`; no missing values.
- **Use:** broad detector training and held-out generator/domain evaluation. It is not content-paired, so nuisance-controlled direction discovery should use HAP-E first.

Load with:

```python
import pandas as pd
train = pd.read_csv("datasets/MAGE/train.csv")
```

## Small examples

Human-readable examples are in `datasets/samples/` and are each below 100 KB. They document schemas only and should not be used for statistical conclusions.
