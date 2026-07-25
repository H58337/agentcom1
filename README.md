# COM: Personalized Communication Skills for Agentic Recommendation

This repository contains a standalone implementation of COM, a multi-agent
recommendation framework in which a user agent consults socially selected
advisor agents before making a final choice. The release contains the COM
pipeline and its SASRec backbone, without attack models or unrelated baselines.

## Method Overview

COM maintains a public four-layer communication skill bank:

1. `why`: identify the decision deficiency that makes external advice useful.
2. `what`: turn the deficiency into a bounded advisor task.
3. `how`: select a single-advisor, cooperative, or competitive protocol.
4. `who`: retrieve trusted, similar, experienced, or two-hop social advisors.

The `similar-users` source is ranked by cosine similarity between user
representations produced by the trained SASRec sequence encoder. Public skills
and personalized user skills can evolve from failed training interactions; the
saved skills remain fixed during testing.

## Requirements

Use Python 3.10 or later.

```bash
pip install -r requirements.txt
```

For API-based LLM inference, configure an API key in PowerShell:

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
```


## Data Format

Processed data lives in `data/clean/<dataset>/` and uses RecBole-style files:

```text
<dataset>.inter
<dataset>.item
<dataset>.social
<dataset>.train.inter
<dataset>.valid.inter
<dataset>.test.inter
```

`librarything` is included. For a new dataset, provide the raw `.inter`,
`.item`, and `.social` files, then run the split stage. Users with fewer than
five interactions are filtered by default.

## SASRec Backbone and Candidates

COM always uses SASRec. On training, it loads an existing checkpoint or trains
one at:

```text
modelsaved/SASRec/<dataset>/clean/tool/SASRec_model.pth
```

It then writes candidate sets to:

```text
modelsaved/SASRec/<dataset>/candidates.json
modelsaved/SASRec/<dataset>/candidates_val.json
```

Each candidate set has 20 items by default: the held-out target plus the 19
highest-scored SASRec non-target items. COM also derives its PriorHint from
SASRec scores over the same candidate set. The same SASRec encoder produces
the user embeddings used by `similar-users` retrieval.

External candidate JSON or prior CSV files can still be supplied with
`--com_candidates_json_path`, `--com_candidates_val_json_path`,
`--com_prior_csv_path`, and `--com_prior_val_csv_path`. They override the
automatically generated artifacts, while SASRec remains the backbone used for
similar-user retrieval.

## Workflow

1. Split a new dataset:

```powershell
python main.py --dataset librarything --run_stage split --split_data True
```

2. Train SASRec, initialize COM skills, generate SASRec candidates/PriorHint,
   and evolve COM from training failures:

```powershell
python main.py --dataset librarything --run_stage train --split_data False `
  --com_save_dialogue True --agent_workers 10
```

3. Test with the saved SASRec checkpoint and fixed COM skills:

```powershell
python main.py --dataset librarything --run_stage test --split_data False `
  --com_save_dialogue True --agent_workers 10
```

Use `--run_stage train_test` to run both stages in one command. Set
`--sasrec_force_retrain True` to train SASRec again, or pass
`--sasrec_checkpoint_path` and `--sasrec_config_file_path` for custom paths.

## Outputs

- SASRec checkpoints and candidate sets: `modelsaved/SASRec/<dataset>/`
- COM user skills, public skill bank, metrics, and traces:
  `modelsaved/com/<dataset>/`

Run `python main.py --help` for all configuration options.
