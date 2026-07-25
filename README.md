# AgentCom: Personalized Communication Skills for Agentic Recommender Systems

This repository contains the implementation of **AgentCom**, a personalized
communication skill framework for agentic recommender systems. A target
UserAgent first makes a provisional decision over recommender-generated
candidates, consults selected advisor agents, and then makes the final choice.

AgentCom maintains a public `why -> what -> how -> who` communication skill
bank:

1. `why`: identify the decision deficiency that makes advice useful.
2. `what`: define the information task for advisors.
3. `how`: select a single-advisor, cooperative, or competitive protocol.
4. `who`: retrieve trusted, similar, experienced, or friend-of-friend advisors.

During training, failed communication cases are analyzed to refine existing
skills, add missing skills, or correct the personalized route. Skills are fixed
during testing. This release uses SASRec as the candidate-generation backbone.

## Requirements

Use Python 3.10 or later.

```bash
pip install -r requirements.txt
```

Before an LLM-based stage, configure an API key privately:

```cmd
set DEEPSEEK_API_KEY=your-api-key
```

The default LLM is `deepseek-v4-flash`; use `--model <model-name>` to change it.

## Data

Processed data is stored in `data/clean/<dataset>/` using RecBole-style files:

```text
<dataset>.inter
<dataset>.item
<dataset>.social
<dataset>.train.inter
<dataset>.valid.inter
<dataset>.test.inter
```

The processed `librarything` dataset is included. For a new dataset, add the
raw files and use the split stage provided by `main.py`.

## Usage

Use `main.py` to prepare a dataset, initialize AgentCom, train the skill bank,
and evaluate the fixed skills. Run `python main.py --help` to view available
configuration options.

Generated checkpoints, candidate sets, skills, and metrics are written to
`modelsaved/` and are excluded from version control.

## Reference

```text
Personalized Communication Skills for Agentic Recommender Systems
```
