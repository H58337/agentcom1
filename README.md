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
raw files and run:

```cmd
python main.py --dataset <dataset> --run_stage split --split_data True
```

## Workflow

Initialize user policies and the public skill bank:

```cmd
python main.py --dataset librarything --model_name com --model_type agent --tool_name SASRec --tool_type sequential --run_stage train --split_data False --com_bootstrap_user_policy_only True --com_rebuild_initial_user_policy True --com_llm_init_user_core_skill True --agent_workers 10
```

Train AgentCom with failure-driven skill evolution:

```cmd
python main.py --dataset librarything --model_name com --model_type agent --tool_name SASRec --tool_type sequential --run_stage train --split_data False --com_rebuild_initial_user_policy False --com_llm_evolve_user_skill True --com_tree_evolve_final_flush True --com_refresh_public_tree_layout True --agent_workers 10
```

Test with fixed skills:

```cmd
python main.py --dataset librarything --model_name com --model_type agent --tool_name SASRec --tool_type sequential --run_stage test --split_data False --com_tree_evolve_final_flush False --com_refresh_public_tree_layout False --agent_workers 10
```

Generated checkpoints, candidate sets, skills, and metrics are written to
`modelsaved/` and are excluded from version control.

## Reference

```text
Personalized Communication Skills for Agentic Recommender Systems
```
