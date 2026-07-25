"""Shared stage semantics for sequence-based agent recommenders.

Agent experiments use the last two interactions of each user's sequence:
- agent train: predict the validation item from ``*.valid.inter``
- agent test: predict the test item from ``*.test.inter``

This keeps COM inputs aligned with candidates and optional priors prepared for
the same stage.
"""


def normalize_agent_stage(stage):
    stage_key = str(stage or "test").strip().lower()
    if stage_key in {"train", "com_train", "agent_train"}:
        return "train"
    if stage_key in {"val", "valid", "validation"}:
        return "val"
    if stage_key == "test":
        return "test"
    return stage_key


def source_for_agent_stage(arlib_dataset, stage):
    stage_key = normalize_agent_stage(stage)
    if stage_key in {"train", "val"}:
        return getattr(arlib_dataset, "val_data", []), "agent_train_valid_item"
    if stage_key == "test":
        return getattr(arlib_dataset, "test_data", []), "agent_test_item"
    return getattr(arlib_dataset, "training_data", []), "arlib_training_item"


def candidate_suffix_for_agent_stage(stage):
    stage_key = normalize_agent_stage(stage)
    return "_val" if stage_key in {"train", "val"} else ""
