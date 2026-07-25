import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true"):
        return True
    if v in ("false"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def agent_parse_args(return_parser=False):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--MAX_ITEM_LIST_LENGTH", type=int, default=50, help="maximum interaction history length")
    parser.add_argument("--MIN_ITEM_LIST_LENGTH", type=int, default=5, help="minimum interactions retained during splitting")
    parser.add_argument(
        "--sasrec_config_file_path",
        type=str,
        default="",
        help="optional RecBole SASRec YAML; defaults to tool_conf/default/SASRec.yaml",
    )
    parser.add_argument(
        "--sasrec_checkpoint_path",
        type=str,
        default="",
        help="optional SASRec checkpoint; defaults to modelsaved/SASRec/<dataset>/clean/tool/SASRec_model.pth",
    )
    parser.add_argument(
        "--sasrec_force_retrain",
        type=str2bool,
        default=False,
        help="retrain SASRec even when its checkpoint already exists",
    )
    parser.add_argument(
        "--sasrec_candidate_num",
        type=int,
        default=20,
        help="candidate-set size: target item plus highest-scored SASRec non-target items",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        help="OpenAI-compatible API key; set DEEPSEEK_API_KEY or pass --api_key at runtime",
    )

    parser.add_argument(
    "--model",
    type=str,
    default="deepseek-v4-flash",
    help="LLM model name",
)
    parser.add_argument(
        "--llm_input_price_per_mtoken",
        type=float,
        default=0.0,
        help="input token price per 1M tokens for LLM cost estimation",
    )
    parser.add_argument(
        "--llm_output_price_per_mtoken",
        type=float,
        default=0.0,
        help="output token price per 1M tokens for LLM cost estimation",
    )
    parser.add_argument(
        "--llm_cost_currency",
        type=str,
        default="USD",
        help="currency label for LLM cost estimation",
    )
    parser.add_argument(
        "--llm_request_timeout",
        type=float,
        default=120.0,
        help="per LLM HTTP request timeout in seconds",
    )
    parser.add_argument("--agent", type=str2bool, default=True, help="whether to use agent recommender mode")
    parser.add_argument("--agent_workers", type=int, default=10, help="number of agent worker threads")
    parser.add_argument("--local_model_path", type=str, default="", help="optional local model path for Qwen3-8B")
    parser.add_argument("--com_progress_interval", type=int, default=20, help="COM train/test progress file write interval; <=0 disables")
    parser.add_argument(
        "--com_print_diagnostics",
        type=str2bool,
        default=False,
        help="print COM diagnostic progress and summaries; metrics remain saved to files regardless",
    )
    parser.add_argument(
        "--com_save_diagnostics",
        type=str2bool,
        default=False,
        help="save COM dialogue, runtime, progress, usage, and analysis diagnostics; final test metrics are always saved",
    )


    parser.add_argument("--com_max_rounds", type=int, default=3, help="max negotiation rounds for com agent")
    parser.add_argument(
        "--com_user_advisor_rounds",
        type=int,
        default=None,
        help="maximum user-advisor interaction rounds; when unset, falls back to --com_max_communication_rounds and then --com_max_rounds",
    )
    parser.add_argument("--com_pathc_margin", type=float, default=0.05, help="score margin to trigger path C exploration")
    parser.add_argument(
        "--com_enable_tree_trial_exploration",
        type=str2bool,
        default=False,
        help=(
            "enable train-time public-tree sprout/anchor trial exploration. "
            "Default false keeps COM path selection purely driven by each user's communication_route_skill."
        ),
    )
    parser.add_argument(
        "--com_advisor_topk",
        type=int,
        default=3,
        help="maximum advisor speakers used by multi-advisor communication paths",
    )
    parser.add_argument(
        "--com_advisor_discussion_rounds",
        type=int,
        default=1,
        help="number of sequential advisor-to-advisor discussion passes before the summary agent; applies to multi-advisor paths only",
    )
    parser.add_argument(
        "--com_summary_max_tokens",
        type=int,
        default=3000,
        help="max completion tokens for COM advisor summary agent",
    )
    parser.add_argument(
        "--com_advisor_policy",
        type=str,
        default="single",
        help="friend selection policy: single (1-to-1) or topk",
    )
    parser.add_argument(
        "--com_advisor_social_file",
        type=str,
        default=None,
        help="optional social file path for explicit friend mode; default {dataset}.social under data dir",
    )
    parser.add_argument(
        "--com_refresh_communication_initial_evidence",
        type=str2bool,
        default=False,
        help="refresh persisted user communication bootstrap evidence from the current social graph before training",
    )
    parser.add_argument(
        "--com_refresh_public_tree_layout",
        type=str2bool,
        default=False,
        help="reset public_tree to the code-defined initial active nodes and clear generated tree-evolution nodes/indexes before running COM; set true only for a fresh experiment reset",
    )
    parser.add_argument(
        "--com_debug_round",
        type=str2bool,
        default=False,
        help="print per-round COM negotiation logs",
    )
    parser.add_argument(
        "--com_debug_user_limit",
        type=int,
        default=5,
        help="print round logs for first N users only when com_debug_round=True",
    )
    parser.add_argument(
        "--com_save_dialogue",
        type=str2bool,
        default=True,
        help="persist COM per-user negotiation traces to files during test",
    )
    parser.add_argument(
        "--com_dialogue_user_limit",
        type=int,
        default=0,
        help="max users to save dialogue traces for; 0 means all test users",
    )
    parser.add_argument(
        "--com_dialogue_include_history",
        type=str2bool,
        default=False,
        help="include user history text in saved dialogue traces",
    )
    parser.add_argument(
        "--com_test_sample_num",
        type=int,
        default=0,
        help="number of random test users for COM inference; 0 means all",
    )
    parser.add_argument(
        "--com_test_sample_seed",
        type=int,
        default=2026,
        help="random seed for COM test user sampling",
    )
    parser.add_argument(
        "--com_test_sample_offset",
        type=int,
        default=-1,
        help=(
            "non-negative offset for deterministic COM test user windows; "
            "use 0,200,400... with --com_test_sample_num for resumable test batches; "
            "-1 keeps legacy random sampling"
        ),
    )
    parser.add_argument(
        "--com_test_sample_order",
        type=str,
        default="random",
        choices=["random", "sorted"],
        help="order used when --com_test_sample_offset is non-negative: random seed-shuffled window or sorted user-id window",
    )
    parser.add_argument(
        "--com_train_sample_num",
        type=int,
        default=0,
        help="number of random train users for COM interaction/bootstrap; 0 means use com_test_sample_num or all",
    )
    parser.add_argument(
        "--com_train_sample_offset",
        type=int,
        default=-1,
        help=(
            "non-negative offset for deterministic COM train user windows; "
            "use 0,100,200... with --com_train_sample_num for resumable formal batches; "
            "-1 keeps legacy random sampling"
        ),
    )
    parser.add_argument(
        "--com_train_sample_order",
        type=str,
        default="random",
        choices=["random", "sorted"],
        help="order used when --com_train_sample_offset is non-negative: random seed-shuffled window or sorted user-id window",
    )
    parser.add_argument(
        "--com_fixed_user_ids",
        type=str,
        default="",
        help="comma/space separated raw user ids for fixed COM train/test cohorts; overrides random sampling when set",
    )
    parser.add_argument(
        "--com_prior_csv_path",
        type=str,
        default="",
        help="optional external prior csv path for COM test stage",
    )
    parser.add_argument(
        "--com_prior_val_csv_path",
        type=str,
        default="",
        help="optional external prior csv path for COM train/val stage",
    )
    parser.add_argument(
        "--com_prior_user_col",
        type=str,
        default="user_id",
        help="user id column name in external COM prior csv",
    )
    parser.add_argument(
        "--com_prior_item_col",
        type=str,
        default="generate",
        help="item column name in external COM prior csv",
    )
    parser.add_argument(
        "--com_candidates_json_path",
        type=str,
        default="",
        help="optional external candidates json for COM test stage",
    )
    parser.add_argument(
        "--com_candidates_val_json_path",
        type=str,
        default="",
        help="optional external candidates json for COM val stage",
    )
    parser.add_argument(
        "--com_profile_cache_path",
        type=str,
        default="",
        help="optional profile cache file for COM (json or jsonl keyed by user_raw)",
    )
    parser.add_argument(
        "--com_profile_workers",
        type=int,
        default=1,
        help="worker threads for COM profile generation",
    )
    parser.add_argument(
        "--com_enable_skill_framework",
        type=str2bool,
        default=True,
        help="enable staged COM skill framework (internal proposal -> skill selection -> memory update)",
    )
    parser.add_argument(
        "--com_skill_confidence_no_comm",
        type=int,
        default=85,
        help="confidence threshold above which COM can skip communication and finalize directly",
    )
    parser.add_argument(
        "--com_skill_compare_margin",
        type=float,
        default=0.05,
        help="top-2 candidate score margin below which COM marks candidate_comparison uncertainty",
    )
    parser.add_argument(
        "--com_skill_shortlist_limit",
        type=int,
        default=3,
        help="number of candidates kept in the internal COM shortlist",
    )
    parser.add_argument(
        "--com_preserve_initial_skills",
        type=str2bool,
        default=True,
        help="save immutable initial COM public-tree and user-skill snapshots before train-time evolution",
    )
    parser.add_argument(
        "--com_reset_skill_state",
        type=str2bool,
        default=False,
        help="archive current trained COM skills and restart from the preserved initial skill state",
    )
    parser.add_argument(
        "--com_rebuild_initial_user_policy",
        type=str2bool,
        default=False,
        help="overwrite COM initial/runtime user policies from current initialization logic during train bootstrap",
    )
    parser.add_argument(
        "--com_rebuild_communication_user_policy_only",
        type=str2bool,
        default=False,
        help="reinitialize only COM communication_route_skill during train bootstrap while preserving item_selection_skill",
    )
    parser.add_argument(
        "--com_llm_init_user_core_skill",
        type=str2bool,
        default=True,
        help="use LLM to initialize each user's core item reasoning skill when a user policy is bootstrapped or rebuilt",
    )
    parser.add_argument(
        "--com_llm_evolve_user_skill",
        type=str2bool,
        default=True,
        help="use LLM counterfactual reflection to evolve each user's reasoning skill after train interactions",
    )
    parser.add_argument(
        "--com_llm_evolve_item_skill",
        type=str2bool,
        default=True,
        help="use LLM reflection for Stage1 item_selection_skill updates; set False to keep item skill frozen while preserving communication/tree evolution",
    )
    parser.add_argument(
        "--com_bootstrap_user_policy_only",
        type=str2bool,
        default=False,
        help="only bootstrap/rebuild COM user policies, then stop before interaction training",
    )
    parser.add_argument(
        "--com_stage1_only",
        type=str2bool,
        default=False,
        help="train/evaluate only COM user initial item selection and hesitation shortlist; skip communication, advisor feedback, redecision, and public-tree evolution",
    )
    parser.add_argument(
        "--com_max_advisors",
        type=int,
        default=2,
        help="maximum advisors retrieved for a multi-advisor COM path; multi paths always use at least two when available",
    )
    parser.add_argument(
        "--com_train_item_during_communication",
        type=str2bool,
        default=False,
        help="during full COM train, also update item_selection_skill for users whose Stage1 initial proposal misses; keep False for later communication-only replay rounds",
    )
    parser.add_argument(
        "--com_reuse_cached_stage1_slim",
        type=str2bool,
        default=False,
        help="reuse user assets/slim_cache_proposal.json for Stage1 item-skill retrieval when available; useful after item_selection_skill is frozen, but less candidate-context-specific",
    )
    parser.add_argument(
        "--com_train_communication_eligible_only",
        type=str2bool,
        default=True,
        help="during COM train, run advisor communication and communication-skill updates only when the train target is in the Stage1 focus/hesitation set",
    )
    parser.add_argument(
        "--com_write_failed_user_queue",
        type=str2bool,
        default=True,
        help="write TW/WW business-failure users after COM train for optional replay in a later run",
    )
    parser.add_argument(
        "--com_failed_user_queue_path",
        type=str,
        default="",
        help="optional failed-user queue jsonl path for COM train replay",
    )
    parser.add_argument(
        "--com_replay_failed_users_only",
        type=str2bool,
        default=False,
        help="during COM train, process only users loaded from --com_failed_user_queue_path",
    )
    parser.add_argument(
        "--com_tree_evolve_batch_size",
        type=int,
        default=50,
        help="run public communication tree batch evolution after this many pending effective train communication/tree diagnoses; <=0 means only run once at train end",
    )
    parser.add_argument(
        "--com_tree_evolve_final_flush",
        type=str2bool,
        default=True,
        help="when train ends, run public tree evolution for any pending buffer rows; set False for collect-only small/debug runs to avoid LLM-1/LLM-2 calls before the batch threshold is reached",
    )
    parser.add_argument(
        "--com_max_communication_rounds",
        type=int,
        default=None,
        help="bounded communication rounds for COM; when unset, falls back to --com_max_rounds",
    )
    parser.add_argument(
        "--com_enable_shareable_user_brief",
        type=str2bool,
        default=True,
        help="pass advisors a privacy-filtered requester item brief instead of the requester full private slim skill",
    )
    parser.add_argument(
        "--com_enable_advisor_own_skill",
        type=str2bool,
        default=True,
        help="include each advisor's own lightweight item-skill/profile evidence in advisor prompts when available",
    )
    parser.add_argument(
        "--com_why_trigger_only",
        type=str2bool,
        default=True,
        help='derive communication action from matched/selected why nodes instead of asking for an independent action decision',
    )
    parser.add_argument(
        "--com_skip_ineligible_advisor_cost",
        type=str2bool,
        default=True,
        help="when communication training gate marks a train sample ineligible, skip advisor calls and return a diagnostic trace",
    )
    parser.add_argument(
        "--com_ensure_target_in_candidates",
        type=str2bool,
        default=False,
        help="optionally repair COM candidates by appending the current target item when it is absent; disabled by default to keep experiments faithful to the provided candidate set",
    )
    parser.add_argument(
        "--com_strict_target_candidate_check",
        type=str2bool,
        default=False,
        help="raise an error instead of repairing when the current target item is absent from COM candidates",
    )
    parser.add_argument(
        "--com_train_force_target_in_focus",
        type=str2bool,
        default=False,
        help="optionally force the current train target into the communication shortlist/focus; disabled by default so focus reflects user/system selection rather than oracle target injection",
    )
    parser.add_argument(
        "--com_slow_user_log_seconds",
        type=float,
        default=30.0,
        help="print a COM slow-user line when one user interaction takes at least this many seconds",
    )
    parser.add_argument(
        "--com_social_path",
        type=str,
        default="",
        help="optional social file path for COM profile generation",
    )
    parser.add_argument(
        "--com_inter_path",
        type=str,
        default="",
        help="optional raw .inter path for COM profile generation",
    )
    parser.add_argument(
        "--com_item_path",
        type=str,
        default="",
        help="optional raw .item path for COM profile generation",
    )

    if return_parser:
        return parser
    return parser.parse_known_args()[0]
