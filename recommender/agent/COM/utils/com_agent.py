import json
import os
import threading
import time
from types import SimpleNamespace

from recommender.agent.COM.utils.api_request import api_request, pop_last_llm_call_usage

_PROMPT_TRACE_LOCAL = threading.local()
_PHASE_USAGE_LOCK = threading.Lock()
_LLM_PHASE_USAGE = {}


def set_llm_prompt_trace(trace_sink, base_context=None):
    _PROMPT_TRACE_LOCAL.trace_sink = trace_sink
    _PROMPT_TRACE_LOCAL.base_context = dict(base_context or {})
    _PROMPT_TRACE_LOCAL.call_index = 0


def update_llm_prompt_trace_context(**context):
    current = dict(getattr(_PROMPT_TRACE_LOCAL, "base_context", {}) or {})
    current.update({k: v for k, v in dict(context or {}).items() if v is not None})
    _PROMPT_TRACE_LOCAL.base_context = current


def clear_llm_prompt_trace():
    _PROMPT_TRACE_LOCAL.trace_sink = None
    _PROMPT_TRACE_LOCAL.base_context = {}
    _PROMPT_TRACE_LOCAL.call_index = 0


def reset_llm_phase_usage_stats():
    with _PHASE_USAGE_LOCK:
        _LLM_PHASE_USAGE.clear()


def get_llm_phase_usage_stats(reset=False):
    with _PHASE_USAGE_LOCK:
        snapshot = {
            str(phase): {
                "calls": int(row.get("calls", 0)),
                "prompt_tokens": int(row.get("prompt_tokens", 0)),
                "completion_tokens": int(row.get("completion_tokens", 0)),
                "total_tokens": int(row.get("total_tokens", 0)),
                "estimated_cost": float(row.get("estimated_cost", 0.0)),
            }
            for phase, row in _LLM_PHASE_USAGE.items()
        }
    if reset:
        reset_llm_phase_usage_stats()
    return snapshot


def get_last_llm_request_usage():
    usage = getattr(_PROMPT_TRACE_LOCAL, "last_llm_usage", None)
    return dict(usage or {}) if isinstance(usage, dict) else {}


def _current_llm_phase():
    context = dict(getattr(_PROMPT_TRACE_LOCAL, "base_context", {}) or {})
    return str(context.get("phase", "") or "unknown")


def _record_llm_phase_usage(usage):
    if not isinstance(usage, dict):
        return
    phase = _current_llm_phase()
    with _PHASE_USAGE_LOCK:
        row = _LLM_PHASE_USAGE.setdefault(
            phase,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
            },
        )
        row["calls"] += 1
        row["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        row["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        row["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        row["estimated_cost"] += float(usage.get("estimated_cost", 0.0) or 0.0)


def _pop_api_call_usage():
    return pop_last_llm_call_usage()


def _append_llm_prompt_trace(system_prompt, user_prompt, response, args, error=None, llm_usage=None):
    trace_sink = getattr(_PROMPT_TRACE_LOCAL, "trace_sink", None)
    if not isinstance(trace_sink, list):
        return
    try:
        import inspect

        caller = None
        for frame in inspect.stack()[2:8]:
            filename = str(frame.filename or "")
            if filename.replace("\\", "/").endswith("utils/com_agent.py") and frame.function == "llm_request":
                continue
            caller = frame
            break
        callsite = {}
        if caller is not None:
            callsite = {
                "file": str(caller.filename or ""),
                "function": str(caller.function or ""),
                "line": int(caller.lineno or 0),
            }
    except Exception:
        callsite = {}

    try:
        call_index = int(getattr(_PROMPT_TRACE_LOCAL, "call_index", 0) or 0) + 1
        _PROMPT_TRACE_LOCAL.call_index = call_index
    except Exception:
        call_index = 0

    context = dict(getattr(_PROMPT_TRACE_LOCAL, "base_context", {}) or {})
    response_text = "" if response is None else str(response)
    row = {
        "event": "llm_prompt_io",
        "llm_call_index": int(call_index),
        "timestamp": time.time(),
        "model": str(getattr(args, "model", "") or ""),
        "context": context,
        "callsite": callsite,
        "system_prompt_chars": len(str(system_prompt or "")),
        "user_prompt_chars": len(str(user_prompt or "")),
        "response_chars": len(response_text),
        "llm_usage": dict(llm_usage or {}),
        "response_is_none": response is None,
        "error": str(error or "") if error else "",
        "system_prompt": str(system_prompt or ""),
        "user_prompt": str(user_prompt or ""),
        "response": response_text,
    }
    trace_sink.append(row)


def _call_api_request(system_prompt, user_prompt, args):
    return api_request(system_prompt, user_prompt, args)

from recommender.agent.COM.utils.com_parser import (
    parse_advisor_review,
    parse_communication_control,
    parse_communication_path_choice,
    parse_friend_summary,
    parse_self_confidence,
    parse_stage6_decision,
    parse_user_final_decision,
    parse_user_proposal,
    parse_user_proposal_decision,
)


def llm_request(system_prompt, user_prompt, args):
    import logging
    logger = logging.getLogger(__name__)
    bot = getattr(args, "local_bot", None)
    lock = getattr(args, "local_bot_lock", None)
    response = None
    error = None
    call_usage = None
    usage_recorded = False
    _PROMPT_TRACE_LOCAL.last_llm_usage = {}

    if bot is not None:
        try:
            if lock is None:
                response = bot.chat(system_prompt, user_prompt)
                return response
            with lock:
                response = bot.chat(system_prompt, user_prompt)
                return response
        except Exception as e:
            error = repr(e)
            logger.warning("[llm_request] local_bot.chat() failed: %s", e)
            return None
        finally:
            _PROMPT_TRACE_LOCAL.last_llm_usage = dict(call_usage or {})
            _append_llm_prompt_trace(system_prompt, user_prompt, response, args, error=error, llm_usage=call_usage)

    try:
        response = _call_api_request(system_prompt, user_prompt, args)
        call_usage = _pop_api_call_usage()
        if call_usage:
            _record_llm_phase_usage(call_usage)
            usage_recorded = True
        return response
    except Exception as e:
        error = repr(e)
        raise
    finally:
        if call_usage is None:
            call_usage = _pop_api_call_usage()
        if call_usage and not usage_recorded:
            _record_llm_phase_usage(call_usage)
        _PROMPT_TRACE_LOCAL.last_llm_usage = dict(call_usage or {})
        _append_llm_prompt_trace(system_prompt, user_prompt, response, args, error=error, llm_usage=call_usage)


def _render_skill_payload(slim_user_policy):
    if not slim_user_policy:
        return "none"
    if isinstance(slim_user_policy, str):
        return slim_user_policy
    try:
        return json.dumps(slim_user_policy, ensure_ascii=False)
    except Exception:
        return str(slim_user_policy)


def _render_requester_brief(target_user_skill):
    if not target_user_skill:
        return "none"
    if isinstance(target_user_skill, str):
        return target_user_skill
    try:
        payload = dict(target_user_skill or {})
    except Exception:
        return str(target_user_skill)
    if "requester_shareable_item_brief" in payload:
        brief = dict(payload.get("requester_shareable_item_brief", {}) or {})
    else:
        brief = payload
    rows = []
    prefs = brief.get("relevant_preference_summary", brief.get("preferences", []))
    if isinstance(prefs, str):
        prefs = [prefs]
    prefs = [str(x).strip() for x in list(prefs or []) if str(x).strip()]
    if prefs:
        rows.append("likes/style: " + " | ".join(prefs[:3]))
    uncertainty = brief.get("current_uncertainty", brief.get("uncertainty_points", []))
    if isinstance(uncertainty, str):
        uncertainty = [uncertainty]
    uncertainty = [str(x).strip() for x in list(uncertainty or []) if str(x).strip()]
    if uncertainty:
        rows.append("current_uncertainty: " + ", ".join(uncertainty[:6]))
    style = str(brief.get("style", "") or "").strip()
    if style:
        rows.append("style: " + style)
    return "\n".join(rows) if rows else "none"


def _render_tree_options(public_tree_options):
    if not public_tree_options:
        return "none"
    try:
        return json.dumps(public_tree_options, ensure_ascii=False, indent=2)
    except Exception:
        return str(public_tree_options)


def _option_ids(public_tree_options, level):
    values = (dict(public_tree_options or {}).get(level, []) or [])
    ids = []
    for row in values:
        if isinstance(row, dict):
            node_id = str(row.get("node_id", "") or "").strip()
        else:
            node_id = str(row or "").strip()
        if node_id:
            ids.append(node_id)
    return ids


def _confidence_band(value):
    try:
        value = int(value or 0)
    except Exception:
        value = 0
    if value >= 85:
        return "high"
    if value >= 60:
        return "medium"
    return "low"


def _compact_comm_rows(rows, allowed_keys=None, limit=3):
    allowed = set(allowed_keys or [])
    compact = []
    for row in list(rows or [])[: max(0, int(limit))]:
        if not isinstance(row, dict):
            continue
        out = {}
        for key in allowed:
            if key in row and row.get(key) not in [None, ""]:
                value = row.get(key)
                if key == "path" and isinstance(value, dict):
                    out[key] = {
                        sub_key: value.get(sub_key)
                        for sub_key in ['why', "who", "how"]
                        if value.get(sub_key) not in [None, ""]
                    }
                elif isinstance(value, (str, int, float, bool)):
                    out[key] = value
        if "confidence" in row:
            try:
                out["confidence"] = round(float(row.get("confidence", 0.0) or 0.0), 3)
            except Exception:
                pass
        if "confidence_label" in row:
            out["confidence_label"] = row.get("confidence_label")
        if out:
            compact.append(out)
    return compact


def _compact_reliability_bias(reliability_bias):
    out = {}
    for who, stats in dict(reliability_bias or {}).items():
        if not isinstance(stats, dict):
            continue
        compact_stats = {}
        for key, value in stats.items():
            if isinstance(value, (int, float, bool)):
                compact_stats[str(key)] = value
            elif str(key) in ["advisor_id", "who", "advisor_source", "reliability", "outcome"] and isinstance(value, str):
                compact_stats[str(key)] = value
        if compact_stats:
            out[str(who)] = compact_stats
    return out


def _communication_skill_payload(slim_user_policy):
    if not isinstance(slim_user_policy, dict):
        return {
            "communication_selection_skill": "unstructured skill omitted to avoid item-level prompt leakage",
        }
    comm_skill = dict((slim_user_policy or {}).get("communication_selection_skill", {}) or {})
    payload = {
        "phase": str((slim_user_policy or {}).get("phase", "") or ""),
        "communication_selection_skill": {
            "trigger_strength": comm_skill.get("trigger_strength", 0.5),
            "top_who_preferences": _compact_comm_rows(
                comm_skill.get("top_who_preferences", []),
                allowed_keys=["attribute", "who", "advisor_source", "advisor_id"],
                limit=3,
            ),
            "top_how_preferences": _compact_comm_rows(
                comm_skill.get("top_how_preferences", []),
                allowed_keys=["attribute", "how", "mode"],
                limit=3,
            ),
            "trigger_rules": _compact_comm_rows(
                comm_skill.get("trigger_rules", []),
                allowed_keys=["condition", "mode", 'why', "trigger"],
                limit=3,
            ),
            "path_memory": _compact_comm_rows(
                comm_skill.get("path_memory", []),
                allowed_keys=['why', "who", "how", "outcome", "path", "advisor_id"],
                limit=3,
            ),
            "communication_round_memory": _compact_comm_rows(
                comm_skill.get("communication_round_memory", []),
                allowed_keys=['why', "who", "how", "outcome", "round_result", "advisor_id"],
                limit=3,
            ),
            "advisor_reliability_memory": _compact_comm_rows(
                comm_skill.get("advisor_reliability_memory", []),
                allowed_keys=["who", "advisor_source", "advisor_id", "reliability", "outcome"],
                limit=3,
            ),
        },
        "advisor_reliability_bias": _compact_reliability_bias(
            (slim_user_policy or {}).get("retrieved_reliability_bias", {})
        ),
    }
    return payload


def _communication_memory_summary(updated_memory):
    if not updated_memory or str(updated_memory).strip().lower() in ["none", "null", ""]:
        return "none"
    if not isinstance(updated_memory, dict):
        return {"exists": True}
    evidence_packet = dict(updated_memory.get("evidence_packet", {}) or {})
    summary = {}
    remaining = [str(x) for x in list(updated_memory.get("remaining_uncertainty", []) or [])[:3]]
    if remaining:
        summary["remaining_uncertainty"] = remaining
    if len(list(evidence_packet.get("silent_focus_candidates", []) or [])) > 0:
        summary["missing_evidence"] = True
    if bool(updated_memory.get("candidate_comparison")):
        summary["comparison_gap"] = True
    if bool(updated_memory.get("advisor_reliability_observation")):
        summary["advisor_reliability_signal"] = True
    return summary or "none"


def _communication_planning_state(decision_state, phase, updated_memory="none"):
    decision_state = dict(decision_state or {})
    uncertainty_points = [str(x) for x in list(decision_state.get("uncertainty_points", []) or [])]
    hesitation_size = len(list(decision_state.get("shortlist", []) or decision_state.get("candidate_shortlist", []) or []))
    if hesitation_size <= 1:
        hesitation_scope = "single"
    elif hesitation_size <= 3:
        hesitation_scope = "small"
    else:
        hesitation_scope = "broad"
    state = {
        "phase": str(phase or "initial"),
        "uncertainty": uncertainty_points or ["none"],
        "confidence": _confidence_band(decision_state.get("self_confidence", 0)),
        "hesitation_scope": hesitation_scope,
    }
    previous_round = _communication_memory_summary(updated_memory)
    if previous_round != "none":
        state["previous_round"] = previous_round
    return state


def _render_allowed_candidates(candidate_names):
    return ", ".join([str(x) for x in (candidate_names or []) if str(x or "").strip()]) or "none"


def _advisor_listening_rule():
    return (
        "AdvisorPerspectiveRule: treat friend feedback as one additional source of perspective, not as a command, vote, "
        "or higher-priority rule. Independently compare it with UserReasoningSkillSlim, History, and the current evidence. "
        "Keeping the CurrentProposal and changing it are both valid: retain it when it remains better supported, and change "
        "it when a specific comparison makes another candidate more convincing. State the evidence that supports the chosen option."
    )


def _is_product_domain(args):
    dataset = str(getattr(args, "dataset", "") or "").lower()
    return "epinions" in dataset


def _is_book_domain(args):
    dataset = str(getattr(args, "dataset", "") or "").lower()
    return "librarything" in dataset


def _item_selection_domain_text(args):
    if _is_product_domain(args):
        return {
            "hesitation_slot": "closest product category/use-case/feature signal or uncertainty source",
            "coverage": (
                "CandidateCoverageProtocol: before choosing Item, explicitly consider every candidate in "
                "CandidateSet against the user's long-term product categories/use-cases, recent need signal, "
                "and item-level product evidence: category, brand/manufacturer family, feature/function, price/value, "
                "quality/durability, reliability, design/form factor, compatibility/accessory relationship, "
                "review/rating sentiment, and substitute/complement bridges. Do not stop after finding matches to "
                "the most obvious category."
            ),
            "shortlist": (
                "Prefer 3-5 names. Preserve candidates that match stable minority product categories/use-cases, "
                "recent need drift, close substitutes or complements to Item, PriorHint conflicts, or products with "
                "plausible history bridges but incomplete evidence. "
            ),
        }
    if _is_book_domain(args):
        return {
            "hesitation_slot": "closest book genre/topic/author-style signal or uncertainty source",
            "coverage": (
                "CandidateCoverageProtocol: before choosing Item, explicitly consider every candidate in "
                "CandidateSet against the user's long-term book genres/topics/author-style clusters, recent reading drift, "
                "and item-level book evidence: fiction/non-fiction genre, literary form, topic/subject, author style, "
                "narrative tone, era/setting, cultural-language signal, audience/age category, series/franchise relation, "
                "canonical/award/niche level, and adjacent theme bridges. Do not stop after finding matches to "
                "the most obvious genre/topic."
            ),
            "shortlist": (
                "Prefer 3-5 names. Preserve candidates that match stable minority book genres/topics/author styles, "
                "recent reading drift, close thematic or literary-form alternatives to Item, PriorHint conflicts, or books with "
                "plausible history bridges but incomplete evidence. "
            ),
        }
    return {
        "hesitation_slot": "closest taste cluster or uncertainty source",
        "coverage": (
            "CandidateCoverageProtocol: before choosing Item, explicitly consider every candidate in "
            "CandidateSet against the user's long-term clusters, recent-interaction signal, and item-level "
            "music evidence. Do not stop after finding matches to the most obvious cluster."
        ),
        "shortlist": (
            "Prefer 3-5 names. Preserve candidates that match stable minority taste clusters, recent-history drift, "
            "close alternatives to Item, PriorHint conflicts, or candidates with plausible history bridges but "
            "incomplete evidence. "
        ),
    }


def _mask_secret(secret, keep=4):
    s = str(secret or "")
    if not s:
        return "missing"
    if len(s) <= int(keep):
        return "*" * len(s)
    return ("*" * (len(s) - int(keep))) + s[-int(keep) :]


_RUNTIME_LOG_LOCK = threading.Lock()
_RUNTIME_LOGGED_SIGNATURES = set()


def _log_runtime_args_once(args):
    model = str(getattr(args, "model", "")) or "missing"
    api_key_masked = _mask_secret(getattr(args, "api_key", ""), keep=4)
    max_retry_num = getattr(args, "max_retry_num", "missing")
    temperature = getattr(args, "temperature", "missing")
    in_price = float(getattr(args, "llm_input_price_per_mtoken", 0.0) or 0.0)
    out_price = float(getattr(args, "llm_output_price_per_mtoken", 0.0) or 0.0)
    currency = str(getattr(args, "llm_cost_currency", "USD") or "USD")

    signature = (str(model), str(api_key_masked), str(max_retry_num), str(temperature), float(in_price), float(out_price), str(currency))

    with _RUNTIME_LOG_LOCK:
        if signature in _RUNTIME_LOGGED_SIGNATURES:
            return
        _RUNTIME_LOGGED_SIGNATURES.add(signature)

    msg = (
        "[com] runtime args check: "
        f"model={model}, api_key={api_key_masked}, "
        f"max_retry_num={max_retry_num}, temperature={temperature}, "
        f"input_price_per_mtoken={in_price}, output_price_per_mtoken={out_price}, currency={currency}"
    )
    if in_price <= 0.0 and out_price <= 0.0:
        msg += " (cost estimation disabled: set --llm_input_price_per_mtoken/--llm_output_price_per_mtoken)"
    print(msg)


def _ensure_runtime_args(args):
    """Backfill runtime attrs needed by API wrapper when caller passes raw Namespace."""
    if not hasattr(args, "max_retry_num"):
        setattr(args, "max_retry_num", 3)
    if not hasattr(args, "temperature"):
        setattr(args, "temperature", 0.2)
    _log_runtime_args_once(args)
    return args


def _skill_system_prompt(role, output_contract):
    return (
        f"{role}\n"
        "Read the user's history and current candidate set, then execute the user's own preference skill. "
        "Choose the item the user would like most from the candidate set, and when requested, keep the items "
        "the user would genuinely hesitate about as the hesitation set. Do not introduce items outside the current candidate set.\n\n"
        f"Output strictly:\n{output_contract}"
    )


def _advisor_system_prompt(role, output_contract):
    return (
        f"{role}\n"
        "You are the user's friend advisor. Discuss only the visible HesitationSet. "
        "Do not choose the final item for the user. Follow the output format exactly.\n\n"
        f"Output strictly:\n{output_contract}"
    )


class ComUserAgent:
    def __init__(self, args):
        self.args = _ensure_runtime_args(args)
        self.advisor_policy = str(getattr(args, "com_advisor_policy", "single") or "single").strip().lower()

    def propose(
        self,
        history_str,
        candidate_names,
        prior_hint,
        target_profile="",
        shared_memory="none",
        slim_user_policy=None,
        updated_memory="none",
    ):
        system_prompt = _skill_system_prompt(
            "You are executing the target user's User Reasoning Skill.",
            "Reason: <short reason grounded in UserReasoningSkillSlim and evidence>\n"
            "Item: <exact item name copied from CandidateSet>",
        )
        user_prompt = (
            f"UserReasoningSkillSlim: {_render_skill_payload(slim_user_policy)}\n"
            f"History: {history_str}\n"
            f"CandidateSet({len(candidate_names)}): {_render_allowed_candidates(candidate_names)}\n"
            "CandidateSetRule: choose exactly one item from CandidateSet; copy the item name exactly; do not invent, translate, shorten, rename, or use items that only appear in History.\n"
            f"PriorHint: {prior_hint if prior_hint else 'none'}\n"
            f"UpdatedDecisionMemory: {updated_memory if updated_memory else shared_memory if shared_memory else 'none'}"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return parse_user_proposal(resp)

    def propose_decision(
        self,
        history_str,
        candidate_names,
        prior_hint,
        target_profile="",
        updated_memory="none",
        slim_user_policy=None,
    ):
        domain_text = _item_selection_domain_text(self.args)
        system_prompt = _skill_system_prompt(
            "You are executing the target user's User Reasoning Skill.",
            "Reason: <short reason grounded in UserReasoningSkillSlim and evidence>\n"
            + "Item: <exact item name copied from CandidateSet>\n"
            "HesitationSet: <3-5 exact item names copied from CandidateSet, including Item>\n"
            "HesitationEvidence:\n"
            "<candidate name> | why this candidate belongs in the user's hesitation set\n"
            "HesitationReason: <why these candidate-set items are the user's real hesitation choices>\n"
            "Confidence: <0-100 integer confidence in this item choice>",
        )
        user_prompt = (
            f"UserReasoningSkillSlim: {_render_skill_payload(slim_user_policy)}\n"
            f"History: {history_str}\n"
            f"CandidateSet({len(candidate_names)}): {_render_allowed_candidates(candidate_names)}\n"
            "Task: choose the candidate-set item this user would like most as Item, then keep the candidate-set items the user would genuinely hesitate about as HesitationSet.\n"
            "CandidateSetRule: Item, every HesitationSet entry, and every HesitationEvidence candidate must be copied exactly from CandidateSet. Do not use History-only items, similar outside items, translated names, shortened names, or background-knowledge items.\n"
            f"{domain_text['coverage']}\n"
            "HesitationSetRule: this is not a top-k ranking. Include Item plus candidates the user may plausibly choose after communication because they match different preferences, use cases, brands, or uncertainty points.\n"
            f"{domain_text['shortlist']}\n"
            "EvidenceRule: HesitationEvidence must include one line for every HesitationSet item. Each line only explains why that item entered the hesitation set. Do not label any entry as selected, do not rank them, and do not add repetitive generic doubt such as 'needs verification'.\n"
            f"PriorHint: {prior_hint if prior_hint else 'none'}\n"
            f"UpdatedDecisionMemory: {updated_memory if updated_memory else 'none'}"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return parse_user_proposal_decision(resp)

    def summarize_profile(self, history_str, candidate_names, prior_hint):
        return ""

    def estimate_self_confidence(
        self,
        history_str,
        candidate_names,
        proposal_name,
        proposal_reason,
        shortlist,
        uncertainty_points,
        prior_hint="",
        target_profile="",
        slim_user_policy=None,
        updated_memory="none",
    ):
        system_prompt = _skill_system_prompt(
            "You are executing the confidence-calibration part of the target user's User Reasoning Skill.",
            "Confidence: <0-100 integer>",
        )
        user_prompt = (
            f"History: {history_str}\n"
            f"CandidateSet({len(candidate_names)}): {_render_allowed_candidates(candidate_names)}\n"
            f"PriorHint: {prior_hint if prior_hint else 'none'}\n"
            f"CurrentProposal: {proposal_name}\n"
            f"ProposalReason: {proposal_reason if proposal_reason else 'none'}\n"
            f"HesitationSet: {_render_allowed_candidates(shortlist or [])}\n"
            f"UncertaintyPoints: {', '.join(uncertainty_points or []) if uncertainty_points else 'none'}\n"
            f"UpdatedDecisionMemory: {updated_memory if updated_memory else 'none'}\n"
            f"UserReasoningSkillSlim: {_render_skill_payload(slim_user_policy)}"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        conf = parse_self_confidence(resp)
        return conf if conf is not None else 60

    def choose_communication_path(self, decision_state, slim_user_policy, public_tree_options):
        public_tree_options = dict(public_tree_options or {})
        active_why = _option_ids(public_tree_options, 'why')
        active_who = _option_ids(public_tree_options, "who")
        active_how = _option_ids(public_tree_options, "how")
        planning_state = _communication_planning_state(decision_state, phase="initial")
        communication_skill = _communication_skill_payload(slim_user_policy)
        system_prompt = _skill_system_prompt(
            "You are planning how the target user should navigate the public Communication Tree.",
            'Why: <one available why node>\n'
            "Who: <one available who node>\n"
            "How: <one available how communication mode>\n"
            "Confidence: <0.0-1.0>\n"
            "Reasons: <short reasons grounded in user skill and current state>",
        )
        user_prompt = (
            f"CommunicationPlanningState: {_render_skill_payload(planning_state)}\n"
            f"AvailableWhy: {', '.join(active_why)}\n"
            f"AvailableHow: {', '.join(active_how)}\n"
            f"AvailableWho: {', '.join(active_who)}\n"
            f"CommunicationTreeNodeGuide:\n{_render_tree_options(public_tree_options)}\n"
            f"UserCommunicationSkill: {_render_skill_payload(communication_skill)}\n\n"
            "Tips:\n"
            "1. Do not reason about concrete items here; this step only chooses whether and how to communicate.\n"
            "2. First decide whether a why node matches the uncertainty shape. If none matches, skip communication.\n"
            "3. If communication is needed, choose how before who: how is the conversation skill; who is the advisor source.\n"
            "4. Prefer how/who choices supported by the user's communication skill, path memory, and advisor reliability memory."
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return parse_communication_path_choice(resp)

    def decide_communication_need(self, decision_state, slim_user_policy, updated_memory="none", phase="initial"):
        planning_state = _communication_planning_state(decision_state, phase=phase, updated_memory=updated_memory)
        communication_skill = _communication_skill_payload(slim_user_policy)
        system_prompt = _skill_system_prompt(
            "You are executing the target user's Communication Reasoning Skill.",
            "CommunicationAction: skip or start or continue or stop\n"
            "DecisionConfidence: <0-100 integer>\n"
            "Reasons: <comma-separated list or none>",
        )
        user_prompt = (
            f"CommunicationPlanningState: {_render_skill_payload(planning_state)}\n"
            f"UserCommunicationSkill: {_render_skill_payload(communication_skill)}\n\n"
            "Choose the action from the communication skill and uncertainty shape only. Do not use concrete item information."
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return parse_communication_control(resp)

    def choose_communication_next_step(
        self,
        decision_state,
        slim_user_policy,
        public_tree_options,
        updated_memory="none",
        phase="initial",
    ):
        public_tree_options = dict(public_tree_options or {})
        active_why = _option_ids(public_tree_options, 'why')
        active_who = _option_ids(public_tree_options, "who")
        active_how = _option_ids(public_tree_options, "how")
        planning_state = _communication_planning_state(decision_state, phase=phase, updated_memory=updated_memory)
        communication_skill = _communication_skill_payload(slim_user_policy)
        system_prompt = _skill_system_prompt(
            "You are planning the target user's next step in the public Communication Tree.",
            'MatchedWhyNodes:\n'
            '- <why node>: <why it matches current uncertainty, or none>\n'
            'SelectedWhy: <one matched why node, or none>\n'
            'NoCommunicationReason: <short reason if SelectedWhy is none, otherwise none>\n'
            "CandidateCommunicationPaths:\n"
            "1. why=<decision deficiency>; how=<communication mode>; who=<advisor source>; benefit=<short>; risk=<short>; memory=<short>\n"
            "SelectedPath:\n"
            'Why: <one available why node, or none>\n'
            "Who: <one available who node, or none>\n"
            "How: <one available how communication mode, or none>\n"
            'CommunicationAction: <derive from SelectedWhy: skip if none, otherwise start/continue>\n'
            "DecisionConfidence: <0-100 integer>\n"
            "Confidence: <0.0-1.0>\n"
            "Reasons: <short reasons grounded in user skill and current state>",
        )
        user_prompt = (
            f"CommunicationPlanningState: {_render_skill_payload(planning_state)}\n"
            f"AvailableWhy: {', '.join(active_why)}\n"
            f"AvailableHow: {', '.join(active_how)}\n"
            f"AvailableWho: {', '.join(active_who)}\n\n"
            f"CommunicationTreeNodeGuide:\n{_render_tree_options(public_tree_options)}\n\n"
            f"UserCommunicationSkill: {_render_skill_payload(communication_skill)}\n\n"
            "Tips:\n"
            "1. This is not an item-selection prompt. Do not mention or infer any concrete item.\n"
            '2. Use why only as a trigger. If no why node matches the uncertainty shape, set SelectedWhy to none and skip.\n'
            "3. If a why node matches, choose how first because how defines the advisor communication skill.\n"
            "4. Then choose who as the advisor source, using user preference, path memory, advisor reliability, and availability.\n"
            "5. CandidateCommunicationPaths should compare communication routes, not recommendation items."
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return {
            "control": parse_communication_control(resp),
            "path_choice": parse_communication_path_choice(resp),
            "raw": resp,
        }

    def revise(
        self,
        history_str,
        candidate_names,
        current_proposal,
        friend_alt,
        friend_neg,
        friend_pos,
        prior_hint="",
        target_profile="",
        shared_memory="none",
        slim_user_policy=None,
        updated_memory="none",
    ):
        system_prompt = _skill_system_prompt(
            "You are executing the target user's User Reasoning Skill after communication feedback.",
            "Reason: <short reason grounded in UserReasoningSkillSlim and UpdatedDecisionMemory>\n"
            "Item: <exact item name copied from HesitationSet>",
        )
        user_prompt = (
            f"UserReasoningSkillSlim: {_render_skill_payload(slim_user_policy)}\n"
            f"History: {history_str}\n"
            f"HesitationSet: {_render_allowed_candidates(candidate_names)}\n"
            "HesitationSetRule: choose exactly one item from HesitationSet; copy the item name exactly; do not invent, translate, shorten, rename, or use items outside this set.\n"
            f"PriorHint: {prior_hint if prior_hint else 'none'}\n"
            f"CurrentProposal: {current_proposal}\n"
            f"FriendSuggestedItem: {friend_alt if friend_alt else 'none'}\n"
            f"FriendFeedbackOnProposal: {friend_neg if friend_neg else 'none'}\n"
            f"FriendFeedbackOnAlternative: {friend_pos if friend_pos else 'none'}\n"
            f"{_advisor_listening_rule()}\n"
            f"UpdatedDecisionMemory: {updated_memory if updated_memory else shared_memory if shared_memory else 'none'}"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return parse_user_proposal(resp)

    def debate_friend(
        self,
        history_str,
        candidate_names,
        current_proposal,
        friend_speaker,
        friend_speech,
        prior_hint="",
        target_profile="",
        shared_memory="none",
        slim_user_policy=None,
        updated_memory="none",
    ):
        # Reuse the user arbitration prompt to keep a single user agent.
        revised_reason, _ = self.revise(
            history_str=history_str,
            candidate_names=candidate_names,
            current_proposal=current_proposal,
            friend_alt=current_proposal,
            friend_neg=friend_speech,
            friend_pos="",
            prior_hint=prior_hint,
            target_profile=target_profile,
            shared_memory=shared_memory,
            slim_user_policy=slim_user_policy,
            updated_memory=updated_memory,
        )
        return revised_reason if revised_reason else ""

    def arbitrate_state(
        self,
        history_str,
        candidate_names,
        current_proposal,
        current_reason,
        structured_memory,
        prior_hint="",
        target_profile="",
        shared_memory="none",
        slim_user_policy=None,
        updated_memory="none",
    ):
        system_prompt = _skill_system_prompt(
            "You are executing the target user's User Reasoning Skill to arbitrate whether the current item decision is final.",
            "CurrentDecision: keep or switch\n"
            "DecisionConfidence: <0-100 integer>\n"
            "DecisionState: final or continue\n"
            "UncertaintyPoints: <comma-separated list or none>\n"
            "FeedbackToAdvisors: <specific missing comparison/evidence requests if continue, otherwise none>\n"
            "NextRoundHesitationSet: <candidate names or evidence gaps for next round, otherwise none>\n"
            "SilentOrMissingEvidence: <silent candidates or missing evidence, otherwise none>",
        )
        user_prompt = (
            f"History: {history_str}\n"
            f"HesitationSet: {_render_allowed_candidates(candidate_names)}\n"
            f"PriorHint: {prior_hint if prior_hint else 'none'}\n"
            f"CurrentProposal: {current_proposal}\n"
            f"CurrentReason: {current_reason if current_reason else 'none'}\n"
            f"StructuredMemory: {structured_memory if structured_memory else 'none'}\n"
            f"{_advisor_listening_rule()}\n"
            f"UpdatedDecisionMemory: {updated_memory if updated_memory else 'none'}\n"
            f"UserReasoningSkillSlim: {_render_skill_payload(slim_user_policy)}\n"
            f"SharedMemoryWindow:\n{shared_memory if shared_memory else 'none'}"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return parse_stage6_decision(resp)

    def decide_after_feedback(
        self,
        history_str,
        candidate_names,
        current_proposal,
        current_reason,
        support_block,
        oppose_block,
        structured_memory,
        prior_hint="",
        target_profile="",
        updated_memory="none",
        slim_user_policy=None,
    ):
        system_prompt = _skill_system_prompt(
            "You are executing the target user's integrated post-communication User Reasoning Skill.",
            "Reason: <short reason that first responds to the advisor claims, then applies UserReasoningSkillSlim>\n"
            "Item: <exact item name copied from HesitationSet>\n"
            "CurrentDecision: keep or switch\n"
            "DecisionConfidence: <0-100 integer>\n"
            "DecisionState: final or continue\n"
            "UserClarificationAnswers: <answer advisor AskUser/direct-user-input questions from UserReasoningSkillSlim and history; use none only when truly not inferable>\n"
            "NextRoundHesitationSet: <if DecisionState is continue, write the exact remaining HesitationSet candidates after removing weak items; otherwise none>\n"
            "RemovedFromHesitationSet: <exact candidates removed or down-ranked after advisor discussion, otherwise none>\n"
            "FeedbackToAdvisors: <if DecisionState is continue, include UserClarificationAnswers first, then write what advisors should compare or verify next; otherwise none>\n\n"
            "Output only these fields, one field per line. Do not output JSON, markdown, or extra prose.",
        )
        user_prompt = (
            f"History: {history_str}\n"
            f"HesitationSet: {_render_allowed_candidates(candidate_names)}\n"
            "HesitationSetRule: choose exactly one item from HesitationSet; copy the item name exactly; do not invent, translate, shorten, rename, or use items outside this set.\n"
            f"PriorHint: {prior_hint if prior_hint else 'none'}\n"
            f"InitialProposalBeforeAdvisor: {current_proposal}\n"
            f"InitialReasonBeforeAdvisor: {current_reason if current_reason else 'none'}\n"
            f"FinalAdvisorEvidencePacket: {support_block or structured_memory or 'none'}\n"
            f"{_advisor_listening_rule()}\n"
            f"UpdatedDecisionMemory: {updated_memory if updated_memory else 'none'}\n"
            f"UserReasoningSkillSlim: {_render_skill_payload(slim_user_policy)}"
            "\n\nTips:\n"
            "1. Choose Item only from HesitationSet; History is preference background.\n"
            "2. Treat FinalAdvisorEvidencePacket as information-only evidence, not a vote or winner selection.\n"
            "3. Keep Reason and Item consistent: do not select a candidate whose warning you accept unless you explicitly reject that warning.\n"
            "4. Compare the InitialProposalBeforeAdvisor and advisor feedback fairly. Do not favor either one merely because it was the initial choice or because an advisor mentioned it.\n"
            "5. Keep the initial item when it remains best supported; switch or continue only when the comparison provides clearer support for another candidate.\n"
            "6. If advisors ask the user to clarify preferences, answer from UserReasoningSkillSlim and History when supported; otherwise write none.\n"
            "7. Do not simply pass advisor AskUser questions back unchanged. If the answer is supported, put it in UserClarificationAnswers.\n"
            "8. Set DecisionState=final when advisor evidence plus UserReasoningSkillSlim makes one item clearly strongest; otherwise continue with a smaller NextRoundHesitationSet and focused FeedbackToAdvisors."
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        parsed = parse_user_final_decision(resp)
        if parsed:
            return parsed
        return {
            "_parse_failed": True,
            "raw_response": str(resp or ""),
        }


class ComAdvisorAgent:
    def __init__(self, args):
        self.args = _ensure_runtime_args(args)
        self.advisor_policy = str(getattr(args, "com_advisor_policy", "single") or "single").strip().lower()

    def review(
        self,
        history_str,
        candidate_names,
        proposal,
        proposal_reason="",
        friend_context="",
        friend_history_summary="",
        friend_top_rated="",
        candidate_matches="none",
        candidate_suggestions="none",
        group_memory="none",
        prior_hint="",
        target_profile="",
        shared_memory="none",
        friend_speaker="",
        revote_round="false",
        target_user_skill=None,
    ):
        system_prompt = _advisor_system_prompt(
            "You are an advisor executing the selected communication skill.",
            "Decision: <agree or disagree>\n"
            "Neg: <why the opposed item is weaker for the target user>\n"
            "AltItem: <exact item name copied from HesitationSet, or none if unresolved>\n"
            "Pos: <why AltItem is stronger for the target user>",
        )
        user_prompt = (
            f"RequesterShareableItemBrief: {_render_skill_payload(target_user_skill)}\n"
            f"AdvisorContext: {friend_context if friend_context else 'none'}\n"
            f"FriendHistorySummary: {friend_history_summary if friend_history_summary else 'none'}\n"
            f"FriendTopRated: {friend_top_rated if friend_top_rated else 'none'}\n"
            f"TargetHistory: {history_str}\n"
            f"HesitationSet({len(candidate_names)}): {_render_allowed_candidates(candidate_names)}\n"
            "HesitationSetRule: AltItem must be copied exactly from HesitationSet, or use none if unresolved. Do not invent, translate, shorten, rename, or use items outside this set.\n"
            f"CandidateMatches: {candidate_matches if candidate_matches else 'none'}\n"
            f"CandidateSuggestions: {candidate_suggestions if candidate_suggestions else 'none'}\n"
            f"GroupMemory: {group_memory if group_memory else 'none'}\n"
            f"PriorHint: {prior_hint if prior_hint else 'none'}\n"
            f"UpdatedDecisionMemory: {shared_memory if shared_memory else 'none'}\n"
            f"FriendSpeaker: {str(friend_speaker or 'none')}\n"
            f"RevoteRound: {str(revote_round)}"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return parse_advisor_review(resp)

    def summarize_discussion(
        self,
        history_str,
        candidate_names,
        current_proposal,
        friend_turns,
        aggregate_item,
        aggregate_reason,
        prior_hint="",
        target_profile="",
        shared_memory="none",
    ):
        system_prompt = _advisor_system_prompt(
            "You are an advisor summarizing the selected communication skill result.",
            "HesitationSetCons: <concise risks or limits for relevant HesitationSet candidates>\n"
            "NewItemPros: <concise pros of recommended item>\n"
            "VoteSummary: <structured vote/evidence summary>\n"
            "RecommendedItem: <exact item name copied from HesitationSet>\n"
            "OpenIssue: <remaining concern or none>",
        )
        user_prompt = (
            f"HesitationSet({len(candidate_names)}): {_render_allowed_candidates(candidate_names)}\n"
            "HesitationSetRule: RecommendedItem must be copied exactly from HesitationSet; do not invent, translate, shorten, rename, or use items outside this set.\n"
            f"PriorHint: {prior_hint if prior_hint else 'none'}\n"
            f"FriendTurns:\n{friend_turns if friend_turns else 'none'}\n"
            f"AggregateItem: {aggregate_item}\n"
            f"AggregateReason: {aggregate_reason if aggregate_reason else 'none'}\n"
            f"UpdatedDecisionMemory: {shared_memory if shared_memory else 'none'}"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        reason, item, issue = parse_friend_summary(resp)
        if (not reason) or (not item):
            return aggregate_reason, aggregate_item, "none"
        return reason, item, issue if issue else "none"

    def run_skill_compare_representative(
        self,
        skill_instruction,
        target_history,
        target_profile,
        candidate_names,
        proposal_item,
        defend_item,
        oppose_item,
        representative_history,
        representative_profile,
        representative_label,
        shared_memory="none",
        target_user_skill=None,
    ):
        system_prompt = _advisor_system_prompt(
            "You are an advisor executing the selected communication skill.",
            "Decision: <agree or disagree>\n"
            "Neg: <why the opposing item is weaker for the target user; include the concrete challenge or answered doubt when the selected how node requires discussion>\n"
            "AltItem: <exact RepresentedItem copied from HesitationSet>\n"
            "Pos: <why the represented item is stronger for the target user; include claim, answer/rebuttal, remaining doubt, and whether the discussion is resolved/partially_resolved/unresolved when applicable>",
        )
        user_prompt = (
            f"SelectedCommunicationSkillPayload:\n{str(skill_instruction or '').strip()}\n\n"
            f"RepresentativeLabel: {representative_label}\n"
            f"RepresentedItem: {defend_item}\n"
            f"OpposingItem: {oppose_item}\n"
            f"HesitationSet: {_render_allowed_candidates(candidate_names)}\n"
            "HesitationSetRule: AltItem must equal RepresentedItem exactly, and RepresentedItem must be in HesitationSet. Do not invent, translate, shorten, rename, or use items outside this set.\n"
            "DiscussionProtocolConstraint: follow the selected how node in SelectedCommunicationSkillPayload. "
            "Do not add a generic warning/promotion/debate behavior that is not requested by that how node. "
            "If the how node cannot be resolved with your evidence, say unresolved or partially_resolved instead of inventing consensus.\n"
            f"TargetHistory: {target_history}\n"
            f"RequesterShareableItemBrief: {_render_skill_payload(target_user_skill)}\n"
            f"RepresentativeHistory: {representative_history if representative_history else 'none'}\n"
            f"RepresentativeProfile: {representative_profile if representative_profile else 'none'}\n"
            f"SharedMemory: {shared_memory if shared_memory else 'none'}"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return parse_advisor_review(resp)

    def run_skill_native_review(
        self,
        skill_instruction,
        history_str,
        candidate_names,
        proposal,
        proposal_reason="",
        friend_context="",
        friend_history_summary="",
        friend_top_rated="",
        candidate_matches="none",
        candidate_suggestions="none",
        group_memory="none",
        prior_hint="",
        target_profile="",
        shared_memory="none",
        friend_speaker="",
        revote_round="false",
        target_user_skill=None,
        advisor_own_skill="",
    ):
        advisor_guidance = {}
        visible_working_memory = shared_memory
        if isinstance(shared_memory, dict):
            advisor_guidance = dict(shared_memory.get("advisor_guidance", {}) or shared_memory.get("assigned_task", {}) or {})
            user_task = str(shared_memory.get("user_task", "") or advisor_guidance.get("user_task", "") or "")
            task_type = str(shared_memory.get("task_type", "") or advisor_guidance.get("task_type", "") or "")
            secondary_what = shared_memory.get("secondary_what", advisor_guidance.get("secondary_what", []))
            criteria = shared_memory.get("criteria", advisor_guidance.get("criteria", []))
            task_source = str(shared_memory.get("task_source", "") or "")
            initial_choice_context = shared_memory.get("user_initial_choice_context", "")
            previous_user_feedback = shared_memory.get("previous_user_feedback", "")
            visible_working_memory = {
                key: value
                for key, value in shared_memory.items()
                if key not in {"advisor_guidance", "assigned_task", "user_initial_choice_context", "user_task", "task_type", "secondary_what", "criteria", "task_source"}
            }
        else:
            initial_choice_context = ""
            user_task = ""
            task_type = ""
            secondary_what = []
            criteria = []
            task_source = ""
            previous_user_feedback = ""
        system_prompt = _advisor_system_prompt(
            "You are an advisor executing the selected communication skill.",
            "Follow OutputFormat exactly. "
            "Use exact field labels and one field per line. "
            "Do not choose the final item for the user.",
        )
        original_feedback = str(previous_user_feedback or "").strip()
        if original_feedback.lower() in {"none", "null", "n/a", "na"}:
            original_feedback = ""
        previous_round_evidence = ""
        previous_friend_views = ""
        if isinstance(visible_working_memory, dict):
            previous_round_evidence = str(visible_working_memory.get("previous_round_summary", "") or "").strip()
            previous_friend_views = str(visible_working_memory.get("current_round_discussion_memory", "") or "").strip()
        else:
            previous_friend_views = str(group_memory or "").strip()
        if previous_round_evidence.lower() in {"none", "null", "n/a", "na"}:
            previous_round_evidence = ""
        if previous_friend_views.lower() in {"none", "null", "n/a", "na"}:
            previous_friend_views = ""
        memory_block = ""
        if previous_round_evidence:
            memory_block += f"PreviousRoundEvidence:\n{previous_round_evidence}\n\n"
        if previous_friend_views:
            memory_block += f"PreviousFriendViews:\n{previous_friend_views}\n\n"
        user_prompt = (
            f"Task:\n{user_task if user_task else 'none'}\n\n"
            + (f"OriginalFeedbackToAdvisors:\n{original_feedback}\n\n" if original_feedback else "")
            + f"HowAndOutput:\n{str(skill_instruction or '').strip()}\n\n"
            f"AdvisorSpeaker: {str(friend_speaker or 'none')}\n\n"
            f"HesitationSet: {_render_allowed_candidates(candidate_names)}\n\n"
            f"WhyInHesitation:\n{initial_choice_context if initial_choice_context else 'none'}\n\n"
            f"UserPreference:\n{_render_requester_brief(target_user_skill)}\n\n"
            f"YourPreferenceSkill:\n{str(advisor_own_skill or '').strip() or 'none'}\n\n"
            f"YourHistory:\n{friend_context if friend_context else 'none'}\n\n"
            + memory_block
            +
            "Rules:\n"
            "1. Candidate names in every set and CandidateView must be copied exactly from HesitationSet.\n"
            "2. Answer Task directly.\n"
            "3. CandidateView is required even when the answer is unresolved; write useful candidate-level observations instead of only saying none.\n"
            "4. Use YourPreferenceSkill and YourHistory only as your own experience evidence. If they are weak, say so inside TaskAnswer or AskUser instead of inventing facts.\n"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        return parse_advisor_review(resp)

    def run_skill_native_summary(
        self,
        skill_instruction,
        current_proposal,
        candidate_names,
        friend_turns,
        aggregate_item,
        aggregate_reason,
        history_str="",
        prior_hint="",
        target_profile="",
        shared_memory="none",
        slim_user_policy=None,
        updated_memory="none",
    ):
        system_prompt = _advisor_system_prompt(
            "You are an advisor summarizing the selected communication skill result.",
            "HesitationSetCons: <concise risks or limits for relevant HesitationSet candidates>\n"
            "NewItemPros: <concise pros of recommended item>\n"
            "VoteSummary: <structured vote/evidence summary>\n"
            "RecommendedItem: <exact item name copied from HesitationSet>\n"
            "OpenIssue: <remaining concern or none>",
        )
        user_prompt = (
            f"SelectedCommunicationSkillPayload:\n{str(skill_instruction or '').strip()}\n\n"
            f"HesitationSet: {_render_allowed_candidates(candidate_names)}\n"
            "HesitationSetRule: RecommendedItem must be copied exactly from HesitationSet; do not invent, translate, shorten, rename, or use items outside this set.\n"
            f"TargetHistory: {history_str if history_str else 'none'}\n"
            f"PriorHint: {prior_hint if prior_hint else 'none'}\n"
            f"FriendTurns:\n{friend_turns if friend_turns else 'none'}\n"
            f"AggregateItem: {aggregate_item}\n"
            f"AggregateReason: {aggregate_reason if aggregate_reason else 'none'}\n"
            f"SharedMemory: {shared_memory if shared_memory else 'none'}"
        )
        resp = llm_request(system_prompt, user_prompt, self.args)
        reason, item, issue = parse_friend_summary(resp)
        if (not reason) or (not item):
            return aggregate_reason, aggregate_item, "none"
        return reason, item, issue if issue else "none"


def build_com_args(args, tool_model, item_size, maxlen):
    ns = SimpleNamespace(**args.__dict__)
    ns.max_retry_num = int(getattr(ns, "max_retry_num", 3))
    ns.temperature = float(getattr(ns, "temperature", 0.2))
    ns.external_tool = tool_model
    ns.item_size = int(item_size)
    ns.maxlen = int(maxlen)
    return ns
