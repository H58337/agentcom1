def _clamp_confidence(value, default=0):
    try:
        return max(0, min(100, int(value)))
    except Exception:
        return int(default)


def _as_list(values):
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, set):
        return list(values)
    return [values]


def build_decision_state(
    proposal_item,
    proposal_reason,
    shortlist,
    uncertainty_points,
    self_confidence,
    primary_trigger,
    communication_need,
    prior_item="",
    slim_user_policy=None,
):
    return {
        "proposal_item": str(proposal_item or ""),
        "proposal_reason": str(proposal_reason or ""),
        "shortlist": [str(x) for x in _as_list(shortlist)],
        "uncertainty_points": [str(x) for x in _as_list(uncertainty_points)],
        "self_confidence": _clamp_confidence(self_confidence, default=0),
        "primary_trigger": str(primary_trigger or ""),
        "communication_need": bool(communication_need),
        "prior_item": str(prior_item or ""),
        "slim_user_policy": slim_user_policy if isinstance(slim_user_policy, str) else dict(slim_user_policy or {}),
    }


def build_communication_path(
    why,
    who,
    how,
    what=None,
    user_task="",
    task_type_hint="",
    task_targets=None,
    criteria=None,
    secondary_what=None,
    mapping_confidence="",
    unmapped_parts=None,
    expected_output="",
    task_source="",
    unmapped_task=False,
    previous_what="",
    previous_how="",
    followup_of_round=0,
    advisor_group_source="",
    path_reason=None,
    path_score=0.0,
    risk_marks=None,
    trial_flag=False,
    pattern_source="direct",
    primary_why="",
    matched_why=None,
    why_reasons=None,
):
    primary_why = str(primary_why or why or "")
    matched_why_rows = [str(x) for x in _as_list(matched_why) if str(x).strip()]
    if primary_why and primary_why not in matched_why_rows and primary_why not in ["skip", "none"]:
        matched_why_rows.insert(0, primary_why)
    return {
        'why': primary_why,
        "primary_why": primary_why,
        "matched_why": matched_why_rows,
        "why_reasons": [str(x) for x in _as_list(why_reasons) if str(x).strip()],
        "what": str(what or ""),
        "who": str(who or ""),
        "how": str(how or ""),
        "legacy_what": str(what or ""),
        "user_task": str(user_task or ""),
        "task_type_hint": str(task_type_hint or ""),
        "task_targets": [str(x) for x in _as_list(task_targets) if str(x).strip()],
        "criteria": [str(x) for x in _as_list(criteria) if str(x).strip()],
        "secondary_what": [str(x) for x in _as_list(secondary_what) if str(x).strip()],
        "mapping_confidence": str(mapping_confidence or ""),
        "unmapped_parts": [str(x) for x in _as_list(unmapped_parts) if str(x).strip()],
        "expected_output": str(expected_output or ""),
        "task_source": str(task_source or ""),
        "unmapped_task": bool(unmapped_task),
        "previous_what": str(previous_what or ""),
        "previous_how": str(previous_how or ""),
        "followup_of_round": int(followup_of_round or 0),
        "advisor_group_source": str(advisor_group_source or ""),
        "path_reason": [str(x) for x in _as_list(path_reason)],
        "path_score": float(path_score or 0.0),
        "risk_marks": [str(x) for x in _as_list(risk_marks)],
        "trial_flag": bool(trial_flag),
        "pattern_source": str(pattern_source or "direct"),
    }


def build_advisor_feedback(
    advisor_id,
    advisor_type,
    stance,
    endorsed_item,
    support_reason="",
    oppose_reason="",
    confidence=0,
    solved_uncertainty=None,
    raw_text="",
):
    return {
        "advisor_id": str(advisor_id or ""),
        "advisor_type": str(advisor_type or ""),
        "stance": str(stance or ""),
        "endorsed_item": str(endorsed_item or ""),
        "support_reason": str(support_reason or ""),
        "oppose_reason": str(oppose_reason or ""),
        "confidence": _clamp_confidence(confidence, default=0),
        "solved_uncertainty": [str(x) for x in _as_list(solved_uncertainty)],
        "raw_text": str(raw_text or ""),
    }


def build_absorbed_memory(
    helpful_points=None,
    rejected_points=None,
    candidate_comparison=None,
    alternative_items=None,
    advisor_reliability_observation=None,
    remaining_uncertainty=None,
    evidence_packet=None,
):
    return {
        "helpful_points": [str(x) for x in _as_list(helpful_points)],
        "rejected_points": [str(x) for x in _as_list(rejected_points)],
        "candidate_comparison": dict(candidate_comparison or {}),
        "alternative_items": [str(x) for x in _as_list(alternative_items)],
        "advisor_reliability_observation": dict(advisor_reliability_observation or {}),
        "remaining_uncertainty": [str(x) for x in _as_list(remaining_uncertainty)],
        "evidence_packet": dict(evidence_packet or {}),
    }


def build_redecision_state(
    current_decision,
    decision_item,
    decision_confidence,
    decision_state,
    remaining_uncertainty=None,
    stop_reason="",
):
    return {
        "current_decision": str(current_decision or ""),
        "decision_item": str(decision_item or ""),
        "decision_confidence": _clamp_confidence(decision_confidence, default=0),
        "decision_state": str(decision_state or ""),
        "remaining_uncertainty": [str(x) for x in _as_list(remaining_uncertainty)],
        "stop_reason": str(stop_reason or ""),
    }


def build_evaluation_result(
    outcome_signal="",
    confidence_change=None,
    uncertainty_reduction=None,
    path_level_feedback=None,
    branch_level_feedback=None,
):
    return {
        "outcome_signal": str(outcome_signal or ""),
        "confidence_change": dict(confidence_change or {}),
        "uncertainty_reduction": dict(uncertainty_reduction or {}),
        "path_level_feedback": dict(path_level_feedback or {}),
        "branch_level_feedback": dict(branch_level_feedback or {}),
    }
