import re


ACTIVE_WHAT_NODES = [
    "reduce_hesitation_set",
    "find_interested_subset",
    "compare_remaining_candidates",
    "evidence_gap_check",
    "reasoning_check",
    "none",
]

TASK_OUTPUT_FORMATS = {
    "reduce_hesitation_set": [
        "RemoveSet: <exact HesitationSet candidates you would remove or down-rank, or none>",
        "CandidateView:\n- <exact removed candidate> | <remove> | <specific evidence or criterion used to exclude this candidate>",
        "RiskReason: <the exclusion criterion and evidence for every RemoveSet candidate, or none>",
        "TaskAnswer: <direct answer to Task>",
    ],
    "find_interested_subset": [
        "InterestedSet: <exact HesitationSet candidates the user may stay interested in, or none>",
        "CandidateView:\n- <exact interested candidate> | <interest> | <specific evidence or criterion showing why the user may stay interested>",
        "InterestReason: <the interest criterion and evidence for every InterestedSet candidate, or none>",
        "TaskAnswer: <direct answer to Task>",
    ],
    "compare_remaining_candidates": [
        "ComparedSet: <exact HesitationSet candidates you actually compared>",
        "StrongerCandidates: <exact candidates with stronger fit, or none>",
        "WeakerCandidates: <exact candidates with weaker fit/risk, or none>",
        "CandidateView:\n- <exact candidate> | <stronger/weaker/mixed/unclear> | <short reason>",
        "KeyTradeoff: <the key comparison criterion, why it matters for this user, and which side it favors>",
        "TaskAnswer: <direct answer to Task>",
    ],
    "evidence_gap_check": [
        "EvidenceGapSet: <exact HesitationSet candidates whose current/original reason is insufficient, or none>",
        "CandidateView:\n- <exact candidate needing more evidence> | <evidence_gap> | <what current/original reason is insufficient and what supplementary reason or evidence is needed>",
        "SupplementReason: <the missing or supplementary reason/evidence needed for every EvidenceGapSet candidate, or none>",
        "TaskAnswer: <direct answer to Task>",
    ],
    "reasoning_check": [
        "ReliableReasons: <initial hesitation reasons or assumptions that are well supported, or none>",
        "WeakReasons: <initial hesitation reasons or assumptions that are weak, unsupported, or need correction, or none>",
        "CandidateView:\n- <exact candidate> | <reason_reliable/reason_weak/mixed/unclear> | <which initial reason was checked and why>",
        "Correction: <how the user should revise the initial reasoning; do not select a final item>",
        "TaskAnswer: <direct answer to Task>",
        "AskUser: <specific question for the user, or none>",
    ],
    "none": [
        "RelevantCandidates: <exact HesitationSet candidates your answer covers, or none>",
        "UnclearSet: <exact HesitationSet candidates you cannot judge yet, or none>",
        "CandidateView:\n- <exact candidate> | <support/risk/unclear> | <short reason>",
        "TaskAnswer: <direct answer to Task as far as possible>",
        "AskUser: <specific question for the user, or none>",
    ],
}


HOW_OUTPUT_FIELDS = {
    "single-advisor": [],
    "multi-cooperative": [
        "ResponseToPrevious: <agreement, refinement, correction, or integration of PreviousFriendViews, or none>",
    ],
    "multi-competitive": [
        "ChallengeOrSupportPrevious: <challenge, support, or correction of a previous claim, or none>",
    ],
}


def _clean(value):
    return " ".join(str(value or "").strip().split())


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _is_empty_task(value):
    text = _clean(value).strip().lower()
    return text in {"", "0", "none", "null", "n/a", "na"}


def read_label(text, label):
    if not text:
        return ""
    pattern = rf"(?:^|\n)\s*(?:[-*]\s*)?{re.escape(label)}\s*:\s*(.*?)(?=\n\s*(?:[-*]\s*)?[A-Za-z][A-Za-z0-9_ ]{{1,40}}\s*:|\Z)"
    match = re.search(pattern, str(text), flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _split_csvish(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in re.split(r"[,;|\n]+", str(value or "")) if x.strip()]


def _mentioned_targets(user_task, hesitation_set):
    task = str(user_task or "").lower()
    rows = []
    for item in list(hesitation_set or []):
        item = str(item or "").strip()
        if item and item.lower() in task:
            rows.append(item)
    return rows


def map_user_task_to_what(user_task="", hesitation_set=None, task_targets=None, criteria=None):
    text = str(user_task or "").lower()
    compact = re.sub(r"\s+", " ", text)
    rule_order = [
        (
            "compare_remaining_candidates",
            [
                "compare",
                "comparison",
                "direct comparison",
                "versus",
                " vs ",
                "stronger",
                "weaker",
                "trade-off",
                "rank",
                "better",
                "challenge",
                "debate",
                "which one",
                "against",
            ],
        ),
        (
            "reasoning_check",
            [
                "is my reason",
                "is my reasoning",
                "does my reasoning",
                "is this assumption",
                "is that true",
                "really fit",
                "surface match",
                "actually fit",
                "verify",
                "check my",
                "initial reason",
            ],
        ),
        (
            "evidence_gap_check",
            [
                "missing",
                "evidence gap",
                "silent",
                "coverage",
                "unanswered",
                "explain",
                "check evidence",
                "answer",
                "not covered",
                "no one discussed",
                "specific information",
                "specific examples",
                "actual listening assessments",
                "details",
                "more specific",
            ],
        ),
        (
            "reduce_hesitation_set",
            [
                "reduce",
                "shrink",
                "eliminate",
                "exclude",
                "remove",
                "down-rank",
                "weak fit",
                "not fit",
                "risk",
                "caution",
                "warning",
                "rule out",
                "clearly do not fit",
            ],
        ),
        (
            "find_interested_subset",
            [
                "retain",
                "keep",
                "interested",
                "promising",
                "worth considering",
                "positive",
                "support",
                "recommend",
                "like",
                "still worth",
            ],
        ),
    ]
    hits = []
    reasons = []
    for what, keywords in rule_order:
        if any(key in compact for key in keywords):
            hits.append(what)
            reasons.append(f"matched {what} from task text")
    primary = hits[0] if hits else "none"
    secondary = [x for x in hits[1:] if x != primary]
    targets = _split_csvish(task_targets) or _mentioned_targets(user_task, hesitation_set or [])
    criteria_rows = _split_csvish(criteria)
    unmapped = primary == "none"
    confidence = "high" if primary != "none" and len(hits) == 1 else ("medium" if primary != "none" else "low")
    return {
        "what": primary,
        "primary_what": primary,
        "secondary_what": secondary,
        "task_targets": targets,
        "criteria": criteria_rows,
        "mapping_confidence": confidence,
        "unmapped_task": bool(unmapped),
        "unmapped_parts": [] if not unmapped else [str(user_task or "")[:240]],
        "mapping_reasons": reasons or ["no canonical what matched"],
    }


def output_format_for_what(what, how=""):
    what = str(what or "").strip()
    how = str(how or "").strip()
    base_what = what.split("/", 1)[0] if "/" in what else what
    base_how = how.split("/", 1)[0] if "/" in how else how
    return list(TASK_OUTPUT_FORMATS.get(base_what) or TASK_OUTPUT_FORMATS["none"]) + list(HOW_OUTPUT_FIELDS.get(base_how, []))


def _candidate_reason_map(hesitation_evidence=None):
    rows = {}
    for row in list(hesitation_evidence or []):
        if isinstance(row, dict):
            candidate = str(row.get("candidate", "") or row.get("item", "") or "").strip()
            reason = str(row.get("reason", "") or row.get("fit", "") or row.get("evidence", "") or "").strip()
        else:
            parts = [p.strip() for p in re.split(r"\s*\|\s*", str(row or "")) if p.strip()]
            candidate = parts[0] if parts else ""
            reason = " | ".join(parts[1:]) if len(parts) >= 2 else ""
        if candidate:
            rows[candidate] = reason
    return rows


def _name_tokens(name):
    text = str(name or "").replace("_", " ").replace("-", " ").lower()
    tokens = [x for x in re.split(r"[^a-z0-9]+", text) if len(x) >= 3]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "plus",
        "pro",
        "new",
        "all",
        "one",
        "two",
        "black",
        "white",
        "edition",
    }
    return [x for x in tokens if x not in stop]


def _hesitation_shape(hesitation_set, hesitation_evidence=None, uncertainty_points=None, matched_why="", matched_whens=None):
    items = [str(x).strip() for x in list(hesitation_set or []) if str(x).strip()]
    evidence = _candidate_reason_map(hesitation_evidence)
    points = {str(x).strip().lower() for x in list(uncertainty_points or []) if str(x).strip()}
    why_rows = [str(x).strip().lower() for x in _as_list(matched_whens) if str(x).strip()]
    matched_why = str(matched_why or "").strip().lower()
    if matched_why and matched_why not in why_rows:
        why_rows.insert(0, matched_why)
    missing_items = [x for x in items if not str(evidence.get(x, "")).strip()]
    generic_markers = [
        "not provide a separate reason",
        "needs verification",
        "need verification",
        "尚未核验",
        "缺少",
        "unclear",
        "unknown",
        "missing",
    ]
    weak_evidence = [
        item
        for item, reason in evidence.items()
        if not reason or any(marker in reason.lower() for marker in generic_markers)
    ]
    token_sets = [set(_name_tokens(x)) for x in items]
    shared_tokens = set.intersection(*token_sets) if token_sets else set()
    all_tokens = set().union(*token_sets) if token_sets else set()
    same_family = bool(shared_tokens) or (len(all_tokens) <= max(4, len(items) * 2) and len(items) >= 2)
    mixed_family = len(items) >= 4 and not same_family
    return {
        "items": items,
        "points": points,
        "matched_why": matched_why,
        "matched_whens": why_rows,
        "missing_items": missing_items,
        "weak_evidence": weak_evidence,
        "same_family": same_family,
        "mixed_family": mixed_family,
    }


def _choose_first_round_what(hesitation_set, hesitation_evidence=None, uncertainty_points=None, matched_why="", matched_whens=None):
    shape = _hesitation_shape(hesitation_set, hesitation_evidence, uncertainty_points, matched_why, matched_whens)
    items = shape["items"]
    points = shape["points"]
    whens = set(shape.get("matched_whens", []) or [])
    if not items:
        return "none", ["empty hesitation set"], []
    has_prior_conflict = (
        "internal-prior-conflict" in whens
        or shape["matched_why"] == "internal-prior-conflict"
        or "internal_prior_conflict" in points
    )
    has_candidate_conflict = (
        "candidate-conflict" in whens
        or "candidate_comparison" in points
        or len(items) >= 2
    )
    has_novelty = "novelty-uncertainty" in whens or "novelty_justification" in points
    has_cold_start = "cold-start" in whens
    if has_prior_conflict and has_candidate_conflict:
        return "reasoning_check", ["multi-when prior conflict + candidate conflict asks for reasoning check"], ["compare_remaining_candidates"]
    if has_novelty and has_candidate_conflict:
        return "find_interested_subset", ["multi-when novelty + candidate conflict asks for interest subset"], ["reduce_hesitation_set"]
    if has_cold_start and (len(shape["missing_items"]) > 0 or len(shape["weak_evidence"]) > 0):
        return "evidence_gap_check", ["cold-start with weak/missing evidence asks for evidence gap check"], []
    if has_prior_conflict:
        return "reasoning_check", ["when/uncertainty indicates internal-prior conflict"], []
    if len(shape["missing_items"]) >= max(1, len(items) // 2) or len(shape["weak_evidence"]) >= max(2, len(items) // 2 + 1):
        return "evidence_gap_check", ["many hesitation candidates lack useful evidence"], []
    if "candidate_comparison" in points and (len(items) <= 3 or shape["same_family"]):
        return "compare_remaining_candidates", ["close candidate comparison is needed"], []
    if len(items) >= 4:
        return "reduce_hesitation_set", ["large hesitation set should be narrowed first"], []
    if "preference_alignment" in points:
        return "find_interested_subset", ["preference alignment uncertainty asks for a retained-interest subset"], []
    if "candidate_comparison" in points:
        return "compare_remaining_candidates", ["candidate comparison uncertainty"], []
    return "reduce_hesitation_set", ["default first-round task narrows the hesitation set"], []


def _rule_user_task_for_what(what, hesitation_set, hesitation_reason="", hesitation_evidence=None, secondary_what=None, matched_whens=None):
    items = [str(x).strip() for x in list(hesitation_set or []) if str(x).strip()]
    item_text = ", ".join(items) if items else "the current hesitation set"
    evidence = _candidate_reason_map(hesitation_evidence)
    missing = [item for item in items if not evidence.get(item)]
    secondary = {str(x).strip() for x in _as_list(secondary_what) if str(x).strip()}
    if what == "reasoning_check":
        if "compare_remaining_candidates" in secondary:
            return (
                f"Please help me check whether my initial reasons for keeping these candidates are actually valid: {item_text}. "
                "Tell me which reasons look reliable or weak, then compare the remaining candidates and say which ones still deserve consideration."
            )
        return (
            f"Please help me check whether my initial reasons for keeping these candidates are actually valid: {item_text}. "
            "Tell me which reasons look reliable, which look weak, and which candidates still deserve consideration."
        )
    if what == "evidence_gap_check":
        if missing:
            return (
                f"Please help me fill the evidence gaps for these hesitation candidates: {item_text}. "
                f"Pay special attention to candidates with unclear reasons such as {', '.join(missing[:4])}, and tell me what can or cannot be judged."
            )
        return (
            f"Please check whether the current evidence is enough for these hesitation candidates: {item_text}. "
            "Tell me what important evidence is still missing before I narrow the set."
        )
    if what == "compare_remaining_candidates":
        return (
            f"Please compare these hesitation candidates directly: {item_text}. "
            "Tell me which candidates fit me better or worse, and which assumptions in my initial thinking need to be corrected."
        )
    if what == "find_interested_subset":
        return (
            f"Please identify the smaller subset I am most likely to stay interested in from these candidates: {item_text}. "
            "Tell me which ones are still promising and which ones are only surface matches."
        )
    if what == "reduce_hesitation_set":
        return (
            f"Please help me narrow this hesitation set: {item_text}. "
            "Tell me which candidates can be removed or down-ranked, which should stay, and what evidence supports that shrink."
        )
    return (
        f"Please review this hesitation set for me: {item_text}. "
        "Tell me what you can judge, what remains unclear, and how I should continue comparing them."
    )


def _fallback_expected_from_task(user_task):
    text = str(user_task or "").lower()
    if any(x in text for x in ["compare", "versus", " vs ", "which one", "better", "stronger", "weaker"]):
        return "compare the named candidates and explain which assumptions hold or fail"
    if any(x in text for x in ["remove", "exclude", "eliminate", "rule out", "not fit", "risk", "warning"]):
        return "identify removable candidates and retain the plausible ones"
    if any(x in text for x in ["keep", "retain", "worth", "interested", "recommend"]):
        return "identify the shorter subset still worth considering"
    return "answer the user's natural-language request without expanding beyond the HesitationSet"


def generate_first_round_task(
    args,
    hesitation_set,
    hesitation_reason="",
    hesitation_evidence=None,
    uncertainty_points=None,
    item_slim_skill=None,
    matched_why="",
    matched_whens=None,
    primary_why="",
):
    hesitation_set = [str(x) for x in list(hesitation_set or []) if str(x).strip()]
    selected_what, what_reasons, secondary_from_why = _choose_first_round_what(
        hesitation_set=hesitation_set,
        hesitation_evidence=hesitation_evidence,
        uncertainty_points=uncertainty_points,
        matched_why=primary_why or matched_why,
        matched_whens=matched_whens,
    )
    user_task = _rule_user_task_for_what(
        selected_what,
        hesitation_set=hesitation_set,
        hesitation_reason=hesitation_reason,
        hesitation_evidence=hesitation_evidence,
        secondary_what=secondary_from_why,
        matched_whens=matched_whens,
    )
    task_targets = ", ".join(hesitation_set)
    criteria = ""
    expected_output = _fallback_expected_from_task(user_task)
    mapped = map_user_task_to_what(
        user_task=user_task,
        hesitation_set=hesitation_set,
        task_targets=task_targets,
        criteria=criteria,
    )
    mapped["what"] = selected_what
    mapped["primary_what"] = selected_what
    mapped["unmapped_task"] = selected_what == "none"
    mapped["mapping_confidence"] = "high" if selected_what != "none" else "low"
    mapped["mapping_reasons"] = list(what_reasons or []) + list(mapped.get("mapping_reasons", []) or [])
    secondary_rows = []
    for row in list(secondary_from_why or []) + list(mapped.get("secondary_what", []) or []):
        row = str(row or "").strip()
        if row and row != selected_what and row not in secondary_rows:
            secondary_rows.append(row)
    packet = {
        "user_task": user_task,
        "task_type_hint": mapped["what"],
        "task_targets": mapped.get("task_targets", []),
        "criteria": mapped.get("criteria", []),
        "expected_output": expected_output,
        "what": mapped["what"],
        "primary_what": mapped["primary_what"],
        "secondary_what": secondary_rows,
        "mapping_confidence": mapped.get("mapping_confidence", ""),
        "unmapped_task": bool(mapped.get("unmapped_task", False)),
        "unmapped_parts": mapped.get("unmapped_parts", []),
        "task_source": "rule_first_round_from_why_and_hesitation",
        "primary_why": str(primary_why or matched_why or ""),
        "matched_why": [str(x) for x in _as_list(matched_whens) if str(x).strip()],
        "mapping_reasons": list(mapped.get("mapping_reasons", []) or []),
        "raw_task_generation": "",
        "raw_task_generation_attempts": [{"attempt": "rule_generated", "response": user_task}],
    }
    return packet


def retarget_first_round_task(packet, selected_what, hesitation_set=None, hesitation_evidence=None):
    packet = dict(packet or {})
    selected_what = str(selected_what or "").strip()
    if not selected_what or str(packet.get("task_source", "") or "") != "rule_first_round_from_why_and_hesitation":
        return packet
    previous_what = str(packet.get("what", "") or packet.get("primary_what", "") or "")
    if previous_what == selected_what:
        return packet
    targets = [str(x).strip() for x in _as_list(hesitation_set) if str(x).strip()]
    if not targets:
        targets = [str(x).strip() for x in _as_list(packet.get("task_targets", [])) if str(x).strip()]
    user_task = _rule_user_task_for_what(
        selected_what,
        hesitation_set=targets,
        hesitation_evidence=hesitation_evidence,
        secondary_what=packet.get("secondary_what", []),
        matched_whens=packet.get("matched_why", []),
    )
    packet["legacy_mapped_what"] = previous_what
    packet["what"] = selected_what
    packet["primary_what"] = selected_what
    packet["task_type_hint"] = selected_what
    packet["user_task"] = user_task
    packet["expected_output"] = _fallback_expected_from_task(user_task)
    packet["mapping_confidence"] = "high" if selected_what != "none" else "low"
    reasons = list(packet.get("mapping_reasons", []) or [])
    reasons.append(f"first-round task retargeted from {previous_what or 'none'} to route-selected what:{selected_what}")
    packet["mapping_reasons"] = reasons
    return packet


def task_packet_from_feedback(feedback, previous_how="", advisor_count=0, hesitation_set=None):
    user_task = _clean(feedback)
    mapped = map_user_task_to_what(user_task=user_task, hesitation_set=hesitation_set or [])
    if user_task and str(mapped.get("what", "") or "") == "none":
        mapped["unmapped_task"] = True
        mapped["mapping_confidence"] = "low"
        mapped["mapping_reasons"] = list(mapped.get("mapping_reasons", []) or []) + [
            "legacy keyword mapping found no stable what; follow-up selector will use selection_profile_v1"
        ]
        mapped["tree_need_signals"] = [
            {
                "level": "what",
                "suggested_node_hint": "unmapped-followup-task",
                "why_current_nodes_insufficient": "A non-empty follow-up FeedbackToAdvisors task did not map to any active what node.",
                "evidence_pattern": "follow-up user task mapped to what=none and used the generic task template",
                "support_strength": "single_user_low",
                "source": "deterministic_task_mapping",
            }
        ]
    legacy_what = str(mapped.get("what", "") or "")
    return {
        "user_task": user_task,
        "task_type_hint": legacy_what,
        "task_targets": mapped.get("task_targets", []),
        "criteria": mapped.get("criteria", []),
        "expected_output": "answer the user's continued feedback task",
        "what": "",
        "primary_what": "",
        "secondary_what": mapped.get("secondary_what", []),
        "how": "",
        "legacy_mapped_what": legacy_what,
        "mapping_confidence": mapped.get("mapping_confidence", ""),
        "unmapped_task": bool(mapped.get("unmapped_task", False)),
        "unmapped_parts": mapped.get("unmapped_parts", []),
        "tree_need_signals": list(mapped.get("tree_need_signals", []) or []),
        "task_source": "feedback_to_advisors",
        "mapping_reasons": list(mapped.get("mapping_reasons", []) or []) + [
            "legacy what mapping retained for debug only; path selection uses what.selection_profile"
        ],
    }
