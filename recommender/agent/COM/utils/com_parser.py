import json
import re


CORE_LABELS = [
    "Reason",
    "Item",
    "RankedCandidates",
    "CandidateEvidence",
    "CandidateShortlist",
    "HesitationEvidence",
    "HesitationSet",
    "HesitationShortlist",
    "HesitationReason",
    "CommunicationTask",
    "Communication Task",
    "UserTask",
    "User Task",
    "UncertainCandidates",
    "Confidence",
    "CurrentDecision",
    "Current Decision",
    "DecisionConfidence",
    "Decision Confidence",
    "DecisionState",
    "Decision State",
    "UncertaintyPoints",
    "Uncertainty Points",
    "UserClarificationAnswers",
    "User Clarification Answers",
    "ClarificationAnswers",
    "Clarification Answers",
    "FeedbackToAdvisors",
    "Feedback To Advisors",
    "NextRoundFocus",
    "Next Round Focus",
    "NextRoundHesitationSet",
    "Next Round Hesitation Set",
    "RemovedFromHesitationSet",
    "Removed From Hesitation Set",
    "RemovedFromHesitation",
    "Removed From Hesitation",
    "SilentOrMissingEvidence",
    "Silent Or Missing Evidence",
    "CommunicationAction",
    "Communication Action",
    'MatchedWhyNodes',
    'Matched Why Nodes',
    'SelectedWhy',
    'Selected Why',
    "SelectedPath",
    "Selected Path",
    "NoCommunicationReason",
    "No Communication Reason",
    "CandidatePaths",
    "Candidate Paths",
    "Reasons",
]


def _clean_model_text(text):
    body = str(text or "").strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json|text)?\s*", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s*```$", "", body)
    body = body.replace("\r\n", "\n").replace("\r", "\n").replace("：", ":")
    return body.strip()


def _label_pattern(labels):
    parts = []
    for label in labels:
        words = re.split(r"[\s_-]+", str(label).strip())
        parts.append(r"[\s_-]*".join(re.escape(w) for w in words if w))
    return "(?:" + "|".join(parts) + ")"


def _extract_label_value(text, labels, stop_labels=None):
    body = _clean_model_text(text)
    stops = stop_labels or CORE_LABELS
    label_pat = _label_pattern(labels)
    stop_pat = _label_pattern(stops)
    pattern = (
        rf"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\s*{label_pat}\s*(?:\*\*)?\s*[:?]\s*"
        rf"(.*?)(?=\n\s*(?:[-*]\s*)?(?:\*\*)?\s*{stop_pat}\s*(?:\*\*)?\s*[:?]|\Z)"
    )
    m = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().strip("*").strip()


def _normalize_choice(value, allowed):
    raw = str(value or "").strip().lower()
    raw = re.sub(r"^[`*_\"'\s]+|[`*_\"'\s]+$", "", raw)
    raw = re.sub(r"\s+", " ", raw)
    if raw in allowed:
        return raw
    first = re.split(r"[\s,.;，。；:：()（）\[\]{}]+", raw, maxsplit=1)[0]
    if first in allowed:
        return first
    for choice in allowed:
        if re.search(rf"\b{re.escape(choice)}\b", raw):
            return choice
    return None


def _parse_confidence_value(value):
    m = re.search(r"\d+", str(value or ""))
    if not m:
        return None
    try:
        return max(0, min(100, int(m.group(0))))
    except Exception:
        return None


def parse_user_proposal(text):
    if not text:
        return None, None
    text = _clean_model_text(text)
    reason = _extract_label_value(text, ["Reason"])
    item = _extract_label_value(text, ["Item"])
    if item:
        return (reason or "").strip(), item.strip()
    m = re.search(
        r"Reason:\s*(.*?)\s*Item:\s*(.*?)(?:\s*(?:RankedCandidates|CandidateEvidence|CandidateShortlist|HesitationEvidence|HesitationSet|HesitationShortlist|HesitationReason|CommunicationTask|UserTask|UncertainCandidates|Confidence|CurrentDecision|DecisionConfidence|DecisionState|UncertaintyPoints):|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        # try the other order
        m2 = re.search(
            r"Item:\s*(.*?)\s*Reason:\s*(.*?)(?:\s*(?:RankedCandidates|CandidateEvidence|CandidateShortlist|HesitationEvidence|HesitationSet|HesitationShortlist|HesitationReason|CommunicationTask|UserTask|UncertainCandidates|Confidence|CurrentDecision|DecisionConfidence|DecisionState|UncertaintyPoints):|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m2:
            return m2.group(2).strip(), m2.group(1).strip()
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def parse_candidate_shortlist(text):
    if not text:
        return []
    text = _clean_model_text(text)
    m = re.search(
        r"(?:HesitationSet|HesitationShortlist|CandidateShortlist|UncertainCandidates):\s*(.*?)(?:\s*(?:RankedCandidates|CandidateEvidence|HesitationEvidence|HesitationReason|CommunicationTask|UserTask|Confidence|CurrentDecision|DecisionConfidence|DecisionState|UncertaintyPoints|Reason|Item):|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []
    body = m.group(1).strip()
    if not body or body.lower() in ["none", "null", "n/a"]:
        return []
    parts = re.split(r"[,;|\n]+", body)
    out = []
    for part in parts:
        item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part.strip()).strip()
        if item:
            out.append(item)
    return out


def parse_candidate_evidence(text):
    if not text:
        return []
    text = _clean_model_text(text)
    m = re.search(
        r"(?:HesitationEvidence|CandidateEvidence):\s*(.*?)(?:\s*(?:HesitationSet|HesitationShortlist|CandidateShortlist|HesitationReason|CommunicationTask|UserTask|Confidence|CurrentDecision|DecisionConfidence|DecisionState|UncertaintyPoints|Reason:|Item:)|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []
    body = m.group(1).strip()
    if not body or body.lower() in ["none", "null", "n/a"]:
        return []
    out = []
    for line in body.splitlines():
        row = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line.strip()).strip()
        if not row:
            continue
        parts = [p.strip() for p in re.split(r"\s*\|\s*", row) if p.strip()]
        if len(parts) >= 4:
            out.append(
                {
                    "candidate": parts[0],
                    "decision": parts[1].lower(),
                    "fit": parts[2],
                    "reason": " | ".join(parts[3:]),
                }
            )
        elif len(parts) >= 2:
            out.append({"candidate": parts[0], "decision": "", "fit": "", "reason": " | ".join(parts[1:])})
        else:
            out.append({"candidate": row, "decision": "", "fit": "", "reason": ""})
    return out


def parse_hesitation_reason(text):
    if not text:
        return ""
    text = _clean_model_text(text)
    m = re.search(
        r"HesitationReason:\s*(.*?)(?:\s*(?:CommunicationTask|UserTask|Confidence|CurrentDecision|DecisionConfidence|DecisionState|UncertaintyPoints|Reason:|Item:)|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1).strip())


def parse_stage1_communication_task(text):
    value = _extract_label_value(text, ["CommunicationTask", "Communication Task", "UserTask", "User Task"])
    return re.sub(r"\s+", " ", str(value or "").strip())


def parse_user_proposal_decision(text):
    reason, item = parse_user_proposal(text)
    confidence = parse_self_confidence(text)
    shortlist = parse_candidate_shortlist(text)
    candidate_evidence = parse_candidate_evidence(text)
    hesitation_reason = parse_hesitation_reason(text)
    communication_task = parse_stage1_communication_task(text)
    return reason, item, confidence, shortlist, candidate_evidence, hesitation_reason, communication_task


def parse_advisor_review(text):
    if not text:
        return None, None, None, None
    def read_label(label):
        m = re.search(
            rf"^\s*(?:[-*]\s*)?(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*:\s*"
            rf"(.*?)(?=^\s*(?:[-*]\s*)?(?:\*\*)?[A-Za-z][A-Za-z0-9_ ]{{0,40}}(?:\*\*)?\s*:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        return m.group(1).strip() if m else ""

    advice = read_label("Advice")
    suggested = read_label("SuggestedItem")
    friend_basis = read_label("FriendBasis")
    reason_for_user = read_label("ReasonForUser")
    concern = read_label("Concern")
    response_prev = read_label("ResponseToPrevious")
    challenge_support_prev = read_label("ChallengeOrSupportPrevious")
    missing_evidence = read_label("MissingEvidence")
    answered_feedback = read_label("AnsweredFeedback")
    independent_thinking = read_label("IndependentThinking")
    task_answer = read_label("TaskAnswer")
    remove_set = read_label("RemoveSet")
    keep_set = read_label("KeepSet")
    unclear_set = read_label("UnclearSet")
    interested_set = read_label("InterestedSet")
    shrink_set = read_label("ShrinkSet")
    retain_set = read_label("RetainSet")
    weak_fit_set = read_label("WeakFitSet")
    relevant_candidates = read_label("RelevantCandidates")
    compared_candidates = read_label("ComparedCandidates") or read_label("ComparedSet")
    stronger_candidate = read_label("StrongerCandidate") or read_label("StrongerCandidates")
    weaker_candidate = read_label("WeakerCandidate") or read_label("WeakerCandidates")
    candidate_view = read_label("CandidateView")
    ask_user = read_label("AskUser")
    risk_reason = read_label("RiskReason")
    keep_reason = read_label("KeepReason")
    interest_reason = read_label("InterestReason")
    weak_fit_reason = read_label("WeakFitReason")
    key_tradeoff = read_label("KeyTradeoff")
    comparison_reason = read_label("ComparisonReason")
    covered_set = read_label("CoveredSet")
    missing_set = read_label("MissingSet")
    evidence_added = read_label("EvidenceAdded")
    evidence_can_add = read_label("EvidenceICanAdd")
    covered_evidence = read_label("CoveredEvidence")
    still_missing = read_label("StillMissing")
    reliable_reasons = read_label("ReliableReasons")
    weak_reasons = read_label("WeakReasons")
    checked_reasoning = read_label("CheckedReasoning") or read_label("CheckedCandidates")
    supported_assumption = read_label("SupportedAssumption") or reliable_reasons or read_label("ValidReasoning")
    questioned_assumption = read_label("QuestionedAssumption") or weak_reasons or read_label("WeakReasoning")
    affected_candidates = read_label("AffectedCandidates")
    correction = read_label("Correction")
    failure_repair_check = read_label("FailureRepairCheck")
    new_protocol_labels = [
        remove_set,
        keep_set,
        unclear_set,
        interested_set,
        weak_fit_set,
        compared_candidates,
        stronger_candidate,
        weaker_candidate,
        candidate_view,
        covered_set,
        missing_set,
        reliable_reasons,
        weak_reasons,
        checked_reasoning,
        supported_assumption,
        questioned_assumption,
        affected_candidates,
        failure_repair_check,
        task_answer,
    ]
    if any(str(x or "").strip() for x in new_protocol_labels):
        # Keep the full advisor response for downstream contract checking.
        # New public-tree nodes may introduce task-specific labels that this
        # generic parser does not know yet; rebuilding speech from a fixed
        # allowlist silently drops those fields and creates false missing-field
        # evolution signals.
        return "answer", str(text or "").strip(), "", ""
    if advice and (
        suggested
        or friend_basis
        or reason_for_user
        or concern
        or independent_thinking
        or task_answer
        or remove_set
        or keep_set
        or interested_set
        or shrink_set
        or retain_set
        or weak_fit_set
        or relevant_candidates
        or compared_candidates
        or stronger_candidate
        or weaker_candidate
        or covered_evidence
        or still_missing
        or checked_reasoning
        or supported_assumption
        or questioned_assumption
        or candidate_view
    ):
        none_like = {"none", "null", "n/a", "na", ""}
        neg_parts = []
        if concern and concern.strip().lower() not in none_like:
            neg_parts.append(f"Concern: {concern}")
        if remove_set and remove_set.strip().lower() not in none_like:
            neg_parts.append(f"RemoveSet: {remove_set}")
        if shrink_set and shrink_set.strip().lower() not in none_like:
            neg_parts.append(f"RemoveSet: {shrink_set}")
        if weak_fit_set and weak_fit_set.strip().lower() not in none_like:
            neg_parts.append(f"WeakFitSet: {weak_fit_set}")
        if weaker_candidate and weaker_candidate.strip().lower() not in none_like:
            neg_parts.append(f"WeakerCandidate: {weaker_candidate}")
        if still_missing and still_missing.strip().lower() not in none_like:
            neg_parts.append(f"StillMissing: {still_missing}")
        if questioned_assumption and questioned_assumption.strip().lower() not in none_like:
            neg_parts.append(f"QuestionedAssumption: {questioned_assumption}")
        pos_parts = []
        if friend_basis and friend_basis.strip().lower() not in none_like:
            pos_parts.append(f"FriendBasis: {friend_basis}")
        if independent_thinking and independent_thinking.strip().lower() not in none_like:
            pos_parts.append(f"IndependentThinking: {independent_thinking}")
        if task_answer and task_answer.strip().lower() not in none_like:
            pos_parts.append(f"TaskAnswer: {task_answer}")
        if keep_set and keep_set.strip().lower() not in none_like:
            pos_parts.append(f"KeepSet: {keep_set}")
        if interested_set and interested_set.strip().lower() not in none_like:
            pos_parts.append(f"InterestedSet: {interested_set}")
        if retain_set and retain_set.strip().lower() not in none_like:
            pos_parts.append(f"KeepSet: {retain_set}")
        if relevant_candidates and relevant_candidates.strip().lower() not in none_like:
            pos_parts.append(f"RelevantCandidates: {relevant_candidates}")
        if stronger_candidate and stronger_candidate.strip().lower() not in none_like:
            pos_parts.append(f"StrongerCandidate: {stronger_candidate}")
        if evidence_can_add and evidence_can_add.strip().lower() not in none_like:
            pos_parts.append(f"EvidenceICanAdd: {evidence_can_add}")
        if covered_evidence and covered_evidence.strip().lower() not in none_like:
            pos_parts.append(f"EvidenceICanAdd: {covered_evidence}")
        if checked_reasoning and checked_reasoning.strip().lower() not in none_like:
            pos_parts.append(f"CheckedReasoning: {checked_reasoning}")
        if supported_assumption and supported_assumption.strip().lower() not in none_like:
            pos_parts.append(f"SupportedAssumption: {supported_assumption}")
        if response_prev and response_prev.strip().lower() not in none_like:
            pos_parts.append(f"ResponseToPrevious: {response_prev}")
        if missing_evidence and missing_evidence.strip().lower() not in none_like:
            pos_parts.append(f"MissingEvidence: {missing_evidence}")
        if answered_feedback and answered_feedback.strip().lower() not in none_like:
            pos_parts.append(f"AnsweredFeedback: {answered_feedback}")
        speech = f"Neg:\n{chr(10).join(neg_parts) if neg_parts else 'none'}\nPos:\n{chr(10).join(pos_parts) if pos_parts else 'none'}"
        alt = "" if suggested.strip().lower() in none_like else suggested.strip()
        return advice.strip().lower(), speech.strip(), alt, " ".join(pos_parts).strip()

    decision_new = read_label("Decision")
    defended = read_label("DefendedItem")
    challenged = read_label("ChallengedItem")
    evidence_for = read_label("EvidenceFor")
    evidence_against = read_label("EvidenceAgainst")
    if decision_new and (defended or evidence_for or evidence_against):
        extras = []
        for label in [
            "WarningEvidence",
            "PromotionEvidence",
            "AddedCoverage",
            "ChallengeToPreviousClaim",
            "AnsweredFeedback",
            "ResponseToMemory",
            "StillMissing",
            "UnresolvedIssue",
        ]:
            value = read_label(label)
            if value:
                extras.append(f"{label}: {value}")
        neg_parts = []
        if challenged and challenged.strip().lower() not in ["none", "null", "n/a"]:
            neg_parts.append(f"ChallengedItem: {challenged}")
        if evidence_against and evidence_against.strip().lower() not in ["none", "null", "n/a"]:
            neg_parts.append(f"EvidenceAgainst: {evidence_against}")
        pos_parts = []
        if evidence_for and evidence_for.strip().lower() not in ["none", "null", "n/a"]:
            pos_parts.append(evidence_for)
        pos_parts.extend(extras)
        speech = f"Neg:\n{chr(10).join(neg_parts) if neg_parts else 'none'}\nPos:\n{chr(10).join(pos_parts) if pos_parts else 'none'}"
        return decision_new.strip().lower(), speech.strip(), defended.strip(), " ".join(pos_parts).strip()

    m = re.search(
        r"Decision:\s*(.*?)\s*AltItem:\s*(.*?)\s*FriendSpeech:\s*(.*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        decision, alt, speech = m.groups()
        decision = decision.strip().lower()
        return decision, speech.strip(), alt.strip(), ""
    m = re.search(
        r"Decision:\s*(.*?)\s*Neg:\s*(.*?)\s*AltItem:\s*(.*?)\s*Pos:\s*(.*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None, None, None, None
    decision, neg, alt, pos = m.groups()
    decision = decision.strip().lower()
    speech = f"Neg: {neg.strip()} Pos: {pos.strip()}".strip()
    return decision, speech, alt.strip(), pos.strip()


def parse_user_arbitration(text):
    if not text:
        return None, None
    text = text + "\n"
    m = re.findall(r"Path:\s*(.*?)\nReason:\s*(.*?)\n", text, re.DOTALL)
    if not m:
        return None, None
    path, reason = m[0]
    path = path.strip().upper()
    if path not in ["A", "B"]:
        return None, None
    return path, reason.strip()


def parse_target_profile(text):
    if not text:
        return None
    m = re.search(r"Profile:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return text.strip()  # Fallback to entire text if formatting is missing
    return m.group(1).strip()


def parse_friend_summary(text):
    if not text:
        return None, None, None
    m = re.search(
        r"CurrentItemCons:\s*(.*?)\s*NewItemPros:\s*(.*?)\s*VoteSummary:\s*(.*?)\s*RecommendedItem:\s*(.*?)\s*OpenIssue:\s*(.*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        cons, pros, votes, item, issue = m.groups()
        reason = f"CurrentItemCons: {cons.strip()} NewItemPros: {pros.strip()} VoteSummary: {votes.strip()}"
        return reason.strip(), item.strip(), issue.strip()
    m = re.search(
        r"SummaryReason:\s*(.*?)\s*RecommendedItem:\s*(.*?)\s*OpenIssue:\s*(.*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None, None, None
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


def parse_self_confidence(text):
    if not text:
        return None
    value = _extract_label_value(text, ["Confidence", "DecisionConfidence", "Decision Confidence"])
    if value is None:
        m = re.search(r"Confidence\s*[:=]\s*(\d+)", _clean_model_text(text), re.DOTALL | re.IGNORECASE)
        value = m.group(1) if m else None
    return _parse_confidence_value(value)


def parse_stage6_decision(text):
    if not text:
        return None
    text = _clean_model_text(text)
    current_decision = None
    decision_confidence = None
    decision_state = None
    uncertainty_points = []
    feedback_to_advisors = []
    next_round_focus = []
    silent_or_missing_evidence = []
    removed_from_hesitation = []
    user_clarification_answers = []

    current_decision = _normalize_choice(
        _extract_label_value(text, ["CurrentDecision", "Current Decision"]),
        ["keep", "switch"],
    )
    decision_confidence = _parse_confidence_value(
        _extract_label_value(text, ["DecisionConfidence", "Decision Confidence", "Confidence"])
    )
    decision_state = _normalize_choice(
        _extract_label_value(text, ["DecisionState", "Decision State"]),
        ["final", "continue"],
    )
    raw = _extract_label_value(text, ["UncertaintyPoints", "Uncertainty Points"])
    if raw is not None:
        raw = raw.strip()
        if raw and raw.lower() not in ["none", "null", "n/a"]:
            uncertainty_points = [seg.strip() for seg in re.split(r"[,;\n]+", raw) if seg.strip()]
    raw_clarification = _extract_label_value(
        text,
        ["UserClarificationAnswers", "User Clarification Answers", "ClarificationAnswers", "Clarification Answers"],
    )
    if raw_clarification is not None and raw_clarification.strip().lower() not in ["", "none", "null", "n/a"]:
        user_clarification_answers = [
            seg.strip("- ").strip()
            for seg in re.split(r"[\n;]+", raw_clarification)
            if seg.strip("- ").strip()
        ]
    raw_feedback = _extract_label_value(text, ["FeedbackToAdvisors", "Feedback To Advisors"])
    if raw_feedback is not None and raw_feedback.strip().lower() not in ["", "none", "null", "n/a"]:
        feedback_to_advisors = [seg.strip("- ").strip() for seg in re.split(r"[\n;]+", raw_feedback) if seg.strip("- ").strip()]
    raw_focus = _extract_label_value(text, ["NextRoundHesitationSet", "Next Round Hesitation Set", "NextRoundFocus", "Next Round Focus"])
    if raw_focus is not None and raw_focus.strip().lower() not in ["", "none", "null", "n/a"]:
        next_round_focus = [seg.strip("- ").strip() for seg in re.split(r"[,;\n]+", raw_focus) if seg.strip("- ").strip()]
    raw_removed = _extract_label_value(text, ["RemovedFromHesitationSet", "Removed From Hesitation Set", "RemovedFromHesitation", "Removed From Hesitation"])
    if raw_removed is not None and raw_removed.strip().lower() not in ["", "none", "null", "n/a"]:
        removed_from_hesitation = [seg.strip("- ").strip() for seg in re.split(r"[,;\n]+", raw_removed) if seg.strip("- ").strip()]
    raw_missing = _extract_label_value(text, ["SilentOrMissingEvidence", "Silent Or Missing Evidence"])
    if raw_missing is not None and raw_missing.strip().lower() not in ["", "none", "null", "n/a"]:
        silent_or_missing_evidence = [seg.strip("- ").strip() for seg in re.split(r"[\n;]+", raw_missing) if seg.strip("- ").strip()]
    if not uncertainty_points and feedback_to_advisors:
        uncertainty_points = list(feedback_to_advisors)

    if current_decision not in ["keep", "switch"]:
        return None
    if decision_state not in ["final", "continue"]:
        return None
    if decision_confidence is None:
        return None
    return {
        "current_decision": current_decision,
        "decision_confidence": decision_confidence,
        "decision_state": decision_state,
        "uncertainty_points": uncertainty_points,
        "user_clarification_answers": user_clarification_answers,
        "feedback_to_advisors": feedback_to_advisors,
        "next_round_focus": next_round_focus,
        "removed_from_hesitation": removed_from_hesitation,
        "silent_or_missing_evidence": silent_or_missing_evidence,
    }


def parse_user_final_decision(text):
    parsed = parse_user_proposal_decision(text)
    reason, item, confidence = parsed[:3]
    arbitration = parse_stage6_decision(text)
    if not item or not arbitration:
        json_parsed = parse_user_final_decision_json(text)
        if json_parsed:
            return json_parsed
        return None
    if confidence is not None and arbitration.get("decision_confidence") is None:
        arbitration["decision_confidence"] = confidence
    return {
        "reason": reason or "",
        "item": item,
        "arbitration": arbitration,
    }


def _json_key_lookup(data, keys):
    if not isinstance(data, dict):
        return None
    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(k).lower()): v
        for k, v in data.items()
    }
    for key in keys:
        value = normalized.get(re.sub(r"[^a-z0-9]+", "", str(key).lower()))
        if value is not None:
            return value
    return None


def _split_list_value(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip() and str(x).strip().lower() not in ["none", "null", "n/a"]]
    raw = str(value or "").strip()
    if raw.lower() in ["", "none", "null", "n/a"]:
        return []
    return [seg.strip("- ").strip() for seg in re.split(r"[\n;]+", raw) if seg.strip("- ").strip()]


def _extract_json_payload(text):
    body = _clean_model_text(text)
    if not body:
        return None
    candidates = [body]
    first = body.find("{")
    last = body.rfind("}")
    if first >= 0 and last > first:
        candidates.append(body[first : last + 1])
    for raw in candidates:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def parse_user_final_decision_json(text):
    data = _extract_json_payload(text)
    if not isinstance(data, dict):
        return None
    arbitration_obj = _json_key_lookup(data, ["arbitration", "decision", "decision_state"]) or {}
    if not isinstance(arbitration_obj, dict):
        arbitration_obj = {}

    reason = _json_key_lookup(data, ["reason", "rationale", "explanation"])
    item = _json_key_lookup(data, ["item", "final_item", "decision_item", "selected_item", "recommended_item"])
    if item is None:
        item = _json_key_lookup(arbitration_obj, ["item", "decision_item", "selected_item", "recommended_item"])
    current_decision = _normalize_choice(
        _json_key_lookup(data, ["current_decision", "currentdecision", "decision_action"])
        or _json_key_lookup(arbitration_obj, ["current_decision", "currentdecision", "decision_action"]),
        ["keep", "switch"],
    )
    decision_confidence = _parse_confidence_value(
        _json_key_lookup(data, ["decision_confidence", "decisionconfidence", "confidence"])
        or _json_key_lookup(arbitration_obj, ["decision_confidence", "decisionconfidence", "confidence"])
    )
    decision_state = _normalize_choice(
        _json_key_lookup(data, ["decision_state", "decisionstate", "state"])
        or _json_key_lookup(arbitration_obj, ["decision_state", "decisionstate", "state"]),
        ["final", "continue"],
    )
    feedback = _split_list_value(
        _json_key_lookup(data, ["feedback_to_advisors", "feedbacktoadvisors", "feedback"])
        or _json_key_lookup(arbitration_obj, ["feedback_to_advisors", "feedbacktoadvisors", "feedback"])
    )
    clarification = _split_list_value(
        _json_key_lookup(data, ["user_clarification_answers", "userclarificationanswers", "clarification_answers", "clarificationanswers"])
        or _json_key_lookup(arbitration_obj, ["user_clarification_answers", "userclarificationanswers", "clarification_answers", "clarificationanswers"])
    )
    next_focus = _split_list_value(
        _json_key_lookup(data, ["next_round_hesitation_set", "nextroundhesitationset", "next_round_focus", "nextroundfocus"])
        or _json_key_lookup(arbitration_obj, ["next_round_hesitation_set", "nextroundhesitationset", "next_round_focus", "nextroundfocus"])
    )
    removed = _split_list_value(
        _json_key_lookup(data, ["removed_from_hesitation_set", "removedfromhesitationset", "removed_from_hesitation", "removedfromhesitation"])
        or _json_key_lookup(arbitration_obj, ["removed_from_hesitation_set", "removedfromhesitationset", "removed_from_hesitation", "removedfromhesitation"])
    )
    uncertainty = _split_list_value(
        _json_key_lookup(data, ["uncertainty_points", "uncertaintypoints"])
        or _json_key_lookup(arbitration_obj, ["uncertainty_points", "uncertaintypoints"])
    )
    if not uncertainty and feedback:
        uncertainty = list(feedback)
    if not item:
        return None
    if current_decision not in ["keep", "switch"]:
        current_decision = "keep"
    if decision_state not in ["final", "continue"]:
        decision_state = "continue" if feedback else "final"
    if decision_confidence is None:
        decision_confidence = 60
    return {
        "reason": str(reason or "").strip(),
        "item": str(item or "").strip(),
        "arbitration": {
            "current_decision": current_decision,
            "decision_confidence": decision_confidence,
            "decision_state": decision_state,
            "uncertainty_points": uncertainty,
            "user_clarification_answers": clarification,
            "feedback_to_advisors": feedback,
            "next_round_focus": next_focus,
            "removed_from_hesitation": removed,
            "silent_or_missing_evidence": [],
        },
    }


def parse_communication_control(text):
    if not text:
        return None
    text = _clean_model_text(text)
    action = None
    confidence = None
    reasons = []

    action = _normalize_choice(
        _extract_label_value(text, ["CommunicationAction", "Communication Action", "Action"]),
        ["skip", "start", "continue", "stop"],
    )
    selected_why = _extract_label_value(text, ['SelectedWhy', 'Selected Why'])
    selected_why_norm = str(selected_why or "").strip().lower()
    if action is None and selected_why is not None:
        action = "skip" if selected_why_norm in ["", "none", "null", "n/a", "no", "skip"] else "start"
    confidence = _parse_confidence_value(
        _extract_label_value(text, ["DecisionConfidence", "Decision Confidence", "Confidence"])
    )
    raw = _extract_label_value(text, ["Reasons", "Reason"])
    if raw is not None:
        raw = raw.strip()
        if raw and raw.lower() not in ["none", "null", "n/a"]:
            reasons = [seg.strip() for seg in re.split(r"[,;\n]+", raw) if seg.strip()]

    if action not in ["skip", "start", "continue", "stop"]:
        return None
    return {
        "communication_action": action,
        "decision_confidence": confidence,
        "selected_why": "" if selected_why is None else str(selected_why).strip(),
        "matched_why_nodes": parse_matched_why_nodes(text),
        "no_communication_reason": _extract_label_value(text, ["NoCommunicationReason", "No Communication Reason"]) or "",
        "reasons": reasons,
    }


def parse_skill_controller(text):
    if not text:
        return None
    next_skill = None
    next_skill_action = None
    reasons = []

    m = re.search(r"(?:InitialSkill|NextSkill):\s*(.*)", text, re.IGNORECASE)
    if m:
        next_skill = m.group(1).strip()
    m = re.search(r"(?:SkillAction|NextSkillAction):\s*(.*)", text, re.IGNORECASE)
    if m:
        next_skill_action = m.group(1).strip().lower()
    m = re.search(r"Reasons:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        if raw and raw.lower() != "none":
            reasons = [seg.strip() for seg in re.split(r"[,;\n]+", raw) if seg.strip()]

    if not next_skill:
        return None
    return {
        "next_skill": next_skill,
        "next_skill_action": next_skill_action,
        "reasons": reasons,
    }


def parse_communication_path_choice(text):
    if not text:
        return None

    def read_field(name):
        selected = _extract_selected_path_block(text)
        body = selected if selected else text
        m = re.search(rf"^\s*{name}:\s*(.*)", body, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else ""

    why = _extract_label_value(text, ['SelectedWhy', 'Selected Why']) or ""
    who = read_field("Who")
    what = read_field("What")
    how = read_field("How")
    confidence = read_field("Confidence")
    reasons = []
    m = re.search(r"Reasons:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        if raw and raw.lower() != "none":
            reasons = [seg.strip("- ").strip() for seg in re.split(r"[\n;]+", raw) if seg.strip("- ").strip()]

    if not any([why, who, how]):
        return None
    try:
        confidence_value = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence_value = 0.0
    return {
        'why': why,
        "who": who,
        "legacy_what": what,
        "how": how,
        "confidence": confidence_value,
        "reasons": reasons,
        "matched_why_nodes": parse_matched_why_nodes(text),
        "candidate_paths": parse_candidate_paths(text),
    }


def _extract_selected_path_block(text):
    body = _clean_model_text(text)
    m = re.search(
        r"(?:^|\n)\s*(?:SelectedPath|Selected Path)\s*:\s*(.*?)(?=\n\s*(?:Reasons|NoCommunicationReason|No Communication Reason|CommunicationAction|Communication Action)\s*:|\Z)",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def parse_matched_why_nodes(text):
    raw = _extract_label_value(text, ['MatchedWhyNodes', 'Matched Why Nodes'])
    if raw is None:
        return []
    raw = raw.strip()
    if not raw or raw.lower() in ["none", "null", "n/a"]:
        return []
    out = []
    for line in raw.splitlines():
        row = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line.strip())
        if not row:
            continue
        node, _, reason = row.partition(":")
        out.append({'why': node.strip(), "reason": reason.strip()})
    if not out:
        for seg in re.split(r"[,;]+", raw):
            seg = seg.strip()
            if seg:
                out.append({'why': seg, "reason": ""})
    return out


def parse_candidate_paths(text):
    raw = _extract_label_value(text, ["CandidatePaths", "Candidate Paths"])
    if raw is None:
        return []
    paths = []
    for line in raw.splitlines():
        row = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line.strip())
        if not row:
            continue
        fields = {}
        for part in row.split(";"):
            key, sep, value = part.partition("=")
            if sep:
                fields[key.strip().lower()] = value.strip()
        if fields:
            paths.append(fields)
    return paths


def parse_user_reply(text):
    if not text:
        return None
    m = re.search(r"Reply:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return text.strip()
    return m.group(1).strip()
