import json
import re

from recommender.agent.COM.tree_engine.schemas import build_redecision_state


class ReDecisionMaker:
    @staticmethod
    def _compact_text(text, max_len=260):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if max_len is None:
            return text
        try:
            max_len = int(max_len)
        except Exception:
            max_len = 260
        if max_len <= 0 or len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _compact_json(value, max_len=600):
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            text = str(value or "")
        return ReDecisionMaker._compact_text(text, max_len=max_len)

    @staticmethod
    def _format_evidence_summary(summary):
        summary = dict(summary or {})
        if not summary:
            return "none"
        lines = [
            f"DecisionPolicy: {summary.get('decision_policy', 'non_binding; user decides')}",
            f"DiscussionResult: {summary.get('discussion_result', 'unknown')}",
            f"RecommendedNextState: {summary.get('recommended_next_state', 'unknown')}",
            f"RetainedCandidates: {', '.join(summary.get('retained_candidates', []) or []) or 'none'}",
            f"ProposalItem: {summary.get('proposal_item', '')}",
        ]
        if str(summary.get("summary_type", "") or "") == "direct_single_advisor_evidence":
            msg = dict(summary.get("single_advisor_message", {}) or {})
            lines.extend(
                [
                    f"SingleAdvisor: {summary.get('single_advisor', 'advisor')}",
                    f"AdvisorDefendedItem: {msg.get('defended_item', '') or 'none'}",
                    f"AdvisorChallengedItem: {msg.get('challenged_item', '') or 'none'}",
                    f"AdvisorSupportReason: {ReDecisionMaker._compact_text(msg.get('support_reason', ''), 220) or 'none'}",
                    f"AdvisorOpposeReason: {ReDecisionMaker._compact_text(msg.get('oppose_reason', ''), 180) or 'none'}",
                    f"AdvisorTaskAnswer: {ReDecisionMaker._compact_text(msg.get('task_answer', ''), 220) or 'none'}",
                    f"AdvisorChallengeOrResponse: {ReDecisionMaker._compact_text(msg.get('challenge_or_support_previous', '') or msg.get('response_to_previous', ''), 220) or 'none'}",
                    f"AdvisorTradeoff: {ReDecisionMaker._compact_text(msg.get('key_tradeoff', '') or msg.get('comparison_reason', ''), 220) or 'none'}",
                    f"AdvisorCorrection: {ReDecisionMaker._compact_text(msg.get('correction', '') or msg.get('questioned_assumption', ''), 220) or 'none'}",
                ]
            )
        silent = [str(x) for x in list(summary.get("silent_focus_candidates", []) or []) if str(x).strip()]
        support_only = [str(x) for x in list(summary.get("support_only_candidates", []) or []) if str(x).strip()]
        if silent:
            lines.append(
                "SilentHesitationSetItems: "
                + ", ".join(silent)
                + " (missing advisor evidence; do not treat silence as negative evidence)"
            )
        if support_only:
            lines.append(
                "SupportOnlyCandidates: "
                + ", ".join(support_only)
                + " (unchallenged support may reflect advisor assignment, not true superiority)"
            )
        candidate_evidence = list(summary.get("candidate_evidence", []) or [])
        if candidate_evidence:
            lines.append("CandidateEvidence:")
            for row in candidate_evidence[:8]:
                row = dict(row or {})
                counts = dict(row.get("counts", {}) or {})
                count_text = (
                    f"{int(counts.get('support', 0) or 0)} support, "
                    f"{int(counts.get('risk', 0) or 0)} risk, "
                    f"{int(counts.get('unclear', 0) or 0)} unclear"
                )
                lines.append(
                    "- "
                    + str(row.get("candidate", "") or "")
                    + " | "
                    + str(row.get("status", "unclear") or "unclear")
                    + " | "
                    + count_text
                    + " | "
                    + (ReDecisionMaker._compact_text(row.get("reason", ""), 220) or "none")
                )
        discussion_summary = dict(summary.get("discussion_summary", {}) or {})
        if discussion_summary:
            lines.append("DiscussionSummary:")
            lines.append(f"  MainAgreement: {discussion_summary.get('main_agreement', 'none') or 'none'}")
            lines.append(f"  MainConflict: {discussion_summary.get('main_conflict', 'none') or 'none'}")
            lines.append(f"  RemainingGap: {discussion_summary.get('remaining_gap', 'none') or 'none'}")
            questions = [str(x) for x in list(discussion_summary.get("advisor_questions_for_user", []) or []) if str(x).strip()]
            lines.append(f"  AdvisorQuestionsForUser: {' | '.join(questions[:4]) if questions else 'none'}")
        interactions = [dict(x or {}) for x in list(summary.get("advisor_interactions", []) or []) if isinstance(x, dict)]
        if interactions:
            lines.append("AdvisorInteractions:")
            for row in interactions[:6]:
                label = str(row.get("type", "") or "Interaction")
                reason = ReDecisionMaker._compact_text(row.get("reason", ""), 220)
                advisor = str(row.get("advisor", "") or "advisor")
                if reason:
                    lines.append(f"- {label} | {advisor} | {reason}")
        comparative = [dict(x or {}) for x in list(summary.get("comparative_claims", []) or []) if isinstance(x, dict)]
        if comparative:
            lines.append("ComparativeSignals:")
            for row in comparative[:6]:
                label = str(row.get("type", "") or "Comparison")
                reason = ReDecisionMaker._compact_text(row.get("reason", ""), 220)
                advisor = str(row.get("advisor", "") or "advisor")
                if reason:
                    lines.append(f"- {label} | {advisor} | {reason}")
        corrections = [dict(x or {}) for x in list(summary.get("correction_claims", []) or []) if isinstance(x, dict)]
        if corrections:
            lines.append("CorrectionsAndChallenges:")
            for row in corrections[:6]:
                label = str(row.get("type", "") or "Correction")
                reason = ReDecisionMaker._compact_text(row.get("reason", ""), 220)
                advisor = str(row.get("advisor", "") or "advisor")
                if reason:
                    lines.append(f"- {label} | {advisor} | {reason}")
        by_candidate = dict(summary.get("by_candidate", {}) or {})
        for item, row in by_candidate.items():
            if candidate_evidence:
                break
            row = dict(row or {})
            support = [
                ReDecisionMaker._compact_text(x.get("reason", ""), 180)
                for x in list(row.get("support", []) or [])[:2]
                if str(x.get("reason", "") or "").strip()
            ]
            against = [
                ReDecisionMaker._compact_text(x.get("reason", ""), 180)
                for x in list(row.get("against", []) or [])[:2]
                if str(x.get("reason", "") or "").strip()
            ]
            if not support and not against and not row.get("defended_count") and not row.get("attacked_count"):
                continue
            lines.append(f"Candidate: {item}")
            lines.append(f"  Support: {' | '.join(support) if support else 'none'}")
            lines.append(f"  Against: {' | '.join(against) if against else 'none'}")
            lines.append(f"  Counts: defended={int(row.get('defended_count', 0) or 0)}, attacked={int(row.get('attacked_count', 0) or 0)}")
        conflicts = [
            ReDecisionMaker._compact_text(x.get("reason", ""), 180)
            for x in list(summary.get("key_conflicts", []) or [])[:4]
            if str(x.get("reason", "") or "").strip()
        ]
        unresolved_answers = [
            ReDecisionMaker._compact_text(x.get("reason", ""), 180)
            for x in list(summary.get("unresolved_advisor_answers", []) or [])[:4]
            if str(x.get("reason", "") or "").strip()
        ]
        unresolved = [
            ReDecisionMaker._compact_text(x, 160)
            for x in list(summary.get("unresolved_questions", []) or [])[:4]
            if str(x).strip()
        ]
        removed = [str(x.get("why", "") or "") for x in list(summary.get("repeated_or_weak_arguments_removed", []) or [])[:5]]
        lines.append(f"KeyConflicts: {' | '.join(conflicts) if conflicts else 'none'}")
        if unresolved_answers and not candidate_evidence:
            lines.append(f"UnresolvedAdvisorAnswers: {' | '.join(unresolved_answers)}")
        lines.append(f"UnresolvedQuestions: {' | '.join(unresolved) if unresolved else 'none'}")
        lines.append(f"WeakOrRepeatedEvidenceRemoved: {len([x for x in removed if x])}")
        return "\n".join(lines)

    @staticmethod
    def _format_discussion_result(discussion_result):
        mem = dict(discussion_result or {})
        pkt = dict(mem.get("evidence_packet", {}) or {})
        synthesis = (
            dict(pkt.get("advisor_synthesis_packet", {}) or {})
            or dict(pkt.get("synthesis_packet", {}) or {})
            or dict(pkt if pkt.get("source") in {"advisor_summary_agent_v1", "advisor_summary_agent_fallback_v1"} else {})
        )
        if synthesis:
            def compact_value(value, max_len=220, depth=0):
                if isinstance(value, dict):
                    return {
                        str(k): compact_value(v, max_len=max_len, depth=depth + 1)
                        for k, v in list(value.items())[:8]
                        if str(k).strip()
                    }
                if isinstance(value, list):
                    return [compact_value(v, max_len=max_len, depth=depth + 1) for v in value[:6]]
                return ReDecisionMaker._compact_text(value, max_len)

            task_specific_summary = synthesis.get("task_specific_summary", {})
            if not isinstance(task_specific_summary, dict):
                task_specific_summary = {}
            extra_task_summary = synthesis.get("extra_task_summary", {})
            if not isinstance(extra_task_summary, dict):
                extra_task_summary = {}
            interaction_summary = synthesis.get("interaction_summary", {})
            if not isinstance(interaction_summary, dict):
                interaction_summary = {}
            extra_interaction_summary = synthesis.get("extra_interaction_summary", {})
            if not isinstance(extra_interaction_summary, dict):
                extra_interaction_summary = {}
            decision_guidance = synthesis.get("decision_guidance", {})
            if not isinstance(decision_guidance, dict):
                decision_guidance = {}
            candidate_summaries = synthesis.get("candidate_summaries", {})
            if not isinstance(candidate_summaries, dict):
                candidate_summaries = {}
            compact_candidates = {}
            for item, row in list(candidate_summaries.items())[:8]:
                row = dict(row or {}) if isinstance(row, dict) else {"support_summary": str(row or "")}
                compact_candidates[str(item)] = {
                    "support": ReDecisionMaker._compact_text(row.get("support_summary", ""), 180),
                    "risk": ReDecisionMaker._compact_text(row.get("risk_summary", ""), 180),
                    "tradeoff": ReDecisionMaker._compact_text(row.get("tradeoff_summary", ""), 180),
                    "agreement": str(row.get("advisor_agreement", "") or "none"),
                    "evidence": [
                        ReDecisionMaker._compact_text(x, 140)
                        for x in list(row.get("key_evidence", []) or [])[:2]
                        if str(x).strip()
                    ],
                }
            safe_packet = {
                "decision_policy": "information_only_no_vote",
                "source": str(synthesis.get("source", "") or "advisor_summary_agent_v1"),
                "what_was_answered": ReDecisionMaker._compact_text(synthesis.get("what_was_answered", ""), 260),
                "candidate_summaries": compact_candidates,
                "task_specific_summary": compact_value(task_specific_summary, max_len=180),
                "decision_guidance": compact_value(decision_guidance, max_len=180),
                "extra_task_summary": compact_value(extra_task_summary, max_len=180),
                "interaction_summary": compact_value(interaction_summary, max_len=180),
                "extra_interaction_summary": compact_value(extra_interaction_summary, max_len=180),
                "remaining_uncertainty": [
                    ReDecisionMaker._compact_text(x, 160)
                    for x in list(synthesis.get("remaining_uncertainty", []) or [])[:6]
                    if str(x).strip()
                ],
                "do_not_decide_winner": True,
            }
            return (
                "FinalAdvisorSynthesisPacket:\n"
                f"{ReDecisionMaker._compact_json(safe_packet, max_len=None)}\n"
                "RawDecisionPolicy: information_only_no_vote; advisor synthesis is not a vote and does not select an item."
            )
        evidence_summary = dict(pkt.get("evidence_summary", {}) or {})
        return (
            f"RemainingUncertainty: {', '.join(mem.get('remaining_uncertainty', []) or []) or 'none'}\n"
            f"FinalAdvisorDiscussionResult:\n{ReDecisionMaker._format_evidence_summary(evidence_summary)}"
        )

    @staticmethod
    def _decision_focus_candidates(decision_state, discussion_result, proposal_name):
        out = []
        seen = set()

        def add(value):
            value = str(value or "").strip()
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                out.append(value)

        mem = dict(discussion_result or {})
        pkt = dict(mem.get("evidence_packet", {}) or {})
        evidence_summary = dict(pkt.get("evidence_summary", {}) or {})
        scoped_focus = list(evidence_summary.get("focus_candidates", []) or pkt.get("focus_candidates", []) or [])
        if scoped_focus:
            for item in scoped_focus:
                add(item)
            return out

        add(proposal_name)
        state = dict(decision_state or {})
        for key in ["candidate_shortlist", "shortlist", "focus_candidates"]:
            for item in list(state.get(key, []) or []):
                add(item)
        return out

    @staticmethod
    def _previous_user_feedback_text(decision_state):
        feedback = (decision_state or {}).get("previous_user_feedback", "")
        if isinstance(feedback, str):
            return ReDecisionMaker._compact_text(feedback, 420) or "none"
        feedback = dict(feedback or {})
        requests = [str(x) for x in list(feedback.get("feedback_to_advisors", []) or []) if str(x).strip()]
        remaining = [str(x) for x in list(feedback.get("remaining_uncertainty", []) or []) if str(x).strip()]
        silent = [str(x) for x in list(feedback.get("silent_or_missing_evidence", []) or []) if str(x).strip()]
        parts = []
        if requests:
            parts.append("User asked the next advisors to address: " + "; ".join(requests[:2]))
        if remaining:
            parts.append("Remaining uncertainty: " + ", ".join(remaining[:3]))
        if silent:
            parts.append("Missing/silent evidence: " + ", ".join(silent[:3]))
        return ReDecisionMaker._compact_text(" ".join(parts), 420) if parts else "none"

    @staticmethod
    def _path_skill_payload(path):
        path = dict(path or {})
        payload = dict(path.get("path_skill_payload", {}) or {})
        if not payload:
            payload = {
                'why': str(path.get('why', "") or ""),
                "who": str(path.get("who", "") or ""),
                "how": str(path.get("how", "") or ""),
            }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _path_context(path):
        path = dict(path or {})
        payload = dict(path.get("path_skill_payload", {}) or {})
        return json.dumps(
            {
                'why': path.get('why', ""),
                "who": path.get("who", ""),
                "how": path.get("how", ""),
                "why_use": ((payload.get('why') or {}) if isinstance(payload.get('why'), dict) else {}).get("use_why", ""),
                "who_use": ((payload.get("who") or {}) if isinstance(payload.get("who"), dict) else {}).get("use_why", ""),
                "how_use": ((payload.get("how") or {}) if isinstance(payload.get("how"), dict) else {}).get("use_why", ""),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _norm_text(value):
        return " ".join(str(value or "").strip().lower().split())

    def _candidate_mentioned_in_text(self, raw_text, candidate_names):
        text = self._norm_text(raw_text)
        if not text:
            return ""
        candidates = [str(x) for x in (candidate_names or []) if str(x or "").strip()]
        for name in sorted(candidates, key=lambda x: len(str(x)), reverse=True):
            norm = self._norm_text(name)
            if norm and norm in text:
                return name
        return ""

    @staticmethod
    def _parse_confidence_from_raw(raw_text, default=60):
        m = re.search(r"(?:confidence|DecisionConfidence|置信|信心)\D{0,20}(\d{1,3})", str(raw_text or ""), re.IGNORECASE)
        if not m:
            return int(default)
        try:
            return max(0, min(100, int(m.group(1))))
        except Exception:
            return int(default)

    def _repair_unparseable_final_decision(self, raw_response, proposal_name, candidate_names):
        candidates = [str(x) for x in (candidate_names or []) if str(x or "").strip()]
        raw_text = str(raw_response or "").strip()
        mentioned = self._candidate_mentioned_in_text(raw_text, candidates)
        fallback_item = mentioned or proposal_name
        if fallback_item not in candidates and candidates:
            fallback_item = proposal_name if proposal_name in candidates else candidates[0]
        low = raw_text.lower()
        decision_state = "continue" if re.search(r"\bcontinue\b|继续|还需要|需要继续", low) else "final"
        current_decision = "switch" if self._norm_text(fallback_item) != self._norm_text(proposal_name) else "keep"
        feedback = []
        if decision_state == "continue" and raw_text:
            feedback = [re.sub(r"\s+", " ", raw_text)[:300]]
        reason = re.sub(r"\s+", " ", raw_text).strip()
        if not reason:
            reason = "Post-feedback decision was repaired from an unparseable LLM response; keep the safest candidate under the current item-selection skill."
        return {
            "reason": reason[:600],
            "item": str(fallback_item or proposal_name),
            "arbitration": {
                "current_decision": current_decision,
                "decision_confidence": self._parse_confidence_from_raw(raw_text, default=60),
                "decision_state": decision_state,
                "uncertainty_points": list(feedback),
                "feedback_to_advisors": list(feedback),
                "user_clarification_answers": [],
                "next_round_focus": [],
                "silent_or_missing_evidence": [],
                "parse_repaired": True,
                "raw_response": raw_text[:1000],
            },
        }

    def redecide(
        self,
        host,
        user_agent,
        decision_state,
        discussion_result,
        path,
        candidate_names,
        cands_int,
        prior_hint,
        target_profile,
        history_str,
        shared_memory,
        slim_user_policy=None,
    ):
        proposal_name = str((decision_state or {}).get("proposal_item", "") or "")
        proposal_reason = str((decision_state or {}).get("proposal_reason", "") or "")
        decision_candidates = self._decision_focus_candidates(decision_state, discussion_result, proposal_name)
        communication_summary = self._format_discussion_result(discussion_result)
        previous_feedback_text = self._previous_user_feedback_text(decision_state)
        history_for_redecision = (
            "covered_by_UserReasoningSkillSlim"
            if slim_user_policy not in [None, "", {}, []]
            else str(history_str or "")
        )
        final_decision = user_agent.decide_after_feedback(
            history_str=history_for_redecision,
            candidate_names=list(decision_candidates or []),
            current_proposal=proposal_name,
            current_reason=proposal_reason,
            support_block=communication_summary,
            oppose_block="none",
            structured_memory=communication_summary,
            prior_hint=prior_hint,
            target_profile=target_profile,
            slim_user_policy=slim_user_policy,
            updated_memory=previous_feedback_text,
        ) or {}
        if final_decision.get("_parse_failed"):
            final_decision = self._repair_unparseable_final_decision(
                raw_response=final_decision.get("raw_response", ""),
                proposal_name=proposal_name,
                candidate_names=decision_candidates,
            )
        if not final_decision:
            final_decision = self._repair_unparseable_final_decision(
                raw_response="",
                proposal_name=proposal_name,
                candidate_names=decision_candidates,
            )
        revised_reason = str(final_decision.get("reason", "") or "")
        revised_name = str(final_decision.get("item", "") or "")
        if not revised_name:
            raise ValueError("User Reasoning Skill returned an empty post-feedback item.")

        revised_iid = host._match_name_to_iid(revised_name, cands_int)
        if revised_iid is None:
            candidate_names = [str(host._get_item_name(int(iid))) for iid in (cands_int or [])]
            raise ValueError(
                "User Reasoning Skill selected a post-feedback item outside the candidate set: "
                f"item={revised_name!r}; candidates={candidate_names}"
            )

        arbitration = dict(final_decision.get("arbitration", {}) or {})

        current_decision = str(arbitration.get("current_decision", "keep") or "keep")
        decision_confidence = int(arbitration.get("decision_confidence", (decision_state or {}).get("self_confidence", 0)) or 0)
        decision_state_value = str(arbitration.get("decision_state", "continue") or "continue")
        remaining_uncertainty = list(arbitration.get("uncertainty_points", (discussion_result or {}).get("remaining_uncertainty", [])) or [])

        if hasattr(host, "_name_key"):
            revised_key = host._name_key(revised_name)
            proposal_key = host._name_key(proposal_name)
            same_revised_as_proposal = bool(revised_key and proposal_key and revised_key == proposal_key)
        else:
            same_revised_as_proposal = str(revised_name or "") == str(proposal_name or "")
        if not same_revised_as_proposal:
            current_decision = "switch"
        elif current_decision not in ["keep", "switch"]:
            current_decision = "keep"

        stop_reason = "communication can stop" if decision_state_value == "final" else "communication should continue"
        arbitration_state = build_redecision_state(
            current_decision=current_decision,
            decision_item=str(revised_name or proposal_name),
            decision_confidence=decision_confidence,
            decision_state=decision_state_value,
            remaining_uncertainty=remaining_uncertainty,
            stop_reason=stop_reason,
        )
        user_clarification_answers = [
            str(x).strip()
            for x in list(arbitration.get("user_clarification_answers", []) or [])
            if str(x).strip()
        ]
        arbitration_state["user_clarification_answers"] = list(user_clarification_answers)
        arbitration_state["feedback_to_advisors"] = list(arbitration.get("feedback_to_advisors", []) or [])
        arbitration_state["next_round_focus"] = list(arbitration.get("next_round_focus", []) or [])
        arbitration_state["removed_from_hesitation"] = list(arbitration.get("removed_from_hesitation", []) or [])
        arbitration_state["silent_or_missing_evidence"] = list(arbitration.get("silent_or_missing_evidence", []) or [])
        if decision_state_value == "continue" and user_clarification_answers:
            answer_prefix = "UserClarificationAnswers: " + "; ".join(user_clarification_answers[:4])
            feedback_text = " ".join(str(x) for x in arbitration_state["feedback_to_advisors"])
            if "UserClarificationAnswers" not in feedback_text and "User clarified" not in feedback_text:
                arbitration_state["feedback_to_advisors"] = [answer_prefix] + arbitration_state["feedback_to_advisors"]
        if decision_state_value == "continue" and not arbitration_state["feedback_to_advisors"]:
            pkt = dict(dict(discussion_result or {}).get("evidence_packet", {}) or {})
            pkt_missing = list(pkt.get("silent_focus_candidates", []) or [])
            unresolved = list((pkt.get("evidence_summary", {}) or {}).get("unresolved_questions", []) or [])
            repair_requests = [str(x) for x in unresolved[:3] if str(x).strip()]
            if pkt_missing:
                repair_requests.append(
                    "Cover missing/silent HesitationSet candidates: " + ", ".join(str(x) for x in pkt_missing if str(x).strip())
                )
            arbitration_state["feedback_to_advisors"] = repair_requests or [
                "Ask advisors for direct comparison on the remaining uncertainty."
            ]
            arbitration_state["next_round_focus"] = []
            arbitration_state["silent_or_missing_evidence"] = []
        return {
            "revised_name": str(revised_name or proposal_name),
            "revised_reason": str(revised_reason or proposal_reason),
            "revised_iid": int(revised_iid) if revised_iid is not None else None,
            "arbitration": arbitration_state,
            "prompt_communication_summary": communication_summary,
            "prompt_decision_context": communication_summary,
        }
