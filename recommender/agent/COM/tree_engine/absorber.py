from recommender.agent.COM.tree_engine.schemas import build_absorbed_memory


class FeedbackAbsorber:
    def absorb(self, decision_state, path, execution_packet):
        committee = dict((execution_packet or {}).get("committee_packet", {}) or {})
        synthesis_packet = dict(committee.get("advisor_synthesis_packet", {}) or {})
        evidence_summary = dict(committee.get("evidence_summary", {}) or {})
        candidate_summaries = dict(synthesis_packet.get("candidate_summaries", {}) or {})
        task_summary = dict(synthesis_packet.get("task_specific_summary", {}) or {})
        interaction_summary = dict(synthesis_packet.get("interaction_summary", {}) or {})
        protocol_issues = list(committee.get("protocol_issues", []) or [])
        advisor_pool_empty = bool(committee.get("advisor_pool_empty", False))

        helpful_points = []
        for item, entry in list(candidate_summaries.items())[:6]:
            entry = dict(entry or {})
            text = str(entry.get("support_summary", "") or entry.get("tradeoff_summary", "") or "").strip()
            if text:
                helpful_points.append(f"{item}: {text}")
        if not helpful_points:
            for text in list(interaction_summary.get("main_agreements", []) or [])[:3]:
                text = str(text or "").strip()
                if text:
                    helpful_points.append(text)
        if not helpful_points:
            for row in list((dict(evidence_summary.get("discussion_summary", {}) or {})).values())[:3]:
                text = str(row or "").strip()
                if text and text.lower() != "none":
                    helpful_points.append(text)

        rejected_points = []
        for item, entry in list(candidate_summaries.items())[:6]:
            entry = dict(entry or {})
            text = str(entry.get("risk_summary", "") or "").strip()
            if text:
                rejected_points.append(f"{item}: {text}")
        for conflict in list(interaction_summary.get("main_disagreements", []) or [])[:4]:
            text = str(conflict or "").strip()
            if text:
                rejected_points.append(text)
        for correction in list(interaction_summary.get("corrections_or_rebuttals", []) or [])[:4]:
            text = str(correction or "").strip()
            if text:
                rejected_points.append(text)

        remaining_uncertainty = []
        if advisor_pool_empty:
            remaining_uncertainty.append("advisor_pool_empty")
        for item in list(evidence_summary.get("silent_focus_candidates", []) or []):
            remaining_uncertainty.append(f"missing_advisor_evidence:{item}")
        if protocol_issues:
            remaining_uncertainty.append("communication_protocol")
        for item in list(synthesis_packet.get("remaining_uncertainty", []) or [])[:8]:
            text = str(item or "").strip()
            if text:
                remaining_uncertainty.append(text)

        discussed_items = []
        for item, entry in candidate_summaries.items():
            entry = dict(entry or {})
            if any(entry.get(bucket) for bucket in ["support_summary", "risk_summary", "tradeoff_summary", "key_evidence"]):
                discussed_items.append(str(item))

        candidate_comparison = {
            "decision_policy": "information_only_no_vote",
            "source": "advisor_summary_agent_v1",
            "candidate_summaries": candidate_summaries,
            "task_specific_summary": task_summary,
            "extra_task_summary": dict(synthesis_packet.get("extra_task_summary", {}) or {}),
            "interaction_summary": interaction_summary,
            "extra_interaction_summary": dict(synthesis_packet.get("extra_interaction_summary", {}) or {}),
            "unresolved_questions": list(synthesis_packet.get("remaining_uncertainty", []) or []),
        }
        advisor_type_reliability = {}
        for row in list(committee.get("advisor_arguments", []) or []):
            advisor_type_reliability[str(row.get("advisor", ""))] = {
                "stance": str(row.get("stance", "")),
                "endorsed_item": str(row.get("endorsed_item", "")),
            }

        out_packet = dict(synthesis_packet or {})
        out_packet["decision_policy"] = "information_only_no_vote"
        out_packet["source"] = str(out_packet.get("source", "") or "advisor_summary_agent_v1")
        out_packet["advisor_synthesis_packet"] = dict(synthesis_packet or {})
        out_packet["evidence_summary"] = evidence_summary
        out_packet["protocol_issues"] = protocol_issues
        out_packet["advisor_pool_empty"] = advisor_pool_empty

        return build_absorbed_memory(
            helpful_points=helpful_points,
            rejected_points=rejected_points,
            candidate_comparison=candidate_comparison,
            alternative_items=discussed_items,
            advisor_reliability_observation=advisor_type_reliability,
            remaining_uncertainty=remaining_uncertainty,
            evidence_packet=out_packet,
        ) | {
            "accepted_points": list(helpful_points),
            "silent_or_missing_evidence": list(evidence_summary.get("silent_focus_candidates", []) or []),
            "feedback_to_advisors_seed": list(synthesis_packet.get("remaining_uncertainty", []) or [])[:4],
        }
