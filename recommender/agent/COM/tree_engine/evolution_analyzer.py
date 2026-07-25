import hashlib


class EvolutionAnalyzer:
    """Deterministic analyzer for diagnosis-driven skill evolution.

    The analyzer intentionally does not update any skill. It converts one
    interaction trace into structured diagnoses for the per-user skill evolver
    and the public tree batch evolver.
    """

    @staticmethod
    def _norm(value):
        return " ".join(str(value or "").strip().lower().split())

    @staticmethod
    def _short(value, limit=320):
        text = " ".join(str(value or "").split())
        return text[:limit]

    @staticmethod
    def _path_key(path, sep=" -> "):
        parts = [
            str((path or {}).get('why', "") or ""),
            str((path or {}).get("what", "") or ""),
            str((path or {}).get("who", "") or ""),
            str((path or {}).get("how", "") or ""),
        ]
        return sep.join([part for part in parts if part])

    @staticmethod
    def _dominant(values):
        counts = {}
        for value in values or []:
            text = str(value or "").strip()
            if not text or text == "none":
                continue
            counts[text] = counts.get(text, 0) + 1
        if not counts:
            return ""
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def _advisor_trust_breakdown(self, trace):
        packet = dict(trace.get("execution_packet") or {})
        profiles = list(packet.get("advisor_profiles", []) or [])
        rows = []
        for profile in profiles:
            if str(profile.get("advisor_type", "") or "") != "trusted-advisors":
                continue
            rows.append(
                {
                    "advisor_id": str(profile.get("u_raw", "") or profile.get("advisor_id", "") or ""),
                    "trust_relation": str(profile.get("trust_relation", "") or "none"),
                    "trust_scope": str(profile.get("trust_scope", "") or "none"),
                    "history_similarity_bucket": str(profile.get("history_similarity_bucket", "") or "none"),
                    "trust_subbranch": str(profile.get("trust_subbranch", "") or "none"),
                    "reliability": float(profile.get("reliability", 0.0) or 0.0),
                    "sim": float(profile.get("sim", 0.0) or 0.0),
                    "experience_score": float(profile.get("experience_score", 0.0) or 0.0),
                }
            )
        return rows

    def _fine_path(self, path, trust_breakdown):
        out = dict(path or {})
        if str(out.get("who", "") or "") == "trusted-advisors":
            dominant = self._dominant([row.get("trust_subbranch", "") for row in trust_breakdown])
            if dominant:
                out["who_subbranch"] = dominant
                out["who_fine"] = f"trusted-advisors[{dominant}]"
        return out

    @staticmethod
    def _path_key_fine(path, sep=" -> "):
        who = str((path or {}).get("who_fine", "") or (path or {}).get("who", "") or "")
        parts = [
            str((path or {}).get('why', "") or ""),
            str((path or {}).get("what", "") or ""),
            who,
            str((path or {}).get("how", "") or ""),
        ]
        return sep.join([part for part in parts if part])

    @staticmethod
    def _prefixes(path):
        parts = [
            str((path or {}).get('why', "") or ""),
            str((path or {}).get("what", "") or ""),
            str((path or {}).get("who", "") or ""),
            str((path or {}).get("how", "") or ""),
        ]
        out = []
        for idx in range(1, len(parts) + 1):
            if all(parts[:idx]):
                out.append(" -> ".join(parts[:idx]))
        return out

    def _prefixes_fine(self, path):
        parts = [
            str((path or {}).get('why', "") or ""),
            str((path or {}).get("what", "") or ""),
            str((path or {}).get("who_fine", "") or (path or {}).get("who", "") or ""),
            str((path or {}).get("how", "") or ""),
        ]
        out = []
        for idx in range(1, len(parts) + 1):
            if all(parts[:idx]):
                out.append(" -> ".join(parts[:idx]))
        return out

    def _contains_target(self, values, target_names):
        targets = {self._norm(x) for x in (target_names or []) if self._norm(x)}
        if not targets:
            return False
        for value in values or []:
            norm = self._norm(value)
            if norm in targets:
                return True
        return False

    def _candidate_contains_target(self, trace, target_names):
        if "candidate_target_overlap" in trace:
            return bool(trace.get("candidate_target_overlap"))
        return self._contains_target(trace.get("candidate_item_names", []) or [], target_names)

    def _advisor_mentions_target(self, execution_packet, absorbed_memory, target_names):
        return bool(
            self._advisor_feedback_mentions_target(execution_packet, target_names)
            or self._absorbed_memory_mentions_target(absorbed_memory, target_names)
        )

    def _target_signal_from_text(self, text):
        text = self._norm(text)
        if not text:
            return {"positive": False, "negative": False}
        negative_markers = [
            "remove",
            "removed",
            "removal",
            "down rank",
            "down ranked",
            "downrank",
            "down-ranked",
            "risk",
            "weak",
            "weaker",
            "unsuitable",
            "not suitable",
            "surface match",
            "only surface",
            "low interest",
            "not promising",
            "mismatch",
            "lacks",
            "duplicate",
            "duplicates",
            "out of place",
            "challenged",
            "warning",
        ]
        positive_markers = [
            "support",
            "supported",
            "strong",
            "stronger",
            "promising",
            "high interest",
            "stay",
            "keep",
            "retain",
            "retained",
            "interested",
            "matches",
            "aligns",
            "fits",
            "consistent",
            "preserve",
            "plausible",
        ]
        return {
            "positive": any(marker in text for marker in positive_markers),
            "negative": any(marker in text for marker in negative_markers),
        }

    def _candidate_summary_target_signal(self, candidate_summaries, target_names):
        targets = {self._norm(x) for x in (target_names or []) if self._norm(x)}
        if not targets:
            return {"mentioned": False, "positive": False, "negative": False}
        result = {"mentioned": False, "positive": False, "negative": False}
        for name, summary in dict(candidate_summaries or {}).items():
            if self._norm(name) not in targets:
                continue
            result["mentioned"] = True
            summary = dict(summary or {}) if isinstance(summary, dict) else {"summary": summary}
            support = " ".join(
                str(summary.get(key, "") or "")
                for key in ["support_summary", "tradeoff_summary", "summary", "reason"]
            )
            risk = " ".join(
                str(summary.get(key, "") or "")
                for key in ["risk_summary", "warning_summary", "against_summary"]
            )
            key_evidence = " ".join(str(x) for x in list(summary.get("key_evidence", []) or []))
            agreement = self._norm(summary.get("advisor_agreement", ""))
            support_signal = self._target_signal_from_text(" ".join([support, key_evidence]))
            risk_signal = self._target_signal_from_text(risk)
            if support.strip() and not risk_signal["negative"]:
                result["positive"] = True
            if support_signal["positive"]:
                result["positive"] = True
            if risk.strip() or risk_signal["negative"] or support_signal["negative"]:
                result["negative"] = True
        return result

    def _by_candidate_target_signal(self, by_candidate, target_names):
        targets = {self._norm(x) for x in (target_names or []) if self._norm(x)}
        if not targets:
            return {"mentioned": False, "positive": False, "negative": False}
        result = {"mentioned": False, "positive": False, "negative": False}
        for name, row in dict(by_candidate or {}).items():
            if self._norm(name) not in targets:
                continue
            result["mentioned"] = True
            row = dict(row or {}) if isinstance(row, dict) else {}
            support = list(row.get("support", []) or [])
            against = list(row.get("against", []) or [])
            defended = int(row.get("defended_count", 0) or 0)
            attacked = int(row.get("attacked_count", 0) or 0)
            if support or defended > 0:
                result["positive"] = True
            if against or attacked > 0:
                result["negative"] = True
            support_text = " ".join(str((x or {}).get("reason", "") if isinstance(x, dict) else x) for x in support)
            against_text = " ".join(str((x or {}).get("reason", "") if isinstance(x, dict) else x) for x in against)
            support_signal = self._target_signal_from_text(support_text)
            against_signal = self._target_signal_from_text(against_text)
            if support_signal["positive"]:
                result["positive"] = True
            if support_signal["negative"] or against_signal["negative"]:
                result["negative"] = True
        return result

    def _advisor_feedback_target_signal(self, execution_packet, target_names):
        targets = {self._norm(x) for x in (target_names or []) if self._norm(x)}
        if not targets:
            return {"mentioned": False, "positive": False, "negative": False}
        result = {"mentioned": False, "positive": False, "negative": False}
        for fb in list((execution_packet or {}).get("advisor_feedbacks", []) or []):
            fb = dict(fb or {})
            target_fields = []
            for key in ["endorsed_item", "defended_item", "suggested_item", "recommended_item"]:
                if self._norm(fb.get(key, "")) in targets:
                    result["mentioned"] = True
                    result["positive"] = True
            for key in ["attacked_item", "challenged_item"]:
                if self._norm(fb.get(key, "")) in targets:
                    result["mentioned"] = True
                    result["negative"] = True
            for key in ["task_answer", "support_reason", "oppose_reason", "raw_text"]:
                value = str(fb.get(key, "") or "")
                if self._contains_target([value], target_names):
                    result["mentioned"] = True
                    target_fields.append(value)
            for view in list(fb.get("candidate_views", []) or []):
                if not isinstance(view, dict):
                    continue
                if self._norm(view.get("candidate", "")) not in targets:
                    continue
                result["mentioned"] = True
                view_text = " ".join(str(view.get(key, "") or "") for key in ["view", "reason"])
                signal = self._target_signal_from_text(view_text)
                bucket = self._norm(view.get("view", ""))
                if bucket in {"support", "stay", "keep", "retain", "interested"} or signal["positive"]:
                    result["positive"] = True
                if bucket in {"risk", "remove", "weak", "weaker", "downrank"} or signal["negative"]:
                    result["negative"] = True
            for text in target_fields:
                signal = self._target_signal_from_text(text)
                if signal["positive"]:
                    result["positive"] = True
                if signal["negative"]:
                    result["negative"] = True
        return result

    def _advisor_feedback_mentions_target(self, execution_packet, target_names):
        feedbacks = list((execution_packet or {}).get("advisor_feedbacks", []) or [])
        endorsed = [fb.get("endorsed_item", "") for fb in feedbacks]
        evidence_values = []
        for fb in feedbacks:
            for key in [
                "defended_item",
                "challenged_item",
                "suggested_item",
                "alternative_item",
                "recommended_item",
                "task_answer",
                "support_reason",
                "oppose_reason",
                "raw_text",
            ]:
                value = (fb or {}).get(key, "")
                if isinstance(value, list):
                    evidence_values.extend(value)
                else:
                    evidence_values.append(value)
            for view in list((fb or {}).get("candidate_views", []) or []):
                if isinstance(view, dict):
                    evidence_values.append(view.get("candidate", ""))
                    evidence_values.append(view.get("reason", ""))
        return self._contains_target(endorsed + evidence_values, target_names)

    def _absorbed_memory_mentions_target(self, absorbed_memory, target_names):
        pkt = dict((absorbed_memory or {}).get("evidence_packet", {}) or {})
        evidence_values = []
        synthesis = dict(pkt.get("advisor_synthesis_packet", {}) or pkt or {})
        evidence_values.extend(list((dict(synthesis.get("candidate_summaries", {}) or {})).keys()))
        evidence_values.extend(list((dict(synthesis.get("by_candidate", {}) or {})).keys()))
        for key in ["alternative_items", "alternatives"]:
            raw = pkt.get(key, [])
            if isinstance(raw, list):
                evidence_values.extend(raw)
            else:
                evidence_values.append(raw)
        evidence_values.extend((absorbed_memory or {}).get("alternative_items", []) or [])
        evidence_values.append(pkt.get("candidate_evidence_text", ""))
        evidence_values.append(synthesis.get("what_was_answered", ""))
        return self._contains_target(evidence_values, target_names)

    def _absorbed_memory_target_signal(self, absorbed_memory, target_names):
        pkt = dict((absorbed_memory or {}).get("evidence_packet", {}) or {})
        synthesis = dict(pkt.get("advisor_synthesis_packet", {}) or pkt or {})
        candidate_signal = self._candidate_summary_target_signal(synthesis.get("candidate_summaries", {}) or {}, target_names)
        by_candidate_signal = self._by_candidate_target_signal(
            synthesis.get("by_candidate", {}) or pkt.get("by_candidate", {}) or {},
            target_names,
        )
        result = {
            "mentioned": bool(candidate_signal["mentioned"] or by_candidate_signal["mentioned"]),
            "positive": bool(candidate_signal["positive"] or by_candidate_signal["positive"]),
            "negative": bool(candidate_signal["negative"] or by_candidate_signal["negative"]),
        }
        text_fields = [
            synthesis.get("what_was_answered", ""),
            pkt.get("candidate_evidence_text", ""),
        ]
        for key in ["alternative_items", "alternatives", "retained_candidates", "interested_set", "keep_set"]:
            raw = pkt.get(key, []) or synthesis.get(key, [])
            if isinstance(raw, list):
                if self._contains_target(raw, target_names):
                    result["mentioned"] = True
                    result["positive"] = True
            elif self._contains_target([raw], target_names):
                result["mentioned"] = True
                result["positive"] = True
        for text in text_fields:
            if self._contains_target([text], target_names):
                result["mentioned"] = True
                signal = self._target_signal_from_text(text)
                if signal["positive"]:
                    result["positive"] = True
                if signal["negative"]:
                    result["negative"] = True
        return result

    def _failure_attribution(self, trace, failure_level, communication_quality_issues):
        evaluation = dict(trace.get("evaluation_result") or {})
        outcome = str(evaluation.get("outcome_signal", "") or "")
        if outcome not in ["TW", "WW"]:
            return "success"
        target_names = list(trace.get("target_item_names") or [])
        candidate_has_target = self._candidate_contains_target(trace, target_names)
        if not candidate_has_target:
            return "candidate_or_data_defect"
        shortlist = list((trace.get("decision_state") or {}).get("shortlist", []) or [])
        focus_target = bool(trace.get("focus_target_overlap", False) or evaluation.get("focus_target_overlap", False))
        if not focus_target and not self._contains_target(shortlist, target_names):
            return "item_selection_defect"

        execution_packet = dict(trace.get("execution_packet") or {})
        absorbed_memory = dict(trace.get("absorbed_memory") or {})
        advisor_delivered = self._advisor_feedback_mentions_target(execution_packet, target_names)
        summary_preserved = self._absorbed_memory_mentions_target(absorbed_memory, target_names)
        advisor_target_signal = self._advisor_feedback_target_signal(execution_packet, target_names)
        summary_target_signal = self._absorbed_memory_target_signal(absorbed_memory, target_names)
        target_positive_preserved = bool(
            summary_target_signal.get("positive", False)
            and not summary_target_signal.get("negative", False)
        )
        advisor_positive_delivered = bool(
            advisor_target_signal.get("positive", False)
            and not advisor_target_signal.get("negative", False)
        )
        issues = {str(x or "") for x in list(communication_quality_issues or [])}
        issue_text = " ".join(sorted(issues)).lower()

        if advisor_delivered and not summary_preserved:
            return "aggregation_or_parser_defect"
        if (
            target_positive_preserved
            and failure_level in ["feedback_absorption", "redecision"] + (["communication_protocol"] if outcome == "TW" else [])
        ):
            return "user_absorption_failure"
        if advisor_positive_delivered and target_positive_preserved:
            return "user_absorption_failure"
        if any(
            marker in issue_text
            for marker in [
                "advisor_pool_empty",
                "multi_candidate_protocol",
                "protocol",
                "focus_shortlist_truncated",
                "unmapped_what_task",
            ]
        ):
            return "tree_defect"
        if failure_level in ["path_selection", "advisor_feedback", "communication_protocol"]:
            return "tree_defect"
        if failure_level in ["shortlist_construction", "initial_proposal"]:
            return "item_selection_defect"
        return "tree_defect"

    def _advisor_summary(self, execution_packet):
        rows = []
        for fb in list((execution_packet or {}).get("advisor_feedbacks", []) or [])[:6]:
            rows.append(
                {
                    "advisor_id": str(fb.get("advisor_id", "") or ""),
                    "advisor_type": str(fb.get("advisor_type", "") or ""),
                    "stance": str(fb.get("stance", "") or ""),
                    "endorsed_item": str(fb.get("endorsed_item", "") or ""),
                    "rationale_summary": self._short(fb.get("support_reason") or fb.get("oppose_reason") or fb.get("raw_text")),
                }
            )
        return rows

    def _communication_quality_issues(self, trace):
        issues = []
        decision_state = dict(trace.get("decision_state") or {})
        shortlist = [self._norm(x) for x in (decision_state.get("shortlist", []) or []) if self._norm(x)]
        execution_packet = dict(trace.get("execution_packet") or {})
        focus = [self._norm(x) for x in (execution_packet.get("focus_candidates", []) or []) if self._norm(x)]
        if shortlist and focus:
            missing = [x for x in shortlist if x not in set(focus)]
            if missing:
                issues.append("focus_shortlist_truncated")

        committee = dict(execution_packet.get("committee_packet") or {})
        for row in list(committee.get("protocol_issues", []) or []):
            if isinstance(row, dict):
                issue = str(row.get("issue", "") or "").strip()
            else:
                issue = str(row or "").strip()
            if issue:
                issues.append(issue.split(":", 1)[0])

        for fb in list(execution_packet.get("advisor_feedbacks", []) or []):
            if not isinstance(fb, dict):
                continue
            for issue in list(fb.get("protocol_issues", []) or []):
                issue = str(issue or "").strip()
                if issue:
                    issues.append(issue.split(":", 1)[0])

        path = dict(trace.get("path") or {})
        if bool(((trace.get("execution_packet") or {}).get("committee_packet") or {}).get("final_advisor_pool_empty", False)):
            issues.append("advisor_pool_empty")
        if str(path.get("how", "") or "") in ["competitive", "multi-candidate-debate", "multi-competitive", "multi-competitive-warning", "multi-competitive-promotion"]:
            feedbacks = list(execution_packet.get("advisor_feedbacks", []) or [])
            defended = {self._norm(fb.get("defended_item", "")) for fb in feedbacks if self._norm(fb.get("defended_item", ""))}
            if len(feedbacks) >= 2 and len(defended) <= 1:
                issues.append("multi_candidate_protocol_not_enforced")

        return sorted(set(issues))

    def _infer_failure_level(self, trace):
        outcome = str((trace.get("evaluation_result") or {}).get("outcome_signal", "") or "")
        path = dict(trace.get("path") or {})
        target_names = list(trace.get("target_item_names") or [])
        candidate_has_target = self._candidate_contains_target(trace, target_names)
        shortlist = list((trace.get("decision_state") or {}).get("shortlist", []) or [])
        execution_packet = dict(trace.get("execution_packet") or {})
        absorbed_memory = dict(trace.get("absorbed_memory") or {})
        final_name = str(trace.get("final_item_name", "") or "")
        proposal_name = str(trace.get("proposal_item_name", "") or "")
        path_who = str(path.get("who", "") or "")
        quality_issues = self._communication_quality_issues(trace)

        if outcome in ["WW", "TW"] and not candidate_has_target:
            return "candidate_generation_or_data_alignment"

        if outcome == "TW":
            if quality_issues:
                return "communication_protocol"
            if path_who in ["none", "skip", ""]:
                return "redecision"
            if self._advisor_mentions_target(execution_packet, absorbed_memory, target_names):
                return "feedback_absorption"
            return "advisor_feedback"

        if outcome == "WW":
            if quality_issues and self._contains_target(shortlist, target_names):
                return "communication_protocol"
            if not self._contains_target(shortlist, target_names):
                return "shortlist_construction"
            if path_who in ["none", "skip", ""]:
                return "initial_proposal" if proposal_name == final_name else "redecision"
            if not self._advisor_mentions_target(execution_packet, absorbed_memory, target_names):
                return "advisor_feedback"
            return "redecision"

        return "success"

    def _failure_problem(self, level, path):
        who = str((path or {}).get("who", "") or "")
        how = str((path or {}).get("how", "") or "")
        if level == "candidate_generation_or_data_alignment":
            return "The correct item is absent from the candidate set for this interaction, so user reasoning and communication cannot recover it."
        if level == "shortlist_construction":
            return "The correct item was available in the candidate set but was not preserved as a focused alternative."
        if level == "initial_proposal":
            return "The initial reasoning selected a plausible wrong item before comparing it with the target-like candidate."
        if level == "advisor_feedback":
            return f"The {who or 'selected'} advisor source did not provide enough item-specific evidence for the correct item."
        if level == "feedback_absorption":
            return "Advisor evidence contained useful target-like information, but the absorbed memory did not preserve it strongly enough."
        if level == "communication_protocol":
            return "The communication protocol produced inconsistent advisor arguments, out-of-focus candidates, missing challenge-answer structure, or non-discriminative multi-candidate evidence."
        if level == "path_selection":
            return "The selected communication path was not suitable for the current candidate conflict."
        if level == "redecision":
            return "The final arbitration selected the wrong item even though the correct item was visible in the evidence space."
        return f"The {how or 'communication'} path failed to separate the correct item from the wrong item."

    def _user_rule(self, success, level, path, outcome):
        who = str((path or {}).get("who", "") or "")
        how = str((path or {}).get("how", "") or "")
        if success:
            if outcome == "WT":
                return 'Why communication corrects an initially wrong decision, reuse the same evidence-checking pattern before finalizing similar candidate conflicts.'
            return 'Why the initial proposal is correct, preserve the historical item-level evidence that justified it and avoid unnecessary switching.'
        if level == "candidate_generation_or_data_alignment":
            return "Do not evolve user reasoning from this round; first repair candidate generation so the correct item is visible to the skill."
        if level == "shortlist_construction":
            return 'Why candidate scores or explanations are close, keep the historically aligned candidate in the shortlist instead of narrowing too early.'
        if level == "initial_proposal":
            return "Before accepting the first plausible candidate, compare it with the candidate that has stronger repeated item-level support in the user's history."
        if level == "advisor_feedback":
            return f"When {who or 'advisor'} feedback is broad or misses the historically aligned candidate, require item-specific counter-evidence before trusting it."
        if level == "feedback_absorption":
            return "If advisor evidence mentions a historically aligned candidate, preserve it as a leading alternative before redecision."
        if level == "communication_protocol":
            return 'Why advisor claims, challenges, answers, endorsed items, and reasons conflict, distrust the communication result and require a constrained summary over focus candidates.'
        if level == "path_selection":
            return f"For close item-level conflicts, avoid relying on {who or 'the selected advisor source'} with {how or 'the selected protocol'} unless it produces direct candidate comparison."
        return "Before finalizing, directly compare the final proposal with any candidate that has stronger historical or advisor-supported evidence."

    def _shortlist_counterfactual_rule(self, trace):
        target_names = [str(x) for x in trace.get("target_item_names", []) or [] if str(x or "").strip()]
        decision_state = dict(trace.get("decision_state") or {})
        shortlist = [str(x) for x in (decision_state.get("shortlist", []) or []) if str(x or "").strip()]
        candidate_evidence = list(decision_state.get("candidate_evidence", []) or [])
        target = target_names[0] if target_names else "the held-out correct item"
        shortlist_count = len(shortlist)
        proposal_reason = self._short(decision_state.get("proposal_reason", ""))

        proposal_evidence = []
        proposal_name = str(trace.get("proposal_item_name", "") or "").strip()
        for row in candidate_evidence:
            if proposal_name and self._norm((row or {}).get("candidate", "")) == self._norm(proposal_name):
                fit = self._short((row or {}).get("fit", "") or "")
                reason = self._short((row or {}).get("reason", "") or row)
                if fit or reason:
                    proposal_evidence.append("; ".join([x for x in [fit, reason] if x]))

        target_evidence = []
        target_style_bits = []
        for row in candidate_evidence:
            if self._norm((row or {}).get("candidate", "")) == self._norm(target):
                fit = self._short((row or {}).get("fit", "") or "")
                reason = self._short((row or {}).get("reason", "") or row)
                if fit:
                    target_style_bits.append(fit)
                if reason:
                    target_evidence.append(reason)

        target_style = "; ".join([x for x in target_style_bits[:2] if x]).strip()
        if not target_style:
            target_style = "the target item's represented preference cluster, style/category, topic, language/cultural signal, or usage/reading pattern"

        evidence_clause = (
            f" Target-style evidence seen this round: {'; '.join(target_evidence[:2])}."
            if target_evidence
            else " No extracted target-style evidence row was produced, so missing evidence must trigger style-level comparison rather than negative judgment."
        )
        proposal_clause = (
            f" The user selected the proposal because these signals were salient: {'; '.join(proposal_evidence[:2])}."
            if proposal_evidence
            else f" The user's proposal rationale was: {proposal_reason}."
            if proposal_reason
            else " First diagnose which history, prior, or easy-to-explain bridge made the proposal salient."
        )
        return (
            "Proposal-first omission reflection: first explain why the user's current favorite became salient, then diagnose why that "
            f"reasoning path kept {shortlist_count} hesitation candidates but omitted a supervised target-style candidate. "
            f"{proposal_clause} Generalize the omitted target evidence as: {target_style}. "
            "Future updates should preserve candidates carrying this transferable signal when it connects to weak history signals, "
            "minority clusters, recent drift, or co-occurrence neighbors, especially when the proposal was selected mainly because of "
            "a dominant cluster, prior hint, or easier-to-explain bridge."
            f"{evidence_clause}"
        )

    def _shortlist_anti_rule(self, trace):
        return (
            "Risky narrowing pattern exposed: do not remove candidates that represent a historically plausible but less obvious taste cluster "
            "merely because another candidate has broader, more familiar, or easier-to-explain genre/profile evidence. Exclusion requires "
            "explicit style-level counter-evidence, not just stronger prose for the current favorite."
        )

    def _extra_user_rule_updates(self, success, failure_level, trace):
        if success or failure_level != "shortlist_construction":
            return []
        return [
            {
                "target_layer": "item_selection_skill",
                "operation": "discover",
                "problem": "The correct item was available but excluded from both first choice and uncertainty set; abstract the missed item into a reusable taste-cluster/style signal.",
                "rule": self._shortlist_counterfactual_rule(trace),
                "confidence": 0.48,
            },
            {
                "target_layer": "item_selection_skill",
                "operation": "weaken",
                "problem": "The user's item-selection reasoning narrowed too early based on easier-to-explain evidence for the wrong favorite.",
                "rule": self._shortlist_anti_rule(trace),
                "confidence": 0.40,
            },
        ]

    def analyze(self, trace):
        trace = dict(trace or {})
        evaluation = dict(trace.get("evaluation_result") or {})
        path = dict(trace.get("path") or {})
        outcome = str(evaluation.get("outcome_signal", "") or "")
        stage1_only = bool(trace.get("stage1_only", False) or evaluation.get("stage1_only", False))
        stage1_shortlist_success = stage1_only and bool(evaluation.get("focus_target_overlap", False))
        success = outcome in ["TT", "WT"] or bool(stage1_shortlist_success)
        failure_level = "success" if success else self._infer_failure_level(trace)
        communication_quality_issues = self._communication_quality_issues(trace)
        failure_attribution = "success" if success else self._failure_attribution(
            trace,
            failure_level=failure_level,
            communication_quality_issues=communication_quality_issues,
        )
        target_layer = "communication_selection_skill" if failure_level in ["path_selection", "advisor_feedback", "feedback_absorption", "communication_protocol"] else "item_selection_skill"
        if failure_attribution == "user_absorption_failure":
            target_layer = "communication_absorption_skill"
        if success and outcome == "WT":
            target_layer = "communication_selection_skill"
        if success and outcome == "TT":
            target_layer = "item_selection_skill"

        diagnosis_id_src = "|".join(
            [
                str(trace.get("user_id", "")),
                str(trace.get("round_id", "")),
                self._path_key(path),
                outcome,
                str(trace.get("final_item_name", "")),
            ]
        )
        diagnosis_id = hashlib.md5(diagnosis_id_src.encode("utf-8")).hexdigest()[:16]
        operation = "reinforce" if success else "discover"
        confidence = 0.55 if success else 0.42
        if stage1_shortlist_success and outcome not in ["TT", "WT"]:
            problem = "Stage1-only training: the target was not the first choice but was preserved in the hesitation set."
            rule = 'Why the user preserves a supervised target-style candidate in the hesitation set, reinforce the transferable preference signal that made it worth keeping, but do not treat other unchosen candidates as dislikes.'
        else:
            problem = "The current interaction validates the user's existing reasoning skill." if success else self._failure_problem(failure_level, path)
            rule = self._user_rule(success=success, level=failure_level, path=path, outcome=outcome)
        extra_user_updates = self._extra_user_rule_updates(success=success, failure_level=failure_level, trace=trace)

        tree_relevant = failure_attribution == "tree_defect"
        tree_operation = "reinforce_branch" if success else ("weaken_branch" if tree_relevant else "record_only")
        if failure_level == "candidate_generation_or_data_alignment":
            operation = "record_only"
            confidence = 0.0
        if failure_attribution in ["candidate_or_data_defect", "aggregation_or_parser_defect"]:
            operation = "record_only"
            confidence = 0.0
        if failure_attribution == "user_absorption_failure":
            operation = "discover"
            confidence = max(confidence, 0.50)
        suggested_new_branch = ""
        if (not success) and tree_relevant and str(path.get("how", "")) in ["competitive", "debate", "pairwise-debate", "multi-candidate-debate", "multi-competitive", "multi-competitive-warning", "multi-competitive-promotion"]:
            if failure_level in ["advisor_feedback", "feedback_absorption", "communication_protocol"]:
                suggested_new_branch = "counterfactual-comparison"
                tree_operation = "split_branch" if str(path.get("how", "")) in ["competitive", "multi-candidate-debate", "multi-competitive", "multi-competitive-warning", "multi-competitive-promotion"] else "grow_branch"
        trust_breakdown = self._advisor_trust_breakdown(trace)
        fine_path = self._fine_path(path, trust_breakdown)
        dominant_trust_subbranch = str(fine_path.get("who_subbranch", "") or "")
        if dominant_trust_subbranch and tree_relevant and not suggested_new_branch:
            suggested_new_branch = f"trusted-advisors/{dominant_trust_subbranch}"

        interaction_trace = {
            "diagnosis_id": diagnosis_id,
            "user_id": str(trace.get("user_id", "") or ""),
            "stage": str(trace.get("stage", "train") or "train"),
            "candidate_items": [str(x) for x in trace.get("candidate_item_names", []) or []],
            "target_item": [str(x) for x in trace.get("target_item_names", []) or []],
            "candidate_target_overlap": bool(self._candidate_contains_target(trace, trace.get("target_item_names", []) or [])),
            "focus_target_overlap": bool(trace.get("focus_target_overlap", False)),
            "missing_target_item_names": [str(x) for x in trace.get("missing_target_item_names", []) or []],
            "selected_item": str(trace.get("final_item_name", "") or ""),
            "success": bool(success),
            "initial_selected_item": str(trace.get("proposal_item_name", "") or ""),
            "final_selected_item": str(trace.get("final_item_name", "") or ""),
            "communication_path": dict(path),
            "communication_fine_path": dict(fine_path),
            "advisor_trust_breakdown": list(trust_breakdown),
            "initial_user_reason": self._short((trace.get("decision_state") or {}).get("proposal_reason", "")),
            "advisor_feedback_summary": self._advisor_summary(trace.get("execution_packet") or {}),
            "final_user_reason": self._short((trace.get("redecision_packet") or {}).get("revised_reason", "")),
            "communication_quality_issues": list(communication_quality_issues),
        }

        return {
            "diagnosis_id": diagnosis_id,
            "success": bool(success),
            "outcome_signal": outcome,
            "primary_failure_level": failure_level,
            "failure_attribution": failure_attribution,
            "interaction_trace": interaction_trace,
            "user_skill_diagnosis": {
                "target_layer": target_layer,
                "operation": operation,
                "problem": problem,
                "rule": rule,
                "confidence": confidence,
                "additional_updates": extra_user_updates,
            },
            "tree_diagnosis": {
                "user_id": str(trace.get("user_id", "") or ""),
                "success": bool(success),
                "outcome_signal": outcome,
                "path": dict(path),
                "fine_path": dict(fine_path),
                "path_key": self._path_key(path),
                "fine_path_key": self._path_key_fine(fine_path),
                "path_prefixes": self._prefixes(path),
                "fine_path_prefixes": self._prefixes_fine(fine_path),
                "who_subbranch": dominant_trust_subbranch,
                "advisor_trust_breakdown": list(trust_breakdown),
                "failed_level": failure_level if not success else "",
                "failure_attribution": failure_attribution,
                "tree_relevance": "high" if tree_relevant else ("positive" if success else "low"),
                "failure_reason": "" if success else problem,
                "communication_quality_issues": list(communication_quality_issues),
                "suggested_operation": tree_operation,
                "suggested_new_branch": suggested_new_branch,
                "diagnosis_id": diagnosis_id,
            },
        }
