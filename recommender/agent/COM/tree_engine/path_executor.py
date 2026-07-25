from collections import defaultdict
import copy
import json
import re
from pathlib import Path

from recommender.agent.COM.tree_engine.schemas import build_advisor_feedback
from recommender.agent.COM.tree_engine.public_tree import (
    infer_communication_family,
    infer_communication_intent,
    infer_communication_shape,
)
from recommender.agent.COM.tree_engine.task_planner import output_format_for_what
from recommender.agent.COM.utils.com_agent import (
    _ensure_runtime_args,
    get_last_llm_request_usage,
    llm_request,
    update_llm_prompt_trace_context,
)


class PathExecutor:
    def __init__(self, args):
        self.args = args

    def _advisor_discussion_rounds(self, shape, path=None):
        if shape != "multi":
            return 1
        return max(1, int(getattr(self.args, "com_advisor_discussion_rounds", 1) or 1))

    @staticmethod
    def _canonical_how(how, focus_count=0):
        how = str(how or "").strip()
        aliases = {
            "single": "single-advisor",
            "single-warning": "single-advisor",
            "single-promotion": "single-advisor",
            "single-evaluation-only": "single-advisor",
            "single-evaluation-warning": "single-advisor",
            "single-evaluation-promotion": "single-advisor",
            "cooperative": "multi-cooperative",
            "cooperative-inquiry": "multi-cooperative",
            "multi-cooperative-warning": "multi-cooperative",
            "multi-cooperative-promotion": "multi-cooperative",
            "feedback-cooperative-repair": "multi-cooperative",
            "feedback-focused-repair": "multi-cooperative",
            "competitive": "multi-competitive",
            "debate": "multi-competitive",
            "pairwise-debate": "multi-competitive",
            "multi-candidate-debate": "multi-competitive",
            "multi-competitive-warning": "multi-competitive",
            "multi-competitive-promotion": "multi-competitive",
            "feedback-competitive-repair": "multi-competitive",
        }
        if how in aliases:
            return aliases[how]
        return how

    @staticmethod
    def _base_node_id(node_id):
        node_id = str(node_id or "").strip()
        return node_id.split("/", 1)[0] if "/" in node_id else node_id

    @staticmethod
    def _output_field_names_from_lines(lines):
        fields = []
        seen = set()
        for raw in list(lines or []):
            text = re.sub(r"^\s*[-*]\s*", "", str(raw or "").strip())
            if not text or text.startswith("<"):
                continue
            label = text.split(":", 1)[0].strip()
            if "|" in label:
                label = label.split("|", 1)[0].strip()
            label = re.sub(r"\s+", "", label)
            if not re.match(r"^[A-Za-z][A-Za-z0-9_]{1,48}$", label):
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            fields.append(label)
        return fields

    @staticmethod
    def _summary_hint_field_name(field):
        text = re.sub(r"^\s*[-*]\s*", "", str(field or "").strip())
        if not text:
            return ""
        text = text.split(":", 1)[0].strip()
        text = text.split("<", 1)[0].strip()
        text = text.split("|", 1)[0].strip()
        text = text.split("(", 1)[0].strip()
        text = re.split(r"\s+-\s+", text, maxsplit=1)[0].strip()
        text = re.sub(r"\s+", "", text)
        aliases = {
            "response_to_previous": "ResponseToPrevious",
            "responsetoprevious": "ResponseToPrevious",
            "challenge_or_support_previous": "ChallengeOrSupportPrevious",
            "challengeorsupportprevious": "ChallengeOrSupportPrevious",
            "correction": "Correction",
            "percandidatevalidation": "PerCandidateValidation",
            "per_candidate_validation": "PerCandidateValidation",
        }
        key = re.sub(r"[^A-Za-z0-9_]", "", text).lower()
        if key in aliases:
            return aliases[key]
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]{1,48}$", text):
            return ""
        return text

    @classmethod
    def _clean_summary_hint_fields(cls, fields, allowed=None, limit=6):
        allowed_set = {str(x).strip() for x in list(allowed or []) if str(x).strip()}
        out = []
        seen = set()
        for field in list(fields or []):
            name = cls._summary_hint_field_name(field)
            if not name:
                continue
            if allowed_set and name not in allowed_set:
                continue
            low = name.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(name)
            if len(out) >= int(limit or 6):
                break
        return out

    @staticmethod
    def _output_contract_fields_for_path(tree, path):
        path = dict(path or {})
        what_id = str(path.get("what", "") or path.get("legacy_what", "") or "none")
        how_id = str(path.get("how", "") or "")
        what_node = (tree.get("what", {}) or {}).get(what_id, {}) or {}
        how_node = (tree.get("how", {}) or {}).get(how_id, {}) or {}
        parent_how_id = how_id.split("/", 1)[0] if "/" in how_id else ""
        parent_how_node = (tree.get("how", {}) or {}).get(parent_how_id, {}) or {}
        output_format = list(what_node.get("task_output_format", []) or [])
        if not output_format:
            output_format = output_format_for_what(what_id, how_id)
        how_format = list(how_node.get("advisor_output_format", []) or parent_how_node.get("advisor_output_format", []) or [])
        fields = PathExecutor._output_field_names_from_lines(output_format + how_format)
        if fields:
            return fields
        return PathExecutor._output_field_names_from_lines(output_format_for_what(what_id, how_id))

    @staticmethod
    def _markdown_section(markdown, heading):
        text = str(markdown or "")
        if not text:
            return ""
        pattern = rf"(?ims)^#{2,3}\s+{re.escape(str(heading or '').strip())}\s*$\s*(.*?)(?=^#{2,3}\s+|\Z)"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _compact_markdown_bullets(text, max_items=4, max_len=140):
        rows = []
        for line in str(text or "").splitlines():
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if not line or line in {"---"}:
                continue
            if line.startswith("#"):
                continue
            rows.append(PathExecutor._compact_text(line, max_len))
            if len(rows) >= int(max_items or 4):
                break
        return rows

    @staticmethod
    def _selected_how_contract(how_id, how_node, parent_how_node=None):
        how_node = dict(how_node or {})
        parent_how_node = dict(parent_how_node or {})
        body = str(how_node.get("skill_body", "") or "")
        hints = dict(how_node.get("summary_hints", {}) or {})
        lines = [f"SelectedHowContract: {how_id or 'none'}"]
        goal = (
            how_node.get("description")
            or how_node.get("if_selected")
            or hints.get("task_focus")
            or PathExecutor._markdown_section(body, "Goal")
            or PathExecutor._markdown_section(body, "If Selected")
        )
        if goal:
            lines.append(f"- Goal: {PathExecutor._compact_text(goal, 220)}")
        use_why = how_node.get("use_why") or PathExecutor._markdown_section(body, "Use When")
        if use_why:
            lines.append(f"- UseWhen: {PathExecutor._compact_text(use_why, 180)}")
        actions = PathExecutor._compact_markdown_bullets(
            PathExecutor._markdown_section(body, "Required Actions"),
            max_items=4,
            max_len=140,
        )
        if actions:
            lines.append("- RequiredActions: " + " | ".join(actions))
        preserve = PathExecutor._clean_summary_hint_fields(
            hints.get("preserve_interaction_fields", []),
            allowed={
                "ChallengeOrSupportPrevious",
                "ResponseToPrevious",
                "Correction",
                "PerCandidateValidation",
            },
            limit=4,
        )
        if preserve:
            lines.append("- InteractionFocus: " + ", ".join(preserve[:4]))
        return "\n".join(lines)

    @staticmethod
    def _advisor_skill_payload(tree, path, advisor_profile=None):
        who_id = str((path or {}).get("who", "") or "")
        what_id = str((path or {}).get("what", "") or (path or {}).get("legacy_what", "") or "none")
        how_id = str((path or {}).get("how", "") or "")
        base_what_id = PathExecutor._base_node_id(what_id)
        who_node = (tree.get("who", {}) or {}).get(who_id, {}) or {}
        what_node = (tree.get("what", {}) or {}).get(what_id, {}) or {}
        how_node = (tree.get("how", {}) or {}).get(how_id, {}) or {}
        parent_how_id = how_id.split("/", 1)[0] if "/" in how_id else ""
        parent_how_node = (tree.get("how", {}) or {}).get(parent_how_id, {}) or {}
        advisor_role = str((advisor_profile or {}).get("advisor_type", "") or who_node.get("advisor_role", "") or who_id or "advisor")
        how_summary = str(how_node.get("if_selected", "") or how_node.get("description", "") or how_id or "none")
        output_format = list(what_node.get("task_output_format", []) or [])
        if base_what_id == "reasoning_check":
            # Reasoning-check semantics are easy to confuse with ordinary
            # candidate comparison. Prefer the current canonical format even if
            # an older public-tree node still carries legacy labels.
            output_format = []
        if not output_format:
            output_format = output_format_for_what(what_id, how_id)
        how_format = list(how_node.get("advisor_output_format", []) or parent_how_node.get("advisor_output_format", []) or [])
        for line in how_format:
            line = str(line or "").strip()
            if not line:
                continue
            label = line.split(":", 1)[0].strip().lower() if ":" in line else ""
            replaced = False
            if label:
                for idx, existing in enumerate(list(output_format)):
                    existing_label = str(existing or "").split(":", 1)[0].strip().lower() if ":" in str(existing or "") else ""
                    if existing_label == label:
                        output_format[idx] = line
                        replaced = True
                        break
            if not replaced and line not in output_format:
                output_format.append(line)
        output_format = [str(line).strip() for line in output_format if str(line or "").strip()]
        output_contract_fields = PathExecutor._output_field_names_from_lines(output_format)
        family = infer_communication_family(how_id)
        protocol = {
            "single": "Independently answer Task. No response to previous advisors is required.",
            "cooperative": "First answer Task independently, then use PreviousFriendViews to complement, agree, refine, correct, or integrate useful points. Do not invent novelty just to differ.",
            "competitive": "First answer Task independently. If PreviousFriendViews exist, later advisors must question or rebut one specific prior claim before giving their own judgment; generic agreement, support-only replies, and summaries are protocol violations.",
        }.get(family, "Answer Task under the selected how organization.")
        selected_how_contract = ""
        if "/" in how_id:
            selected_how_contract = PathExecutor._selected_how_contract(how_id, how_node, parent_how_node)
        what_instruction = ""
        if base_what_id == "reasoning_check":
            what_instruction = (
                "WhatInstruction: reasoning_check verifies whether the user's own WhyInHesitation or initial assumptions are reliable. "
                "Do not select a final item and do not turn this into a normal candidate ordering. "
                "Use CandidateView labels reason_reliable, reason_weak, mixed, or unclear to describe the checked reasoning.\n"
            )
        return (
            f"AdvisorRole: {advisor_role}\n"
            f"MappedWhat: {what_id or 'none'}\n"
            f"How: {how_id or 'none'}\n"
            + (f"ParentHow: {parent_how_id}\n" if parent_how_id else "")
            + f"HowInstruction: {protocol}\n"
            + f"HowMeaning: {how_summary}\n"
            + (f"{selected_how_contract}\n" if selected_how_contract else "")
            + what_instruction
            + (
                "SelectedOutputFields: "
                + ", ".join(output_contract_fields)
                + "\nOutputContractRule: Follow this selected what/how OutputFormat exactly. Emit each listed label once; if a listed field is not answerable, write the label with value none or unresolved instead of omitting it. Do not add unrelated fields.\n"
                if output_contract_fields
                else ""
            )
            + "OutputFormat:\n"
            + "\n".join(f"- {line}" for line in output_format)
        ).strip()

    def _protocol_instruction(self, path, advisor_count=0):
        how = self._canonical_how((path or {}).get("how", ""))
        user_task = str((path or {}).get("user_task", "") or "")
        shape = infer_communication_shape(how)
        family = infer_communication_family(how)
        runtime_shape = "one-to-one" if shape == "single" else "one-to-many"
        behavior = {
            "single": "Single mode: independently answer UserTask as one friend; do not invent a debate.",
            "cooperative": "Cooperative mode: independently answer UserTask first, then use memory to complement, refine, integrate, or agree with useful prior points. Do not invent differences just to avoid overlap.",
            "competitive": "Competitive mode: independently form your answer or claim first. If PreviousFriendViews exist, you must explicitly question or rebut at least one earlier claim before giving your final judgment. Do not merely agree, summarize, or repeat earlier claims.",
        }.get(family, "Follow the selected how node.")
        return (
            "Task is the user's real request and must be answered directly. "
            "MappedWhat only determines the output fields; How only determines how to use PreviousFriendViews. "
            "The visible decision space is HesitationSet only; do not use outside candidates or treat any candidate as protected. "
            "CandidateView is required even when the answer is unresolved: write the useful candidate-level observations you can make. "
            f"{behavior}"
        )

    def _how_execution_checklist(self, how, idx, advisor_count, has_previous_memory=False):
        how = self._canonical_how(how)
        family = infer_communication_family(how)
        idx = max(1, int(idx or 1))
        advisor_count = max(1, int(advisor_count or 1))
        if family == "competitive":
            if idx <= 1 or not has_previous_memory:
                return (
                    "HowExecutionChecklist:\n"
                    "1. Make one explicit candidate-level claim that answers UserTask.\n"
                    "2. Fill ChallengeOrSupportPrevious with new_claim and the claim you are making.\n"
                    "3. If several candidates all look valid, name the discriminating dimension or missing preference signal.\n"
                    "4. Do not mark the task resolved without candidate-level discrimination."
                )
            return (
                "HowExecutionChecklist:\n"
                "1. First form your own claim about the HesitationSet.\n"
                "2. Then read PreviousFriendViews and choose one prior advisor claim to question or rebut before you answer.\n"
                "3. Fill ChallengeOrSupportPrevious with rebut/question/correct plus the exact prior advisor or claim you are challenging; do not write none, generic support, or simple agreement.\n"
                "4. If the prior claim is mostly right, still identify its weakest assumption, missing evidence, overreach, or untested candidate comparison.\n"
                "5. If all candidates look valid, name the discriminating dimension or missing preference signal."
            )
        if family == "cooperative":
            if idx <= 1 or not has_previous_memory:
                return (
                    "HowExecutionChecklist:\n"
                    "1. Give your independent answer to UserTask.\n"
                    "2. Cover the clearest candidate evidence you can provide.\n"
                    "3. Name any remaining gap rather than inventing certainty.\n"
                    "4. Fill ResponseToPrevious with none because there is no useful previous view yet."
                )
            return (
                "HowExecutionChecklist:\n"
                "1. First form your independent answer to UserTask.\n"
                "2. Then read PreviousFriendViews and fill ResponseToPrevious with add/refine/integrate/agree/correct plus the prior point you reference.\n"
                "3. Add missing candidate coverage or a missing angle when possible.\n"
                "4. Agreement is allowed only when your own evidence supports the earlier point."
            )
        return (
            "HowExecutionChecklist:\n"
            "1. Independently answer UserTask.\n"
            "2. Use your own history and visible user preference brief as evidence.\n"
            "3. Do not choose the final item for the user.\n"
            "4. CandidateView is still required by the selected what node."
        )

    def _advisor_turn_role(self, family, intent, idx, advisor_count):
        idx = max(1, int(idx or 1))
        advisor_count = max(1, int(advisor_count or 1))
        if intent == "feedback-repair" and family == "cooperative":
            if idx == 1:
                role, task = "cooperative feedback responder", "answer the user's explicit follow-up question using previous-round memory"
            elif idx == advisor_count:
                role, task = "cooperative repair integrator", "check whether the user's feedback is answered and name remaining gaps"
            else:
                role, task = "cooperative evidence completer", "add complementary evidence for a user-requested candidate or comparison"
        elif intent == "feedback-repair" and family == "competitive":
            if idx == 1:
                role, task = "competitive feedback claimant", "answer the user's explicit follow-up with a claim grounded in previous-round memory"
            elif idx == advisor_count:
                role, task = "competitive repair judge", "question or rebut at least one prior claim, then state which answer best resolves the user's question"
            else:
                role, task = "competitive feedback challenger", "question or rebut at least one previous claim in response to the user's question"
        elif family == "cooperative":
            if idx == 1:
                role, task = "cooperative opener", "give one concrete coverage point and name one remaining gap"
            elif idx == advisor_count:
                role, task = "cooperative integrator", "use prior memory as context and give your own grounded judgment about retained/removed/unresolved evidence"
            else:
                role, task = "cooperative extender", "respond to prior memory and add your own evidence-grounded judgment; agreement is allowed when justified"
        elif family == "competitive":
            if idx == 1:
                role, task = "competitive claimant", "make one strong evidence-backed claim"
            elif idx == advisor_count:
                role, task = "competitive claim tester", "question or rebut at least one prior claim, then state which claim survives or remains unresolved"
            else:
                role, task = "competitive challenger", "question or rebut at least one prior claim before making a stronger or more precise claim"
        elif family == "repair":
            if idx == 1:
                role, task = "feedback responder", "answer the user's explicit remaining feedback directly"
            elif idx == advisor_count:
                role, task = "repair checker", "verify whether the user's doubt is resolved or still unresolved"
            else:
                role, task = "missing-evidence responder", "address one silent or missing HesitationSet point when your evidence supports it"
        else:
            role, task = "single evaluator", "give concise advice and name missing evidence"
        return {
            "turn_role": role,
            "turn_task": task,
            "should_consider_previous": bool(idx > 1 and advisor_count > 1),
        }

    def _advisor_friend_guidance(self, how, idx, advisor_count, focus_candidates, proposal_name, previous_user_feedback=None, user_task="", task_type="none"):
        how = self._canonical_how(how)
        shape = infer_communication_shape(how)
        family = infer_communication_family(how)
        focus = [str(x) for x in (focus_candidates or []) if str(x or "").strip()]
        if not focus:
            focus = [str(proposal_name or "")]
        advice_values = "keep | switch | caution | unresolved"
        role = "friend advisor"
        objective = "Give the requester practical advice from their perspective, using your evidence and the shared requester brief."
        if shape == "single":
            role = "single friend advisor"
            objective = "Answer UserTask independently from your own evidence and understanding of the requester."
        elif family == "cooperative":
            role = "cooperative friend advisor"
            objective = "Answer the same UserTask as the other friends: think independently first, then use memory to complement, refine, integrate, or agree."
        elif family == "competitive":
            role = "competitive friend advisor"
            objective = "Answer UserTask by making a claim and, after the first advisor, explicitly questioning or rebutting an earlier claim before giving your own judgment."
        turn = self._advisor_turn_role(family, "user-task", idx, advisor_count)

        guidance = {
            "advisor_index": int(idx),
            "advisor_count": max(0, int(advisor_count or 0)),
            "selected_how": str(how or ""),
            "user_task": str(user_task or ""),
            "task_type": str(task_type or "none"),
            "mode_family": str(family or ""),
            "intent": "user-task",
            "role": role,
            "turn_role": turn["turn_role"],
            "turn_task": turn["turn_task"],
            "should_consider_previous": bool(turn["should_consider_previous"]),
            "allowed_advice_values": advice_values,
            "objective": objective,
            "decision_space": list(focus),
            "group_memory_definition": "current_round_discussion_memory is lightweight: each line is advisor_id: essence of that advisor's useful contribution.",
            "thinking_order": [
                "first read UserTask, the whole HesitationSet, and UserInitialChoiceContext",
                "then form your independent answer from your evidence and understanding of the requester",
                (
                    "after that, read group memory and question or rebut one earlier friend before your own judgment"
                    if family == "competitive"
                    else "after that, read group memory and respond to earlier friends by agreeing, refining, challenging, or adding useful context"
                ),
            ],
            "memory_policy": (
                "Use previous_round_discussion_memory for continuity across rounds and current_round_discussion_memory for this round only after independent thinking. "
                "In competitive mode, later advisors must explicitly question or rebut a prior claim; do not answer with generic agreement or support-only text."
                if family == "competitive"
                else "Use previous_round_discussion_memory for continuity across rounds and current_round_discussion_memory for this round only after independent thinking. You may agree with an earlier advisor if your own evidence supports it; explain your own reason. Do not invent novelty just to be different."
            ),
        }
        if task_type == "reduce_hesitation_set":
            guidance["set_policy"] = [
                "the task is helping the requester shrink the HesitationSet",
                "look at all candidates before naming risk",
                "ShrinkSet can include one or more candidates the requester is likely not interested in",
                "retain or mark unresolved when evidence is insufficient",
            ]
            guidance["no_protected_default_candidate"] = True
            guidance["mode_goal"] = "task_reduce_hesitation_set"
        elif task_type == "find_interested_subset":
            guidance["set_policy"] = [
                "the task is helping the requester keep a shorter interested set",
                "look at all candidates before naming support",
                "RetainSet should be shorter than the original HesitationSet whenever the evidence allows",
                "mark support weak or unresolved when evidence is insufficient",
            ]
            guidance["mode_goal"] = "task_find_interested_subset"
        elif task_type == "compare_remaining_candidates":
            guidance["claim_policy"] = [
                "compare the candidates named by UserTask, or the most relevant HesitationSet candidates if none are named",
                "in competitive mode, later advisors must question or rebut a prior claim before making their own comparison",
                "competition may end unresolved if no comparison is decisive",
            ]
            guidance["mode_goal"] = "task_compare_remaining_candidates"
        elif task_type == "evidence_gap_check":
            guidance["evidence_policy"] = [
                "check whether the current or original reason for each relevant candidate is insufficient",
                "state the supplementary reason or evidence needed to make that candidate-level reasoning useful",
                "do not refuse to speak just because evidence is partial",
            ]
            guidance["mode_goal"] = "task_evidence_gap_check"
        elif task_type == "reasoning_check":
            guidance["reasoning_policy"] = [
                "check whether the user's initial reasoning, assumptions, or surface matches are actually valid",
                "separate supported assumptions from weak or questionable assumptions",
                "do not simply restate the user's initial reason; verify it from the requester history and your evidence",
            ]
            guidance["mode_goal"] = "task_reasoning_check"
        guidance["memory_use"] = "Use current_round_discussion_memory after independent thinking; use previous_round_discussion_memory to keep continuity across rounds with the same friends."
        guidance["how_execution_checklist"] = self._how_execution_checklist(
            how,
            idx,
            advisor_count,
            has_previous_memory=bool(int(idx or 1) > 1 and int(advisor_count or 1) > 1),
        )
        if previous_user_feedback:
            guidance["previous_user_feedback_focus"] = previous_user_feedback
        return guidance

    def _focus_candidates(self, host, cands_int, shortlist_names, proposal_name, limit=None):
        names = []
        seen = set()
        for raw in [proposal_name] + list(shortlist_names or []):
            key = str(raw or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            names.append(key)
        # Advisors see one merged Stage-1 HesitationSet, not a separate
        # proposal plus alternatives and not the original recommender pool.
        if limit is None:
            return names
        return names[: max(1, int(limit))]

    @staticmethod
    def _split_candidate_mentions(value):
        if isinstance(value, list):
            rows = []
            for item in value:
                rows.extend(PathExecutor._split_candidate_mentions(item))
            return rows
        return [x.strip() for x in re.split(r"[,;|\n]+", str(value or "")) if x.strip()]

    @staticmethod
    def _task_scoped_focus_candidates(base_focus, path, round_type):
        base = [str(x).strip() for x in list(base_focus or []) if str(x or "").strip()]
        if str(round_type or "") != "continued_user_task" or not base:
            return base, []
        allowed = {x.lower(): x for x in base}
        user_task = str((path or {}).get("user_task", "") or "")
        selected = []
        for raw in PathExecutor._split_candidate_mentions((path or {}).get("task_targets", [])):
            key = str(raw or "").strip().lower()
            if key in allowed and allowed[key] not in selected:
                selected.append(allowed[key])
        task_lower = user_task.lower()
        for item in base:
            if item.lower() in task_lower and item not in selected:
                selected.append(item)
        if 1 < len(selected) < len(base):
            return selected, [
                "follow-up UserTask names a narrower candidate set; advisor-visible HesitationSet is scoped to those items"
            ]
        return base, []

    @staticmethod
    def _norm_name(value):
        return " ".join(str(value or "").strip().lower().split())

    def _resolve_allowed_item(self, item_name, allowed_items):
        norm = self._norm_name(item_name)
        if not norm:
            return ""
        for item in allowed_items or []:
            if self._norm_name(item) == norm:
                return str(item)
        return ""

    def _resolve_allowed_item_set(self, value, allowed_items):
        rows = []
        seen = set()
        for raw in self._split_candidate_mentions(value):
            item = self._resolve_allowed_item(raw, allowed_items)
            if item and item not in seen:
                seen.add(item)
                rows.append(item)
        return rows

    def _parse_candidate_view(self, value, allowed_items):
        views = []
        seen = set()
        allowed = list(allowed_items or [])
        allowed_sorted = sorted([str(x) for x in allowed if str(x or "").strip()], key=len, reverse=True)
        for raw in str(value or "").splitlines():
            line = re.sub(r"^\s*[-*]\s*", "", str(raw or "")).strip()
            if not line or self._is_none_like(line):
                continue
            parts = [p.strip() for p in re.split(r"\s*\|\s*", line, maxsplit=2)]
            item = ""
            label = ""
            reason = ""
            if len(parts) >= 2:
                item = self._resolve_allowed_item(parts[0], allowed)
                label = str(parts[1] or "").strip().lower()
                reason = parts[2].strip() if len(parts) >= 3 else ""
            if not item:
                inline_matches = []
                for candidate in allowed_sorted:
                    pattern = rf"{re.escape(candidate)}\s*(?:[:：]|[-–—]\s+)"
                    for match in re.finditer(pattern, line, flags=re.IGNORECASE):
                        inline_matches.append((match.start(), match.end(), candidate))
                if inline_matches:
                    inline_matches.sort(key=lambda row: (row[0], -(row[1] - row[0])))
                    filtered = []
                    occupied_until = -1
                    for start, end, candidate in inline_matches:
                        if start < occupied_until:
                            continue
                        filtered.append((start, end, candidate))
                        occupied_until = end
                    for idx, (start, end, candidate) in enumerate(filtered):
                        next_start = filtered[idx + 1][0] if idx + 1 < len(filtered) else len(line)
                        inline_reason = line[end:next_start].strip(" ;,，。")
                        if not inline_reason:
                            continue
                        key = (candidate, "evidence", self._argument_key(inline_reason))
                        if key in seen:
                            continue
                        seen.add(key)
                        views.append(
                            {
                                "candidate": candidate,
                                "view": "evidence",
                                "reason": inline_reason,
                            }
                        )
                    continue
            if not item:
                for candidate in allowed_sorted:
                    pattern = rf"^\s*{re.escape(candidate)}\s*(?:[:：]|[-–—]\s+)\s*(.+?)\s*$"
                    match = re.match(pattern, line, flags=re.IGNORECASE)
                    if match:
                        item = candidate
                        label = "evidence"
                        reason = match.group(1).strip()
                        break
            if not item:
                for candidate in allowed_sorted:
                    candidate_norm = self._norm_name(candidate)
                    line_norm = self._norm_name(line)
                    if candidate_norm and line_norm.startswith(candidate_norm):
                        tail = line[len(candidate) :].strip(" :-–—\t")
                        if tail:
                            item = candidate
                            label = "evidence"
                            reason = tail.strip()
                            break
            if not item:
                continue
            if not label:
                label = "evidence"
            if not reason and len(parts) >= 2:
                reason = str(parts[1] or "").strip()
            key = (item, label, self._argument_key(reason))
            if key in seen:
                continue
            seen.add(key)
            views.append(
                {
                    "candidate": item,
                    "view": label,
                    "reason": reason,
                }
            )
        return views

    @staticmethod
    def _candidate_view_bucket(label):
        label = str(label or "").strip().lower()
        support = {"keep", "interest", "interested", "stronger", "valid", "reason_reliable", "covered", "support", "positive"}
        risk = {"remove", "weak", "weaker", "reason_weak", "risk", "weak_fit", "invalid", "negative"}
        unclear = {"unclear", "unknown", "unresolved", "mixed", "partial", "partially_resolved", "missing", "evidence_gap", "gap"}
        if label in support:
            return "support"
        if label in risk:
            return "risk"
        if label in unclear:
            return "unclear" if label != "mixed" else "mixed"
        if "missing" in label:
            return "unclear"
        if "weak" in label or "remove" in label or "risk" in label:
            return "risk"
        if "reliable" in label or "keep" in label or "support" in label or "strong" in label or "valid" in label or "cover" in label:
            return "support"
        return "unclear"

    @staticmethod
    def _normalize_candidate_view_label(label, task_type=""):
        label = str(label or "").strip().lower()
        task_type = str(task_type or "").strip().split("/", 1)[0]
        if task_type == "reasoning_check":
            mapping = {
                "valid": "reason_reliable",
                "support": "reason_reliable",
                "positive": "reason_reliable",
                "weak": "reason_weak",
                "weaker": "reason_weak",
                "invalid": "reason_weak",
                "negative": "reason_weak",
            }
            return mapping.get(label, label or "unclear")
        return label or "unclear"

    def _candidate_evidence_sources(self, candidate_views, task_type=""):
        sources = []
        for view in list(candidate_views or []):
            if not isinstance(view, dict):
                continue
            source_label = str(view.get("view", "") or "").strip().lower()
            normalized = self._normalize_candidate_view_label(source_label, task_type)
            sources.append(
                {
                    "candidate": str(view.get("candidate", "") or ""),
                    "label": self._candidate_view_bucket(normalized),
                    "source_label": normalized,
                    "reason": str(view.get("reason", "") or ""),
                    "what": str(task_type or ""),
                }
            )
        return sources

    def _extract_neg_reason(self, speech):
        text = str(speech or "")
        m = re.search(r"Neg:\s*(.*?)\s*Pos:", text, re.DOTALL | re.IGNORECASE)
        if m:
            reason = m.group(1).strip()
            reason = re.sub(
                r"^\s*ChallengedItem\s*:\s*.*?(?=(?:EvidenceAgainst|WarningEvidence|PromotionEvidence|AddedCoverage|ChallengeToPreviousClaim|ResponseToMemory|UnresolvedIssue)\s*:|$)",
                "",
                reason,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            return reason
        return ""

    @staticmethod
    def _is_none_like(value):
        return str(value or "").strip().lower() in ["", "none", "null", "n/a", "na"]

    @staticmethod
    def _extract_labeled_value(text, label):
        body = str(text or "")
        if not body:
            return ""
        m = re.search(
            rf"^\s*(?:[-*]\s*)?(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*:\s*"
            rf"(.*?)(?=^\s*(?:[-*]\s*)?(?:\*\*)?[A-Za-z][A-Za-z0-9_ ]{{0,40}}(?:\*\*)?\s*:|\Z)",
            body,
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_labeled_fields(text):
        body = str(text or "")
        if not body:
            return {}
        pattern = re.compile(
            r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?([A-Za-z][A-Za-z0-9_ ]{0,40})(?:\*\*)?\s*:\s*"
            r"(.*?)(?=^\s*(?:[-*]\s*)?(?:\*\*)?[A-Za-z][A-Za-z0-9_ ]{0,40}(?:\*\*)?\s*:|\Z)",
            re.DOTALL | re.MULTILINE,
        )
        fields = {}
        for match in pattern.finditer(body):
            label = re.sub(r"\s+", "", str(match.group(1) or "").strip())
            value = str(match.group(2) or "").strip()
            if not label or not value:
                continue
            key = PathExecutor._snake_field_name(label)
            if key in fields:
                fields[key] = f"{fields[key]}\n{value}".strip()
            else:
                fields[key] = value
        return fields

    @staticmethod
    def _known_advisor_field_keys():
        labels = [
            "FriendBasis",
            "RemoveSet",
            "ShrinkSet",
            "KeepSet",
            "RetainSet",
            "InterestedSet",
            "WeakFitSet",
            "StrongerCandidate",
            "StrongerCandidates",
            "WeakerCandidate",
            "WeakerCandidates",
            "UnclearSet",
            "CoveredSet",
            "MissingSet",
            "EvidenceGapSet",
            "EvidenceICanAdd",
            "CoveredEvidence",
            "EvidenceAdded",
            "StillMissing",
            "MissingEvidence",
            "SupplementReason",
            "SupplementalReason",
            "SupportedAssumption",
            "ReliableReasons",
            "ValidReasoning",
            "QuestionedAssumption",
            "WeakReasons",
            "WeakReasoning",
            "Concern",
            "TaskAnswer",
            "AskUser",
            "ResponseToPrevious",
            "ChallengeOrSupportPrevious",
            "KeyTradeoff",
            "ComparisonReason",
            "Correction",
            "CandidateView",
            "ChallengedItem",
        ]
        return {PathExecutor._snake_field_name(label) for label in labels}

    def _extract_challenged_item(self, speech, allowed_items):
        raw = self._extract_labeled_value(speech, "ChallengedItem")
        if self._is_none_like(raw):
            return ""
        return self._resolve_allowed_item(raw, allowed_items)

    def _reason_mentions_other_candidate(self, reason, endorsed_item, allowed_items):
        text = self._norm_name(reason)
        endorsed_norm = self._norm_name(endorsed_item)
        if not text or not endorsed_norm or endorsed_norm in text:
            return ""
        for item in allowed_items or []:
            item_norm = self._norm_name(item)
            if item_norm and item_norm != endorsed_norm and item_norm in text:
                return str(item)
        return ""

    def _advisor_context(self, host, profile):
        top_items = [str(x) for x in list(profile.get("top_item_names", []) or []) if str(x).strip()]
        return ", ".join(top_items[:12]) if top_items else "none"

    def _advisor_own_skill_context(self, host, profile, max_chars=900):
        if not bool(getattr(self.args, "com_enable_advisor_own_skill", True)):
            return ""
        advisor_raw = str((profile or {}).get("u_raw", "") or "").strip()
        if not advisor_raw:
            return ""
        store = getattr(host, "user_policy_store", None)
        if store is None:
            tree_engine = getattr(host, "tree_engine", None)
            store = getattr(tree_engine, "user_policy_store", None)
        if store is None or not hasattr(store, "_paths"):
            return ""
        def clean_skill(value):
            text = str(value or "").strip()
            if not text:
                return ""
            low = text.lower()
            if (
                text.startswith("{")
                and (
                    "item_selection_skill" in low
                    or "communication_route_skill" in low
                    or "'phase': 'communication" in low
                    or '"phase": "communication' in low
                )
            ):
                return ""
            return re.sub(r"\s+", " ", text).strip()

        skill = ""
        try:
            paths = store._paths(advisor_raw)
            # Advisor speech needs the advisor's own taste evidence. Prefer the
            # proposal/item slim cache; the generic latest cache may have been
            # overwritten by communication/redecision phases and can contain a
            # large structured policy dump instead of a clean taste summary.
            for key in (
                "slim_cache_proposal_json",
                "slim_cache_post_feedback_json",
                "slim_cache_json",
                "slim_cache_communication_json",
            ):
                candidate_path = paths.get(key)
                if not candidate_path or not Path(candidate_path).exists():
                    continue
                try:
                    with open(candidate_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    skill = clean_skill(payload.get("compact_slim_policy", "") or payload.get("slim_policy", ""))
                elif isinstance(payload, str):
                    skill = clean_skill(payload)
                if skill:
                    break
        except Exception:
            skill = ""
        if not skill:
            return ""
        if len(skill) > int(max_chars):
            skill = skill[: int(max_chars) - 3].rstrip() + "..."
        return skill

    @staticmethod
    def _compact_initial_choice_context(proposal_reason, hesitation_reason, candidate_evidence, focus_candidates, proposal_name=""):
        allowed = {str(x).strip() for x in (focus_candidates or []) if str(x or "").strip()}

        def compact(text, max_len=220):
            text = re.sub(r"\s+", " ", str(text or "")).strip()
            if len(text) <= max_len:
                return text
            return text[: max(0, max_len - 3)].rstrip() + "..."

        def norm(text):
            return " ".join(str(text or "").strip().lower().split())

        lines = [f"HesitationReason: {compact(hesitation_reason) if hesitation_reason else 'none'}"]
        evidence_by_candidate = {}
        for row in list(candidate_evidence or []):
            if not isinstance(row, dict):
                continue
            candidate = str(row.get("candidate", "") or "").strip()
            if allowed and candidate not in allowed:
                continue
            reason = compact(row.get("reason", ""), 140)
            fit = compact(row.get("fit", ""), 120)
            why = reason or fit
            if candidate and why:
                evidence_by_candidate[candidate] = why
        evidence_lines = []
        for candidate in list(focus_candidates or []):
            candidate = str(candidate or "").strip()
            if not candidate:
                continue
            why = evidence_by_candidate.get(candidate)
            if not why and proposal_name and norm(candidate) == norm(proposal_name):
                why = compact(proposal_reason, 140)
            if not why:
                why = "Stage1 hesitation candidate; re-check against UserTask and advisor evidence."
            evidence_lines.append(f"{candidate} | {why}")
        lines.append("WhyEachItemIsInHesitation:")
        lines.extend(evidence_lines[:5] if evidence_lines else ["none"])
        return "\n".join(lines)

    @staticmethod
    def _compact_discussion_memory(memory_rows, max_items=4, max_reason_len=180):
        def compact(text, limit):
            text = re.sub(r"\s+", " ", str(text or "")).strip()
            if len(text) > int(limit):
                return text[: int(limit) - 3].rstrip() + "..."
            return text

        rows = []
        for row in list(memory_rows or [])[-max_items:]:
            advisor = str(row.get("advisor", "") or "advisor")
            main_claim = compact(row.get("main_claim", "") or row.get("reason", "") or "", max_reason_len)
            candidate_views = list(row.get("candidate_views", []) or [])
            gaps = compact(row.get("remaining_gap", ""), 140)
            ask_user = compact(row.get("ask_user", ""), 140)
            if ask_user and ask_user == gaps:
                ask_user = ""
            response = compact(row.get("response_to_previous", "") or row.get("challenge_or_support_previous", "") or "", 160)
            block = [f"{advisor}:"]
            block.append(f"MainClaim: {main_claim or 'no concise claim'}")
            if candidate_views:
                block.append("CandidateView:")
                for view in candidate_views[:3]:
                    block.append(
                        f"- {view.get('candidate', '')} | {view.get('view', 'unclear')} | {compact(view.get('reason', '') or 'no short reason', 120)}"
                    )
            if gaps:
                block.append(f"RemainingGap: {gaps}")
            if ask_user:
                block.append(f"AskUser: {ask_user}")
            if response:
                block.append(f"ResponseToPrevious: {response}")
            rows.append("\n".join(block))
        return "\n".join(rows) if rows else "none"

    @staticmethod
    def _compact_previous_round_summary(previous_round_summary, max_len=650):
        if not previous_round_summary:
            return "none"
        if isinstance(previous_round_summary, str):
            text = previous_round_summary
        elif isinstance(previous_round_summary, dict):
            packet = dict(previous_round_summary.get("evidence_packet", {}) or {})
            summary = dict(packet.get("evidence_summary", {}) or previous_round_summary.get("evidence_summary", {}) or {})
            parts = []
            result = str(summary.get("discussion_result", "") or previous_round_summary.get("source", "") or "").strip()
            retained = [str(x) for x in list(summary.get("retained_candidates", []) or []) if str(x).strip()]
            silent = [str(x) for x in list(summary.get("silent_focus_candidates", []) or []) if str(x).strip()]
            unresolved = [str(x) for x in list(summary.get("unresolved_questions", []) or previous_round_summary.get("remaining_uncertainty", []) or []) if str(x).strip()]
            by_candidate = dict(summary.get("by_candidate", {}) or {})
            candidate_evidence = list(summary.get("candidate_evidence", []) or [])
            discussion_summary = dict(summary.get("discussion_summary", {}) or {})
            if result:
                parts.append(f"discussion_result={result}")
            if retained:
                parts.append("retained=" + ", ".join(retained[:4]))
            if candidate_evidence:
                parts.append("CandidateEvidence:")
                for row in candidate_evidence[:3]:
                    row = dict(row or {})
                    counts = dict(row.get("counts", {}) or {})
                    parts.append(
                        f"- {row.get('candidate', '')} | {row.get('status', 'unclear')} | "
                        f"{counts.get('support', 0)} support, {counts.get('risk', 0)} risk, {counts.get('unclear', 0)} unclear | "
                        f"{row.get('reason', '') or 'none'}"
                    )
            if discussion_summary:
                gap = str(discussion_summary.get("remaining_gap", "") or "").strip()
                questions = [str(x) for x in list(discussion_summary.get("advisor_questions_for_user", []) or []) if str(x).strip()]
                if gap:
                    parts.append("RemainingGap: " + gap)
                if questions:
                    parts.append("AdvisorQuestionsForUser: " + " | ".join(questions[:4]))
            for item, row in ([] if candidate_evidence else list(by_candidate.items())[:5]):
                row = dict(row or {})
                support = [str(x.get("reason", "") or "").strip() for x in list(row.get("support", []) or [])[:1] if str(x.get("reason", "") or "").strip()]
                against = [str(x.get("reason", "") or "").strip() for x in list(row.get("against", []) or [])[:1] if str(x.get("reason", "") or "").strip()]
                item_parts = []
                if support:
                    item_parts.append("support: " + support[0])
                if against:
                    item_parts.append("concern: " + against[0])
                if item_parts:
                    parts.append(f"{item}: " + " | ".join(item_parts))
            if silent:
                parts.append("not_discussed=" + ", ".join(silent[:4]))
            if unresolved:
                parts.append("unresolved=" + ", ".join(unresolved[:4]))
            text = "\n".join(parts) if parts else "none"
        else:
            text = str(previous_round_summary)
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(text) > int(max_len):
            text = text[: int(max_len) - 3].rstrip() + "..."
        return text or "none"

    def _append_discussion_memory(self, memory_rows, profile, feedback):
        candidate_views = list((feedback or {}).get("candidate_views", []) or [])
        if not candidate_views:
            for key, status in [
                ("remove_set", "remove"),
                ("weak_fit_set", "weak"),
                ("weaker_candidates", "weaker"),
                ("keep_set", "keep"),
                ("interested_set", "interest"),
                ("stronger_candidates", "stronger"),
                ("covered_set", "covered"),
                ("missing_set", "missing"),
                ("evidence_gap_set", "evidence_gap"),
                ("unclear_set", "unclear"),
            ]:
                for item in list((feedback or {}).get(key, []) or []):
                    candidate_views.append(
                        {
                            "candidate": str(item),
                            "view": status,
                            "reason": str((feedback or {}).get("task_answer", "") or (feedback or {}).get("support_reason", "") or (feedback or {}).get("oppose_reason", "") or ""),
                        }
                    )
        main_claim = str((feedback or {}).get("task_answer", "") or (feedback or {}).get("key_tradeoff", "") or (feedback or {}).get("support_reason", "") or (feedback or {}).get("oppose_reason", "") or "").strip()
        if not main_claim:
            sets = []
            for label, key in [("remove", "remove_set"), ("keep", "keep_set"), ("unclear", "unclear_set")]:
                values = [str(x) for x in list((feedback or {}).get(key, []) or []) if str(x).strip()]
                if values:
                    sets.append(f"{label}: {', '.join(values[:3])}")
            main_claim = "; ".join(sets)
        remaining_gap = str((feedback or {}).get("still_missing", "") or (feedback or {}).get("ask_user", "") or "").strip()
        memory_rows.append(
            {
                "advisor": str(profile.get("u_raw", "") or "advisor"),
                "stance": str((feedback or {}).get("stance", "") or ""),
                "item": str((feedback or {}).get("attacked_item", "") or (feedback or {}).get("defended_item", "") or (feedback or {}).get("endorsed_item", "") or ""),
                "defended_item": str((feedback or {}).get("defended_item", "") or (feedback or {}).get("endorsed_item", "") or ""),
                "attacked_item": str((feedback or {}).get("attacked_item", "") or ""),
                "reason": main_claim,
                "main_claim": main_claim,
                "candidate_views": candidate_views[:8],
                "remaining_gap": remaining_gap,
                "ask_user": str((feedback or {}).get("ask_user", "") or ""),
                "response_to_previous": str((feedback or {}).get("response_to_previous", "") or ""),
                "challenge_or_support_previous": str((feedback or {}).get("challenge_or_support_previous", "") or ""),
            }
        )

    def _parse_feedback(
        self,
        host,
        profile,
        decision,
        speech,
        alt_name,
        pos,
        proposal_name,
        allowed_items=None,
        forced_endorsed_item="",
        forced_attacked_item="",
        protocol="",
        advisor_guidance=None,
        group_memory="",
    ):
        allowed_items = [str(x) for x in (allowed_items or []) if str(x or "").strip()]
        issues = []
        guidance = dict(advisor_guidance or {})
        task_type = str(guidance.get("task_type", "") or (guidance.get("selected_what", "") if isinstance(guidance, dict) else "") or "")
        task_type = self._base_node_id(task_type)
        stance = "support_proposal"
        endorsed_item = proposal_name
        support_reason = str(pos or "").strip()
        oppose_reason = self._extract_neg_reason(speech)
        decision_norm = self._norm_name(decision)
        structured_fields = self._extract_labeled_fields(speech)
        known_field_keys = self._known_advisor_field_keys()
        output_contract_fields = [
            str(x)
            for x in list(guidance.get("output_contract_fields", []) or guidance.get("required_output_fields", []) or [])
            if str(x).strip()
        ]
        output_contract_keys = {self._snake_field_name(x): str(x) for x in output_contract_fields}
        missing_output_contract_fields = [
            label
            for key, label in output_contract_keys.items()
            if key not in structured_fields
        ]
        extra_structured_fields = {
            key: value
            for key, value in structured_fields.items()
            if key not in known_field_keys and not self._is_none_like(value)
        }

        decision_item = self._resolve_allowed_item(decision, allowed_items)
        if self._is_none_like(alt_name):
            alt_name = ""
        alt_item = self._resolve_allowed_item(alt_name, allowed_items)
        friend_basis = self._extract_labeled_value(speech, "FriendBasis")
        remove_set = self._resolve_allowed_item_set(
            self._extract_labeled_value(speech, "RemoveSet") or self._extract_labeled_value(speech, "ShrinkSet"),
            allowed_items,
        )
        keep_set = self._resolve_allowed_item_set(
            self._extract_labeled_value(speech, "KeepSet") or self._extract_labeled_value(speech, "RetainSet"),
            allowed_items,
        )
        interested_set = self._resolve_allowed_item_set(self._extract_labeled_value(speech, "InterestedSet"), allowed_items)
        weak_fit_set = self._resolve_allowed_item_set(self._extract_labeled_value(speech, "WeakFitSet"), allowed_items)
        stronger_set = self._resolve_allowed_item_set(
            self._extract_labeled_value(speech, "StrongerCandidate") or self._extract_labeled_value(speech, "StrongerCandidates"),
            allowed_items,
        )
        weaker_set = self._resolve_allowed_item_set(
            self._extract_labeled_value(speech, "WeakerCandidate") or self._extract_labeled_value(speech, "WeakerCandidates"),
            allowed_items,
        )
        unclear_set = self._resolve_allowed_item_set(self._extract_labeled_value(speech, "UnclearSet"), allowed_items)
        covered_set = self._resolve_allowed_item_set(self._extract_labeled_value(speech, "CoveredSet"), allowed_items)
        missing_set = self._resolve_allowed_item_set(self._extract_labeled_value(speech, "MissingSet"), allowed_items)
        evidence_gap_set = self._resolve_allowed_item_set(self._extract_labeled_value(speech, "EvidenceGapSet"), allowed_items)
        evidence_can_add = (
            self._extract_labeled_value(speech, "EvidenceICanAdd")
            or self._extract_labeled_value(speech, "CoveredEvidence")
            or self._extract_labeled_value(speech, "EvidenceAdded")
        )
        still_missing = self._extract_labeled_value(speech, "StillMissing") or self._extract_labeled_value(speech, "MissingEvidence")
        supplement_reason = self._extract_labeled_value(speech, "SupplementReason") or self._extract_labeled_value(speech, "SupplementalReason")
        supported_assumption = (
            self._extract_labeled_value(speech, "SupportedAssumption")
            or self._extract_labeled_value(speech, "ReliableReasons")
            or self._extract_labeled_value(speech, "ValidReasoning")
        )
        questioned_assumption = (
            self._extract_labeled_value(speech, "QuestionedAssumption")
            or self._extract_labeled_value(speech, "WeakReasons")
            or self._extract_labeled_value(speech, "WeakReasoning")
        )
        concern = self._extract_labeled_value(speech, "Concern")
        task_answer = self._extract_labeled_value(speech, "TaskAnswer")
        ask_user = self._extract_labeled_value(speech, "AskUser")
        response_prev = self._extract_labeled_value(speech, "ResponseToPrevious")
        challenge_prev = self._extract_labeled_value(speech, "ChallengeOrSupportPrevious")
        key_tradeoff = self._extract_labeled_value(speech, "KeyTradeoff")
        comparison_reason = self._extract_labeled_value(speech, "ComparisonReason")
        correction = self._extract_labeled_value(speech, "Correction")
        candidate_views = self._parse_candidate_view(self._extract_labeled_value(speech, "CandidateView"), allowed_items)
        if task_type == "reasoning_check" and candidate_views:
            normalized_views = []
            for view in candidate_views:
                view = dict(view or {})
                view["view"] = self._normalize_candidate_view_label(view.get("view", ""), task_type)
                normalized_views.append(view)
            candidate_views = normalized_views
        candidate_evidence_sources = self._candidate_evidence_sources(candidate_views, task_type)
        forced_item = self._resolve_allowed_item(forced_endorsed_item, allowed_items)
        forced_attack_item = self._resolve_allowed_item(forced_attacked_item, allowed_items)
        generated_attacked_item = self._extract_challenged_item(speech, allowed_items)
        attacked_item = forced_attack_item
        if not attacked_item:
            attacked_item = generated_attacked_item
        if not attacked_item and (remove_set or weak_fit_set or weaker_set):
            attacked_item = (remove_set or weak_fit_set or weaker_set)[0]
        if not alt_item and (interested_set or keep_set or stronger_set):
            alt_item = (interested_set or keep_set or stronger_set)[0]
        evidence_item = alt_item or decision_item or forced_item or ""
        has_negative_signal = bool(
            remove_set
            or weak_fit_set
            or weaker_set
            or generated_attacked_item
            or (questioned_assumption and not self._is_none_like(questioned_assumption))
            or (concern and not self._is_none_like(concern))
            or any(self._candidate_view_bucket(v.get("view")) in {"risk", "mixed"} for v in candidate_views)
        )
        has_positive_signal = bool(
            keep_set
            or interested_set
            or stronger_set
            or covered_set
            or (supported_assumption and not self._is_none_like(supported_assumption))
            or (evidence_can_add and not self._is_none_like(evidence_can_add))
            or any(self._candidate_view_bucket(v.get("view")) in {"support", "mixed"} for v in candidate_views)
        )

        if forced_item:
            endorsed_item = forced_item
            if decision_item and decision_item != forced_item:
                issues.append("assigned_candidate_ignored")
            if alt_item and alt_item != forced_item:
                issues.append("assigned_candidate_ignored")
        elif decision_item:
            endorsed_item = decision_item
            issues.append("vote_parse_mismatch")
        elif alt_item:
            endorsed_item = alt_item
        elif alt_name:
            issues.append("out_of_focus_candidate_generated")
            endorsed_item = proposal_name
        if forced_attack_item and generated_attacked_item and generated_attacked_item != forced_attack_item:
            issues.append("assigned_challenge_ignored")
        elif forced_attack_item and not generated_attacked_item:
            issues.append("assigned_challenge_missing")

        valid_decisions = {
            "agree",
            "support",
            "yes",
            "keep",
            "disagree",
            "oppose",
            "no",
            "switch",
            "caution",
            "challenge",
            "answer",
            "unresolved",
            "rebut",
            "resolved",
            "partially_resolved",
        }
        if decision_norm and decision_norm not in valid_decisions and not decision_item:
            issues.append("invalid_decision_value")

        if alt_name and allowed_items and not alt_item:
            issues.append("out_of_focus_candidate_generated")

        if decision_norm in {"answer", "evidence_only"}:
            stance = "evidence_only"
        elif decision_norm in {"unresolved", "partially_resolved"}:
            stance = "unresolved"
        elif decision_norm == "caution":
            stance = "caution"
            if not attacked_item and evidence_item and has_negative_signal:
                attacked_item = evidence_item
            elif not attacked_item and endorsed_item:
                attacked_item = endorsed_item
        elif self._norm_name(endorsed_item) == self._norm_name(proposal_name):
            stance = "support_proposal"
        else:
            stance = "support_candidate"

        if stance == "evidence_only":
            # Structured what-node answers are candidate-level evidence, not a
            # single vote. Keep alt/decision as reviewed_item only; CandidateView
            # and task-specific sets carry support/risk semantics downstream.
            endorsed_item = ""
        if not support_reason and decision_norm not in {"answer", "evidence_only"}:
            support_reason = str(speech or "").strip()
        if decision_norm == "caution" and evidence_item and has_negative_signal:
            # In reasoning_check, SuggestedItem is the candidate being checked, not an endorsement.
            # Keep it out of positive voting while preserving its risk evidence.
            endorsed_item = proposal_name
        mismatch_reference_item = evidence_item if decision_norm == "caution" and evidence_item else endorsed_item
        mismatched_item = self._reason_mentions_other_candidate(support_reason, mismatch_reference_item, allowed_items)
        if mismatched_item:
            issues.append(f"reason_item_mismatch:{mismatched_item}")
        candidate_set_signal = bool(
            remove_set
            or keep_set
            or interested_set
            or weak_fit_set
            or stronger_set
            or weaker_set
            or unclear_set
            or covered_set
            or missing_set
            or evidence_gap_set
            or candidate_views
            or (evidence_item and (has_negative_signal or has_positive_signal))
        )
        if not candidate_set_signal:
            issues.append("no_candidate_set_signal")
        if evidence_item and attacked_item and self._norm_name(evidence_item) == self._norm_name(attacked_item) and has_positive_signal and has_negative_signal:
            issues.append("mixed_evidence_same_candidate")
        low_info_markers = ["insufficient evidence", "not enough information", "cannot definitively", "无法判定", "信息不足", "不能判定"]
        if decision_norm in {"unresolved", "partially_resolved"} and not candidate_set_signal and any(x in support_reason.lower() for x in low_info_markers):
            issues.append("low_information_unresolved")
        advisor_index = int(guidance.get("advisor_index", 1) or 1)
        advisor_count = int(guidance.get("advisor_count", 1) or 1)
        family = infer_communication_family(protocol)
        group_memory_text = str(group_memory or "").strip()
        has_previous_memory = bool(group_memory_text and group_memory_text.lower() not in {"none", "[]", "{}"})
        if family == "competitive" and advisor_count > 1 and advisor_index > 1 and has_previous_memory:
            if self._is_none_like(challenge_prev):
                issues.append("competitive_no_response_to_previous")
            else:
                challenge_text = str(challenge_prev or "").strip().lower()
                rebuttal_markers = [
                    "rebut",
                    "question",
                    "challenge",
                    "disagree",
                    "correct",
                    "weak",
                    "missing",
                    "overreach",
                    "assumption",
                    "flaw",
                    "untested",
                    "incomplete",
                    "not enough",
                    "however",
                    "but ",
                    "risk",
                    "?",
                ]
                if not any(marker in challenge_text for marker in rebuttal_markers):
                    issues.append("competitive_no_rebuttal_or_question")
        if family == "cooperative" and advisor_count > 1 and advisor_index > 1 and has_previous_memory:
            if self._is_none_like(response_prev):
                issues.append("cooperative_no_response_to_previous")
        if family == "competitive" and len(allowed_items) >= 3 and candidate_views:
            buckets = [self._candidate_view_bucket(v.get("view")) for v in candidate_views]
            has_risk_or_mixed = any(x in {"risk", "mixed"} for x in buckets)
            has_discriminator = any(
                str(x or "").strip() and not self._is_none_like(x)
                for x in [key_tradeoff, comparison_reason, correction, questioned_assumption, challenge_prev]
            )
            if not has_risk_or_mixed and not has_discriminator:
                issues.append("no_discrimination_reason_missing")

        feedback = build_advisor_feedback(
            advisor_id=profile.get("u_raw", ""),
            advisor_type=profile.get("advisor_type", ""),
            stance=stance,
            endorsed_item=endorsed_item,
            support_reason=support_reason,
            oppose_reason=oppose_reason,
            confidence=int(round(float(profile.get("reliability", 0.0)) * 100)),
            solved_uncertainty=[],
            raw_text=f"speech={speech}; pos={pos}; decision={decision}; alt={alt_name}",
        )
        feedback["vote_action"] = stance
        feedback["defended_item"] = "" if stance == "evidence_only" else str(forced_item or endorsed_item)
        feedback["attacked_item"] = str(attacked_item or "")
        feedback["friend_basis"] = str(friend_basis or "")
        feedback["remove_set"] = list(remove_set)
        feedback["keep_set"] = list(keep_set)
        feedback["interested_set"] = list(interested_set)
        feedback["weak_fit_set"] = list(weak_fit_set)
        feedback["stronger_candidates"] = list(stronger_set)
        feedback["weaker_candidates"] = list(weaker_set)
        feedback["unclear_set"] = list(unclear_set)
        feedback["covered_set"] = list(covered_set)
        feedback["missing_set"] = list(missing_set)
        feedback["evidence_gap_set"] = list(evidence_gap_set)
        feedback["candidate_views"] = list(candidate_views)
        feedback["candidate_evidence_sources"] = list(candidate_evidence_sources)
        feedback["task_answer"] = str(task_answer or "")
        feedback["ask_user"] = str(ask_user or "")
        feedback["response_to_previous"] = str(response_prev or "")
        feedback["challenge_or_support_previous"] = str(challenge_prev or "")
        feedback["key_tradeoff"] = str(key_tradeoff or "")
        feedback["comparison_reason"] = str(comparison_reason or "")
        feedback["correction"] = str(correction or "")
        feedback["still_missing"] = str(still_missing or "")
        feedback["evidence_can_add"] = str(evidence_can_add or "")
        feedback["supplement_reason"] = str(supplement_reason or "")
        feedback["evidence_item"] = str(evidence_item or "")
        feedback["reviewed_item"] = str(evidence_item or "")
        feedback["supported_assumption"] = str(supported_assumption or "")
        feedback["questioned_assumption"] = str(questioned_assumption or "")
        feedback["decision_value"] = str(decision_norm or "")
        feedback["task_type"] = str(task_type or "")
        feedback["what"] = str(task_type or "")
        feedback["protocol"] = str(protocol or "")
        feedback["structured_fields"] = dict(structured_fields)
        feedback["extra_structured_fields"] = dict(extra_structured_fields)
        feedback["output_contract_fields"] = list(output_contract_fields)
        feedback["missing_output_contract_fields"] = list(missing_output_contract_fields)
        feedback["required_output_fields"] = list(output_contract_fields)
        feedback["missing_required_output_fields"] = list(missing_output_contract_fields)
        feedback["task_specific_fields"] = {
            key: value
            for key, value in extra_structured_fields.items()
            if not output_contract_keys or key in output_contract_keys
        }
        if missing_output_contract_fields:
            issues.append("missing_output_contract_fields")
        feedback["protocol_issues"] = sorted(set(issues))
        return feedback

    @staticmethod
    def _compact_text(text, max_len=260):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 3)].rstrip() + "..."

    @staticmethod
    def _argument_key(text):
        text = re.sub(r"\s+", " ", str(text or "").lower()).strip()
        text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _snake_field_name(field):
        text = re.sub(r"[^A-Za-z0-9]+", "_", str(field or "")).strip("_")
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
        return text.lower()

    @staticmethod
    def _dedupe_text_list(values):
        out = []
        seen = set()
        for value in values or []:
            if isinstance(value, dict):
                text = str(value.get("reason", "") or value.get("candidate", "") or value)
            else:
                text = str(value or "")
            key = re.sub(r"\s+", " ", text.strip().lower())
            if key and key not in seen:
                seen.add(key)
                out.append(value)
        return out

    def _summary_task_guidance(self, what_id, how_id):
        base_what = self._base_node_id(what_id)
        family = infer_communication_family(how_id)
        what_notes = {
            "compare_remaining_candidates": "Preserve the key comparison axis, candidate-level tradeoffs, and KeyTradeoff. Do not rank or pick a winner.",
            "reduce_hesitation_set": "Summarize only why candidates may be removed or down-ranked. Do not describe non-removed candidates as retained, keepable, or unanimously kept unless advisors explicitly produced such a field; absence from RemoveSet means no removal evidence was summarized.",
            "find_interested_subset": "Summarize why candidates may interest the requester. Do not invent weak-fit or unclear conclusions.",
            "reasoning_check": "Summarize whether the user's reason or assumptions are reliable, weak, mixed, or corrected.",
            "evidence_gap_check": "Summarize what evidence or supplementary reason is missing or added for the user's original reasoning.",
        }.get(base_what, "Summarize the selected communication task using only the advisor outputs.")
        how_notes = {
            "competitive": "Preserve rebuttals, questions, corrections, and unresolved candidate-level disagreement.",
            "cooperative": "Preserve complementary additions, integrations, refinements, and justified agreement.",
            "single": "Summarize the single advisor as evidence only; do not imply group consensus.",
        }.get(family, "Preserve how-specific interaction signals when present.")
        return {"what_guidance": what_notes, "how_guidance": how_notes}

    @staticmethod
    def _normalize_advisor_agreement(value):
        text = str(value or "").strip().lower().replace("-", "_")
        if not text or text in {"none", "no", "no_agreement", "unknown"}:
            return "none"
        if text in {"single", "single_advisor", "one", "one_advisor"}:
            return "single"
        if text in {"all", "unanimous", "unanimous_keep", "unanimous_remove", "all_advisors"}:
            return "all"
        if text in {"most", "majority", "majority_remove", "majority_keep"}:
            return "most"
        if text in {"some", "partial", "mixed", "several"}:
            return "some"
        return "some"

    def _slim_candidate_views(self, views, max_items=6):
        out = []
        for view in list(views or [])[:max_items]:
            view = dict(view or {})
            item = str(view.get("candidate", "") or "").strip()
            label = str(view.get("view", "") or "").strip()
            reason = self._compact_text(view.get("reason", ""), 220)
            if item or label or reason:
                out.append({"candidate": item, "view": label, "reason": reason})
        return out

    def _candidate_view_focus(self, feedbacks, focus_candidates=None, max_candidates=2):
        focus = {str(x).strip() for x in list(focus_candidates or []) if str(x or "").strip()}
        stats = {}
        for fb in list(feedbacks or []):
            advisor = str((fb or {}).get("advisor_type", "") or (fb or {}).get("advisor_id", "") or "advisor")
            for view in list((fb or {}).get("candidate_views", []) or []):
                view = dict(view or {})
                item = str(view.get("candidate", "") or "").strip()
                if not item or (focus and item not in focus):
                    continue
                label = str(view.get("view", "") or "unclear").strip()
                bucket = self._candidate_view_bucket(label)
                reason = self._compact_text(view.get("reason", ""), 160)
                row = stats.setdefault(
                    item,
                    {
                        "candidate": item,
                        "support": 0,
                        "risk": 0,
                        "mixed": 0,
                        "unclear": 0,
                        "advisor_count": 0,
                        "advisors": set(),
                        "reasons": [],
                    },
                )
                row["advisors"].add(advisor)
                row["advisor_count"] = len(row["advisors"])
                row[bucket if bucket in {"support", "risk", "mixed"} else "unclear"] += 1
                if reason:
                    row["reasons"].append(f"{label}: {reason}")
        rows = []
        for item, row in stats.items():
            total = int(row.get("support", 0) or 0) + int(row.get("risk", 0) or 0) + int(row.get("mixed", 0) or 0) + int(row.get("unclear", 0) or 0)
            disagreement = min(int(row.get("support", 0) or 0), int(row.get("risk", 0) or 0) + int(row.get("mixed", 0) or 0))
            rows.append(
                {
                    "candidate": item,
                    "advisor_count": int(row.get("advisor_count", 0) or 0),
                    "support": int(row.get("support", 0) or 0),
                    "risk": int(row.get("risk", 0) or 0),
                    "mixed": int(row.get("mixed", 0) or 0),
                    "unclear": int(row.get("unclear", 0) or 0),
                    "reasons": self._dedupe_text_list(row.get("reasons", []))[:2],
                    "_score": (int(row.get("advisor_count", 0) or 0) * 3) + disagreement + total,
                }
            )
        rows = sorted(rows, key=lambda x: (-int(x.get("_score", 0) or 0), str(x.get("candidate", ""))))
        for row in rows:
            row.pop("_score", None)
        return rows[:max_candidates]

    def _slim_candidate_views_for_focus(self, views, focus_candidates=None, max_items=2):
        focus = {str(x).strip() for x in list(focus_candidates or []) if str(x or "").strip()}
        selected = []
        fallback = []
        for view in list(views or []):
            view = dict(view or {})
            item = str(view.get("candidate", "") or "").strip()
            label = str(view.get("view", "") or "").strip()
            reason = self._compact_text(view.get("reason", ""), 130)
            if not (item or label or reason):
                continue
            row = {"candidate": item, "view": label, "reason": reason}
            if item in focus:
                selected.append(row)
            else:
                fallback.append(row)
        return (selected + fallback)[:max_items]

    def _compact_field_map(self, fields, max_fields=12, max_value_len=320):
        out = {}
        for key, value in list(dict(fields or {}).items())[:max_fields]:
            key = str(key or "").strip()
            if not key or self._is_none_like(value):
                continue
            if isinstance(value, (dict, list)):
                try:
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    value = str(value)
            out[key] = self._compact_text(value, max_value_len)
        return out

    def _compact_summary_value(self, value, max_items=2, max_value_len=220, depth=0):
        if isinstance(value, dict):
            out = {}
            for key, child in list(value.items())[:max_items + 2]:
                key = str(key or "").strip()
                if not key or self._is_none_like(child):
                    continue
                out[key] = self._compact_summary_value(
                    child,
                    max_items=max_items,
                    max_value_len=max_value_len,
                    depth=depth + 1,
                )
            return out
        if isinstance(value, list):
            return [
                self._compact_summary_value(
                    x,
                    max_items=max_items,
                    max_value_len=max_value_len,
                    depth=depth + 1,
                )
                for x in list(value or [])[:max_items]
                if not self._is_none_like(x)
            ]
        return self._compact_text(value, max_value_len)

    def _normalize_decision_guidance(self, value, focus_candidates=None):
        value = dict(value or {}) if isinstance(value, dict) else {}
        can_finalize = str(value.get("can_finalize", "unclear") or "unclear").strip().lower()
        if can_finalize not in {"yes", "no", "unclear"}:
            can_finalize = "unclear"
        next_action = str(value.get("next_action", "none") or "none").strip().lower()
        allowed_actions = {
            "final",
            "continue_compare",
            "continue_fill_gap",
            "remove_candidates",
            "clarify_user_preference",
            "none",
        }
        if next_action not in allowed_actions:
            next_action = "none"
        allowed_focus = {str(x).strip() for x in list(focus_candidates or []) if str(x or "").strip()}
        focus = []
        for item in list(value.get("focus_candidates", []) or [])[:4]:
            item = str(item or "").strip()
            if item and (not allowed_focus or item in allowed_focus) and item not in focus:
                focus.append(item)
        return {
            "can_finalize": can_finalize,
            "next_action": next_action,
            "reason": self._compact_text(value.get("reason", ""), 220),
            "focus_candidates": focus[:3],
        }

    def _summary_hints_from_path(self, path):
        payload = dict((path or {}).get("path_skill_payload", {}) or {})

        def hints_for(level):
            node = payload.get(level, {})
            if not isinstance(node, dict):
                return {}
            hints = node.get("summary_hints", {})
            return dict(hints or {}) if isinstance(hints, dict) else {}

        what_hints = hints_for("what")
        how_hints = hints_for("how")
        important_fields = []
        preserve_fields = []
        allowed_preserve = {
            "ChallengeOrSupportPrevious",
            "ResponseToPrevious",
            "Correction",
            "PerCandidateValidation",
        }
        for hints in [what_hints, how_hints]:
            important_fields.extend(
                self._clean_summary_hint_fields(
                    hints.get("important_output_fields", []),
                    allowed=None,
                    limit=8,
                )
            )
            preserve_fields.extend(
                self._clean_summary_hint_fields(
                    hints.get("preserve_interaction_fields", []),
                    allowed=allowed_preserve,
                    limit=4,
                )
            )
        important_fields = self._clean_summary_hint_fields(important_fields, allowed=None, limit=8)
        preserve_fields = self._clean_summary_hint_fields(preserve_fields, allowed=allowed_preserve, limit=4)
        what_hints = dict(what_hints or {})
        how_hints = dict(how_hints or {})
        if "important_output_fields" in what_hints:
            what_hints["important_output_fields"] = self._clean_summary_hint_fields(what_hints.get("important_output_fields", []), limit=8)
        if "preserve_interaction_fields" in what_hints:
            what_hints["preserve_interaction_fields"] = self._clean_summary_hint_fields(
                what_hints.get("preserve_interaction_fields", []),
                allowed=allowed_preserve,
                limit=4,
            )
        if "important_output_fields" in how_hints:
            how_hints["important_output_fields"] = self._clean_summary_hint_fields(how_hints.get("important_output_fields", []), limit=8)
        if "preserve_interaction_fields" in how_hints:
            how_hints["preserve_interaction_fields"] = self._clean_summary_hint_fields(
                how_hints.get("preserve_interaction_fields", []),
                allowed=allowed_preserve,
                limit=4,
            )
        return {
            "what": what_hints,
            "how": how_hints,
            "important_output_fields": important_fields,
            "preserve_interaction_fields": preserve_fields,
        }

    def _build_summary_agent_input(
        self,
        proposal_name,
        feedbacks,
        focus_candidates=None,
        path=None,
        evidence_summary=None,
    ):
        path = dict(path or {})
        focus = [str(x) for x in (focus_candidates or []) if str(x or "").strip()]
        node_summary_hints = self._summary_hints_from_path(path)
        candidate_view_focus = self._candidate_view_focus(feedbacks, focus_candidates=focus, max_candidates=2)
        focus_names = [str(row.get("candidate", "") or "") for row in candidate_view_focus if str(row.get("candidate", "") or "").strip()]
        slim_feedbacks = []
        for fb in list(feedbacks or [])[:4]:
            fb = dict(fb or {})
            interaction_note = (
                fb.get("challenge_or_support_previous", "")
                or fb.get("response_to_previous", "")
                or ""
            )
            slim_feedbacks.append(
                {
                    "advisor_type": str(fb.get("advisor_type", "") or ""),
                    "candidate_views": self._slim_candidate_views_for_focus(fb.get("candidate_views", []), focus_candidates=focus_names, max_items=2),
                    "task_answer": self._compact_text(fb.get("task_answer", ""), 220),
                    "task_specific_fields": self._compact_field_map(fb.get("task_specific_fields", {}) or fb.get("extra_structured_fields", {}), max_fields=4, max_value_len=160),
                    "correction": self._compact_text(fb.get("correction", ""), 140),
                    "interaction": self._compact_text(interaction_note, 160),
                    "missing_output_contract_fields": list(fb.get("missing_output_contract_fields", []) or fb.get("missing_required_output_fields", []) or [])[:4],
                    "missing_required_output_fields": list(fb.get("missing_required_output_fields", []) or [])[:4],
                    "protocol_issues": list(fb.get("protocol_issues", []) or [])[:4],
                }
            )
        return {
            "selected_path": {
                'why': str(path.get('why', "") or ""),
                "what": str(path.get("what", "") or path.get("legacy_what", "") or ""),
                "how": str(path.get("how", "") or ""),
                "who": str(path.get("who", "") or ""),
                "who_branch": str(path.get("who_branch", "") or ""),
            },
            "task_context": {
                "user_task": self._compact_text(path.get("user_task", ""), 280),
                "expected_output": self._compact_text(path.get("expected_output", ""), 120),
                "task_type_hint": str(path.get("task_type_hint", "") or ""),
                "task_targets": list(path.get("task_targets", []) or [])[:5],
            },
            "summary_guidance": self._summary_task_guidance(path.get("what", "") or path.get("legacy_what", ""), path.get("how", "")),
            "node_summary_hints": node_summary_hints,
            "proposal_item": str(proposal_name or ""),
            "focus_candidates": list(focus),
            "candidate_view_focus": candidate_view_focus,
            "advisor_feedbacks": slim_feedbacks,
            "diagnostic_context": {
                "discussion_result": str((evidence_summary or {}).get("discussion_result", "") or ""),
                "protocol_issues": list((evidence_summary or {}).get("protocol_issues", []) or [])[:5],
                "silent_focus_candidates": list((evidence_summary or {}).get("silent_focus_candidates", []) or [])[:4],
            },
        }

    def _default_synthesis_packet(self, focus_candidates=None):
        return {
            "decision_policy": "information_only_no_vote",
            "source": "advisor_summary_agent_v1",
            "what_was_answered": "",
            "candidate_summaries": {
                str(item): {
                    "support_summary": "",
                    "risk_summary": "",
                    "tradeoff_summary": "",
                    "advisor_agreement": "none",
                    "key_evidence": [],
                }
                for item in list(focus_candidates or [])
                if str(item or "").strip()
            },
            "task_specific_summary": {
                "compare_key_tradeoff": "",
                "remove_reasons": [],
                "interest_reasons": [],
                "reasoning_checks": [],
                "evidence_gaps": [],
            },
            "decision_guidance": {
                "can_finalize": "unclear",
                "next_action": "none",
                "reason": "",
                "focus_candidates": [],
            },
            "extra_task_summary": {},
            "interaction_summary": {
                "main_agreements": [],
                "main_disagreements": [],
                "corrections_or_rebuttals": [],
                "unresolved_conflicts": [],
            },
            "extra_interaction_summary": {},
            "remaining_uncertainty": [],
            "do_not_decide_winner": True,
        }

    def _summary_agent_schema(self, summary_input, focus_candidates=None):
        summary_input = dict(summary_input or {})
        selected_path = dict(summary_input.get("selected_path", {}) or {})
        base_what = self._base_node_id(selected_path.get("what", ""))
        family = infer_communication_family(selected_path.get("how", ""))
        row_by_what = {
            "compare_remaining_candidates": {
                "tradeoff_summary": "",
                "advisor_agreement": "none|single|some|most|all",
                "key_evidence": [],
            },
            "reduce_hesitation_set": {
                "risk_summary": "",
                "advisor_agreement": "none|single|some|most|all",
                "key_evidence": [],
            },
            "find_interested_subset": {
                "support_summary": "",
                "advisor_agreement": "none|single|some|most|all",
                "key_evidence": [],
            },
            "reasoning_check": {
                "support_summary": "",
                "risk_summary": "",
                "advisor_agreement": "none|single|some|most|all",
                "key_evidence": [],
            },
            "evidence_gap_check": {
                "tradeoff_summary": "",
                "advisor_agreement": "none|single|some|most|all",
                "key_evidence": [],
            },
        }
        task_schema_by_what = {
            "compare_remaining_candidates": {"compare_key_tradeoff": "", "evidence_gaps": []},
            "reduce_hesitation_set": {"remove_reasons": [], "evidence_gaps": []},
            "find_interested_subset": {"interest_reasons": [], "evidence_gaps": []},
            "reasoning_check": {"reasoning_checks": [], "evidence_gaps": []},
            "evidence_gap_check": {"evidence_gaps": [], "reasoning_checks": []},
        }
        interaction_by_family = {
            "competitive": {"main_disagreements": [], "corrections_or_rebuttals": [], "unresolved_conflicts": []},
            "cooperative": {"main_agreements": [], "main_disagreements": [], "unresolved_conflicts": []},
            "single": {},
        }
        schema = {
            "what_was_answered": "",
            "candidate_summaries": {"<candidate from focus_candidates>": dict(row_by_what.get(base_what, row_by_what["compare_remaining_candidates"]))},
            "task_specific_summary": dict(task_schema_by_what.get(base_what, {"evidence_gaps": []})),
            "decision_guidance": {
                "can_finalize": "yes|no|unclear",
                "next_action": "final|continue_compare|continue_fill_gap|remove_candidates|clarify_user_preference|none",
                "reason": "",
                "focus_candidates": [],
            },
            "remaining_uncertainty": [],
        }
        interaction_schema = dict(interaction_by_family.get(family, {"unresolved_conflicts": []}))
        if interaction_schema:
            schema["interaction_summary"] = interaction_schema

        hints = dict(summary_input.get("node_summary_hints", {}) or {})
        default_task_keys = set(schema["task_specific_summary"].keys())
        extra_task_fields = []
        important_hint_fields = self._clean_summary_hint_fields(
            hints.get("important_output_fields", []),
            allowed=None,
            limit=3,
        )
        for field in important_hint_fields:
            key = self._snake_field_name(field)
            if key and key not in default_task_keys:
                extra_task_fields.append(key)
        if extra_task_fields:
            schema["extra_task_summary"] = {key: "" for key in extra_task_fields[:3]}

        default_interaction_keys = set(dict(schema.get("interaction_summary", {}) or {}).keys())
        extra_interaction_fields = []
        allowed_preserve = {
            "ChallengeOrSupportPrevious",
            "ResponseToPrevious",
            "Correction",
            "PerCandidateValidation",
        }
        preserve_hint_fields = self._clean_summary_hint_fields(
            hints.get("preserve_interaction_fields", []),
            allowed=allowed_preserve,
            limit=2,
        )
        for field in preserve_hint_fields:
            key = self._snake_field_name(field)
            if key and key not in default_interaction_keys:
                extra_interaction_fields.append(key)
        if extra_interaction_fields:
            schema["extra_interaction_summary"] = {key: "" for key in extra_interaction_fields[:2]}
        return schema

    def _parse_advisor_synthesis_packet(self, raw_response, focus_candidates=None):
        text = str(raw_response or "").strip()
        payload = None
        if text:
            try:
                payload = json.loads(text)
            except Exception:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    try:
                        payload = json.loads(match.group(0))
                    except Exception:
                        payload = None
        if not isinstance(payload, dict):
            return None
        packet = self._default_synthesis_packet(focus_candidates=focus_candidates)
        packet.update({k: v for k, v in payload.items() if k in packet})
        packet["decision_policy"] = "information_only_no_vote"
        packet["source"] = "advisor_summary_agent_v1"
        packet["do_not_decide_winner"] = True
        candidate_summaries = payload.get("candidate_summaries", {})
        if isinstance(candidate_summaries, dict):
            cleaned = {}
            allowed = {str(x).strip() for x in list(focus_candidates or []) if str(x or "").strip()}
            for item, row in candidate_summaries.items():
                if str(item or "").strip().lower() in {"<candidate from focus_candidates>", "candidate_name", "<candidate>"}:
                    continue
                item = str(item or "").strip()
                if not item or (allowed and item not in allowed):
                    continue
                row = dict(row or {}) if isinstance(row, dict) else {"support_summary": str(row or "")}
                has_content = any(
                    str(row.get(key, "") or "").strip()
                    for key in ["support_summary", "risk_summary", "tradeoff_summary"]
                ) or any(str(x or "").strip() for x in list(row.get("key_evidence", []) or []))
                if not has_content:
                    continue
                cleaned[item] = {
                    "support_summary": self._compact_text(row.get("support_summary", ""), 260),
                    "risk_summary": self._compact_text(row.get("risk_summary", ""), 260),
                    "tradeoff_summary": self._compact_text(row.get("tradeoff_summary", ""), 260),
                    "advisor_agreement": self._normalize_advisor_agreement(row.get("advisor_agreement", "none")),
                    "key_evidence": [self._compact_text(x, 180) for x in list(row.get("key_evidence", []) or [])[:2] if str(x or "").strip()],
                }
            packet["candidate_summaries"] = cleaned
        if not isinstance(packet.get("task_specific_summary", {}), dict):
            packet["task_specific_summary"] = {}
        if not isinstance(packet.get("extra_task_summary", {}), dict):
            packet["extra_task_summary"] = {}
        if not isinstance(packet.get("interaction_summary", {}), dict):
            packet["interaction_summary"] = {}
        if not isinstance(packet.get("extra_interaction_summary", {}), dict):
            packet["extra_interaction_summary"] = {}
        known_keys = set(packet.keys()) | {"candidate_summaries"}
        for key, value in payload.items():
            key = str(key or "").strip()
            if not key or key in known_keys or self._is_none_like(value):
                continue
            packet["extra_task_summary"][self._snake_field_name(key)] = value
        packet["task_specific_summary"] = self._compact_summary_value(
            packet.get("task_specific_summary", {}),
            max_items=5,
            max_value_len=220,
        )
        packet["decision_guidance"] = self._normalize_decision_guidance(
            packet.get("decision_guidance", {}),
            focus_candidates=focus_candidates,
        )
        packet["extra_task_summary"] = self._compact_field_map(packet.get("extra_task_summary", {}), max_fields=6, max_value_len=260)
        packet["extra_interaction_summary"] = self._compact_field_map(packet.get("extra_interaction_summary", {}), max_fields=4, max_value_len=260)
        for key in ["main_agreements", "main_disagreements", "corrections_or_rebuttals", "unresolved_conflicts"]:
            bucket = dict(packet.get("interaction_summary", {}) or {}).get(key, [])
            packet.setdefault("interaction_summary", {})[key] = [self._compact_text(x, 180) for x in list(bucket or [])[:3] if str(x or "").strip()]
        packet["remaining_uncertainty"] = [self._compact_text(x, 160) for x in list(packet.get("remaining_uncertainty", []) or [])[:4] if str(x or "").strip()]
        return packet

    def _build_minimal_synthesis_fallback(self, summary_input, evidence_summary=None, reason="fallback"):
        summary_input = dict(summary_input or {})
        focus = list(summary_input.get("focus_candidates", []) or [])
        packet = self._default_synthesis_packet(focus_candidates=focus)
        packet["source"] = "advisor_summary_agent_fallback_v1"
        packet["what_was_answered"] = "Summary agent unavailable; rebuilt a compact evidence packet from raw advisor feedback."
        packet["candidate_summaries"] = {}
        by_candidate = {
            str(item): {"support": [], "risk": [], "tradeoff": [], "evidence": []}
            for item in focus
            if str(item or "").strip()
        }
        agreements = []
        disagreements = []
        corrections = []
        task_notes = []
        allowed_preserve = {
            "ChallengeOrSupportPrevious",
            "ResponseToPrevious",
            "Correction",
            "PerCandidateValidation",
        }
        preserve_keys = {
            self._snake_field_name(x)
            for x in self._clean_summary_hint_fields(
                (summary_input.get("node_summary_hints", {}) or {}).get("preserve_interaction_fields", []),
                allowed=allowed_preserve,
                limit=4,
            )
        }
        for fb in list(summary_input.get("advisor_feedbacks", []) or [])[:6]:
            fb = dict(fb or {})
            advisor = str(fb.get("advisor_type", "") or fb.get("advisor_id", "") or "advisor")
            task_answer = str(fb.get("task_answer", "") or "")
            correction = str(fb.get("correction", "") or "")
            fallback_reason = task_answer or correction
            for view in list(fb.get("candidate_views", []) or [])[:6]:
                view = dict(view or {})
                item = str(view.get("candidate", "") or "").strip()
                if item and item in by_candidate:
                    label = str(view.get("view", "") or "unclear")
                    reason = str(view.get("reason", "") or fallback_reason or label)
                    text = f"{advisor}: {label} - {reason}"
                    bucket = self._candidate_view_bucket(label)
                    if bucket == "risk":
                        by_candidate[item]["risk"].append(text)
                    elif bucket == "support":
                        by_candidate[item]["support"].append(text)
                    else:
                        by_candidate[item]["tradeoff"].append(text)
                    by_candidate[item]["evidence"].append(text)
            for item in list(fb.get("remove_set", []) or [])[:4]:
                if str(item) in by_candidate:
                    text = f"{advisor}: remove - {fb.get('risk_reason', '') or task_answer}"
                    by_candidate[str(item)]["risk"].append(text)
                    by_candidate[str(item)]["evidence"].append(text)
            for item in list(fb.get("interested_set", []) or [])[:4]:
                if str(item) in by_candidate:
                    text = f"{advisor}: interest - {fb.get('interest_reason', '') or task_answer}"
                    by_candidate[str(item)]["support"].append(text)
                    by_candidate[str(item)]["evidence"].append(text)
            if task_answer:
                task_notes.append(f"{advisor}: {task_answer}")
            interaction = str(fb.get("interaction", "") or "")
            if interaction:
                disagreements.append(f"{advisor}: {interaction}")
            if correction:
                corrections.append(f"{advisor}: {correction}")
                packet["task_specific_summary"]["reasoning_checks"].append(self._compact_text(f"{advisor}: {correction}", 240))
            if fb.get("key_tradeoff"):
                packet["task_specific_summary"]["compare_key_tradeoff"] = self._compact_text(fb.get("key_tradeoff", ""), 260)
            for key, value in dict(fb.get("extra_structured_fields", {}) or {}).items():
                key = str(key or "").strip()
                if not key or self._is_none_like(value):
                    continue
                target = packet["extra_interaction_summary"] if key in preserve_keys else packet["extra_task_summary"]
                target[key] = self._compact_text(value, 260)
        candidate_focus = [
            str(row.get("candidate", "") or "")
            for row in list(summary_input.get("candidate_view_focus", []) or [])
            if str(row.get("candidate", "") or "").strip()
        ][:2]
        if not candidate_focus:
            scored = []
            for item, buckets in by_candidate.items():
                evidence_count = sum(len(buckets[key]) for key in ["support", "risk", "tradeoff", "evidence"])
                if evidence_count:
                    scored.append((evidence_count, item))
            candidate_focus = [item for _score, item in sorted(scored, key=lambda kv: (-kv[0], kv[1]))[:2]]
        for item in candidate_focus:
            buckets = by_candidate.get(item, {"support": [], "risk": [], "tradeoff": [], "evidence": []})
            support = self._dedupe_text_list([self._compact_text(x, 180) for x in buckets["support"] if str(x or "").strip()])[:2]
            risk = self._dedupe_text_list([self._compact_text(x, 180) for x in buckets["risk"] if str(x or "").strip()])[:2]
            tradeoff = self._dedupe_text_list([self._compact_text(x, 180) for x in buckets["tradeoff"] if str(x or "").strip()])[:2]
            evidence = self._dedupe_text_list([self._compact_text(x, 180) for x in buckets["evidence"] if str(x or "").strip()])[:2]
            if not any([support, risk, tradeoff, evidence]):
                continue
            packet["candidate_summaries"][item] = {
                "key_evidence": evidence,
                "support_summary": " | ".join(support),
                "risk_summary": " | ".join(risk),
                "tradeoff_summary": " | ".join(tradeoff),
                "advisor_agreement": "some" if len(evidence) >= 2 else "single" if evidence else "none",
            }
        packet["interaction_summary"]["main_agreements"] = self._dedupe_text_list(agreements)[:3]
        packet["interaction_summary"]["main_disagreements"] = self._dedupe_text_list(disagreements)[:3]
        packet["interaction_summary"]["corrections_or_rebuttals"] = self._dedupe_text_list(corrections + disagreements)[:3]
        packet["remaining_uncertainty"] = [
            self._compact_text(x, 160)
            for x in list((evidence_summary or {}).get("unresolved_questions", []) or [])[:4]
            if str(x or "").strip()
        ]
        protocol_issues = list((evidence_summary or {}).get("protocol_issues", []) or [])
        if protocol_issues or packet["remaining_uncertainty"]:
            next_action = "continue_fill_gap"
            reason_text = "advisor evidence is incomplete or unresolved; ask the next advisors to close the key gap."
        elif len(packet["candidate_summaries"]) >= 2:
            next_action = "continue_compare"
            reason_text = "advisor evidence covers multiple plausible candidates; compare the most decision-critical tradeoff."
        elif len(packet["candidate_summaries"]) == 1:
            next_action = "none"
            reason_text = "only one candidate has compact advisor evidence; avoid treating fallback evidence as a vote."
        else:
            next_action = "continue_fill_gap"
            reason_text = "no reliable candidate-level evidence was recovered from advisor feedback."
        packet["decision_guidance"] = self._normalize_decision_guidance(
            {
                "can_finalize": "unclear",
                "next_action": next_action,
                "reason": reason_text,
                "focus_candidates": list(packet["candidate_summaries"].keys())[:2],
            },
            focus_candidates=focus,
        )
        packet["summary_agent_failed"] = True
        packet["summary_agent_fallback_reason"] = str(reason or "fallback")
        if task_notes:
            packet["what_was_answered"] = self._compact_text("Actionable advisor signal: " + " | ".join(task_notes[:3]), 420)
        return packet

    def _run_advisor_summary_agent(self, summary_input, focus_candidates=None, evidence_summary=None):
        runtime_args = copy.copy(_ensure_runtime_args(self.args))
        setattr(runtime_args, "max_tokens", int(getattr(runtime_args, "com_summary_max_tokens", 2200) or 2200))
        setattr(runtime_args, "max_retry_num", 0)
        system_prompt = (
            "You are a decision-evidence organizer for the target user after advisor communication.\n"
            "Your job is not to write meeting minutes. Convert advisor claims into a compact evidence packet that helps the user decide the next step.\n"
            "You may give process guidance in decision_guidance: final, continue_compare, continue_fill_gap, remove_candidates, clarify_user_preference, or none.\n"
            "Use decision_guidance.next_action=final only when advisor evidence covers the important focus candidates and no decision-critical uncertainty remains.\n"
            "Do not vote, rank candidates, choose a winning Item, or output candidate_scores.\n"
            "Prefer decision-critical evidence over completeness. Omit weak details rather than producing long or incomplete JSON.\n"
            "Do not summarize every CandidateView. Use candidate_view_focus and include only the 1-2 candidates with strongest advisor agreement, disagreement, risk, or task-specific importance.\n"
            "Leave candidate_summaries empty for candidates without important new evidence; do not fill the whole HesitationSet.\n"
            "Merge repeated claims across advisors into one concise point. Preserve task-specific information from the selected what node only when it changes the next decision.\n"
            "For reasoning_check: state which user reasons are supported, weak, corrected, or still uncertain.\n"
            "For evidence_gap_check: state the most important missing evidence and what should be checked next.\n"
            "For compare_remaining_candidates: state the decisive tradeoff and what remains unresolved.\n"
            "For reduce_hesitation_set: state which candidates should be down-ranked or removed and why.\n"
            "For find_interested_subset: state which candidates are worth keeping in focus and why.\n"
            "Preserve disagreement, rebuttal, correction, and qualification from the selected how node only when it affects the user's next step.\n"
            "advisor_agreement must be exactly one of: none, single, some, most, all.\n"
            "Use real focus candidate names as candidate_summaries keys; do not keep placeholder keys.\n"
            "Use node_summary_hints.important_output_fields and preserve_interaction_fields to decide which evolved-node fields matter.\n"
            "The selected what node determines the task_specific_summary content; do not force every what into the same summary shape.\n"
            "Hard limits: at most 2 candidate_summaries, at most 1 key_evidence per candidate, at most 2 points in each task/interaction list, strings under 160 characters.\n"
            "Valid closed JSON is more important than including every detail.\n"
            "Return only valid JSON matching the requested schema. No markdown fences."
        )
        schema = self._summary_agent_schema(summary_input, focus_candidates=focus_candidates)
        user_prompt = (
            "Build a compact FinalAdvisorEvidencePacket for user redecision.\n"
            "Focus on what the advisor discussion implies the user should do next, without choosing the final Item.\n\n"
            "SummaryAgentInput:\n"
            f"{json.dumps(summary_input, ensure_ascii=False, separators=(',', ':'))}\n\n"
            "Return JSON schema:\n"
            f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
        )
        update_llm_prompt_trace_context(phase="advisor_summary_agent")
        last_raw = ""
        try:
            raw = llm_request(system_prompt, user_prompt, runtime_args)
            last_raw = str(raw or "")
            usage = get_last_llm_request_usage()
            completion_tokens = int((usage or {}).get("completion_tokens", 0) or 0)
            max_completion_tokens = int(getattr(runtime_args, "max_tokens", 2200) or 2200)
            if completion_tokens > max_completion_tokens + 128 or len(last_raw) > 9000:
                reason = (
                    f"summary_agent_output_too_long: completion_tokens={completion_tokens}, "
                    f"response_chars={len(last_raw)}"
                )
                return self._build_minimal_synthesis_fallback(
                    summary_input,
                    evidence_summary=evidence_summary,
                    reason=reason,
                ), "fallback_too_long", last_raw[:1200]
            packet = self._parse_advisor_synthesis_packet(raw, focus_candidates=focus_candidates)
            if packet:
                return packet, "ok", last_raw
        except Exception as exc:
            last_raw = repr(exc)
        return self._build_minimal_synthesis_fallback(summary_input, evidence_summary=evidence_summary, reason=last_raw), "fallback", last_raw

    def _append_unique_argument(self, bucket, text, advisor="", protocol="", max_items=4, removed=None):
        text = self._compact_text(text)
        if not text:
            return False
        key = self._argument_key(text)
        if len(key) < 18:
            if removed is not None:
                removed.append({"reason": text, "why": "too_short_or_empty"})
            return False
        existing = {self._argument_key(row.get("reason", "")) for row in bucket}
        if key in existing:
            if removed is not None:
                removed.append({"reason": text, "why": "duplicate"})
            return False
        if len(bucket) >= max_items:
            if removed is not None:
                removed.append({"reason": text, "why": "overflow"})
            return False
        bucket.append(
            {
                "reason": text,
                "advisor": str(advisor or ""),
                "protocol": str(protocol or ""),
            }
        )
        return True

    @staticmethod
    def _is_question_or_gap_text(text):
        text = str(text or "").strip()
        if not text:
            return False
        lower = text.lower()
        return (
            "?" in text
            or lower.startswith(("can you", "could you", "please", "do you", "would you", "are you"))
            or "provide more details" in lower
            or "need to know" in lower
            or "needs more" in lower
            or "still missing" in lower
            or "remaining gap" in lower
        )

    @staticmethod
    def _clean_candidate_reason_fragment(text):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return ""
        parts = [p.strip() for p in re.split(r"\s+\|\s+", text) if p.strip()]
        if not parts:
            parts = [text]
        kept = []
        for part in parts:
            lower = part.lower()
            if (
                "?" in part
                or lower.startswith(("can you", "could you", "please", "do you", "would you", "are you"))
                or "provide more details" in lower
                or "this will help" in lower
            ):
                continue
            kept.append(part)
        return " | ".join(kept).strip()

    def _candidate_reason_text(self, entry, max_len=320):
        reasons = []
        for bucket in ["support", "against", "unclear"]:
            for arg in list((entry or {}).get(bucket, []) or [])[:3]:
                text = self._clean_candidate_reason_fragment(arg.get("reason", ""))
                if not text or self._is_question_or_gap_text(text):
                    continue
                if text not in reasons:
                    reasons.append(text)
                if len(reasons) >= 3:
                    break
            if len(reasons) >= 3:
                break
        return self._compact_text(" | ".join(reasons), max_len=max_len) if reasons else ""

    def _build_evidence_summary(self, proposal_name, advisor_arguments, focus_candidates=None, protocol_issues=None):
        focus = [str(x) for x in (focus_candidates or []) if str(x or "").strip()]
        protocol_issues = list(protocol_issues or [])
        by_candidate = {
            item: {
                "support": [],
                "against": [],
                "unclear": [],
                "support_advisors": [],
                "against_advisors": [],
                "unclear_advisors": [],
                "defended_count": 0,
                "attacked_count": 0,
                "unclear_count": 0,
            }
            for item in focus
        }
        removed = []
        key_conflicts = []
        useful_unique_arguments = []
        unresolved_advisor_answers = []
        advisor_interactions = []
        comparative_claims = []
        correction_claims = []

        def ensure_item(item):
            item = str(item or "").strip()
            if not item:
                return ""
            if focus and item not in by_candidate:
                return ""
            by_candidate.setdefault(
                item,
                {
                    "support": [],
                    "against": [],
                    "unclear": [],
                    "support_advisors": [],
                    "against_advisors": [],
                    "unclear_advisors": [],
                    "defended_count": 0,
                    "attacked_count": 0,
                    "unclear_count": 0,
                },
            )
            return item

        def append_labeled(bucket, label, text, advisor="", protocol="", max_items=8):
            text = self._compact_text(text, 320)
            if not text:
                return False
            key = (str(label or "").strip().lower(), self._argument_key(text))
            existing = {
                (str(row.get("type", "") or "").strip().lower(), self._argument_key(row.get("reason", "")))
                for row in bucket
            }
            if key in existing or len(bucket) >= max_items:
                return False
            bucket.append(
                {
                    "type": str(label or ""),
                    "reason": text,
                    "advisor": str(advisor or ""),
                    "protocol": str(protocol or ""),
                }
            )
            return True

        for row in advisor_arguments or []:
            advisor = str(row.get("advisor", "") or "")
            protocol = str(row.get("protocol", "") or "")
            stance = str(row.get("stance", "") or "")
            endorsed = ensure_item(row.get("endorsed_item") or row.get("defended_item") or ("" if stance == "evidence_only" else proposal_name))
            attacked = ensure_item(row.get("attacked_item") or "")
            reason_for = str(row.get("reason_for", "") or "")
            reason_against = str(row.get("reason_against", "") or "")
            friend_basis = str(row.get("friend_basis", "") or "")
            evidence_can_add = str(row.get("evidence_can_add", "") or "")
            evidence_item = ensure_item(row.get("evidence_item") or row.get("reviewed_item") or "")
            supported_assumption = str(row.get("supported_assumption", "") or "")
            questioned_assumption = str(row.get("questioned_assumption", "") or "")
            candidate_views = list(row.get("candidate_views", []) or [])
            if not candidate_views:
                for source in list(row.get("candidate_evidence_sources", []) or []):
                    source = dict(source or {})
                    source_label = str(source.get("source_label", "") or source.get("label", "") or "unclear")
                    candidate_views.append(
                        {
                            "candidate": str(source.get("candidate", "") or ""),
                            "view": source_label,
                            "reason": str(source.get("reason", "") or ""),
                        }
                    )
            ask_user = str(row.get("ask_user", "") or "")
            task_answer = str(row.get("task_answer", "") or "")
            still_missing = str(row.get("still_missing", "") or "")
            response_prev = str(row.get("response_to_previous", "") or "")
            challenge_prev = str(row.get("challenge_or_support_previous", "") or "")
            key_tradeoff = str(row.get("key_tradeoff", "") or "")
            comparison_reason = str(row.get("comparison_reason", "") or "")
            correction = str(row.get("correction", "") or "")
            support_set = []
            for key in ["keep_set", "interested_set", "stronger_candidates", "covered_set"]:
                for item in list(row.get(key, []) or []):
                    item = ensure_item(item)
                    if item and item not in support_set:
                        support_set.append(item)
            against_set = []
            for key in ["remove_set", "weak_fit_set", "weaker_candidates"]:
                for item in list(row.get(key, []) or []):
                    item = ensure_item(item)
                    if item and item not in against_set:
                        against_set.append(item)
            unclear_set = []
            for key in ["unclear_set", "missing_set", "evidence_gap_set"]:
                for item in list(row.get(key, []) or []):
                    item = ensure_item(item)
                    if item and item not in unclear_set:
                        unclear_set.append(item)

            for view in candidate_views:
                item = ensure_item((view or {}).get("candidate", ""))
                if not item:
                    continue
                view_label = str((view or {}).get("view", "") or "unclear")
                reason = str((view or {}).get("reason", "") or task_answer or reason_for or reason_against or "").strip()
                bucket = self._candidate_view_bucket(view_label)
                entry = by_candidate[item]
                if bucket in {"support", "mixed"}:
                    entry["defended_count"] += 1
                    if advisor and advisor not in entry["support_advisors"]:
                        entry["support_advisors"].append(advisor)
                    self._append_unique_argument(entry["support"], reason or view_label, advisor=advisor, protocol=protocol, removed=removed)
                if bucket in {"risk", "mixed"}:
                    entry["attacked_count"] += 1
                    if advisor and advisor not in entry["against_advisors"]:
                        entry["against_advisors"].append(advisor)
                    self._append_unique_argument(entry["against"], reason or view_label, advisor=advisor, protocol=protocol, removed=removed)
                if bucket == "unclear":
                    entry["unclear_count"] += 1
                    if advisor and advisor not in entry["unclear_advisors"]:
                        entry["unclear_advisors"].append(advisor)
                    self._append_unique_argument(entry["unclear"], reason or view_label, advisor=advisor, protocol=protocol, max_items=4, removed=removed)
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"{item} | {view_label}: {reason or 'candidate-level observation'}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=16,
                )

            if friend_basis:
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Advisor history basis: {friend_basis}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=12,
                )
            if evidence_can_add:
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Evidence added: {evidence_can_add}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=12,
                )
            if ask_user:
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Advisor question for user: {ask_user}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=12,
                )
            if still_missing:
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Still missing: {still_missing}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=12,
                )
            if response_prev:
                append_labeled(advisor_interactions, "ResponseToPrevious", response_prev, advisor=advisor, protocol=protocol)
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Response to previous advisor: {response_prev}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=16,
                )
            if challenge_prev:
                append_labeled(advisor_interactions, "ChallengeOrSupportPrevious", challenge_prev, advisor=advisor, protocol=protocol)
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Cross-advisor response: {challenge_prev}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=16,
                )
                challenge_low = challenge_prev.lower()
                if any(
                    marker in challenge_low
                    for marker in [
                        "rebut",
                        "question",
                        "challenge",
                        "disagree",
                        "correct",
                        "weak",
                        "missing",
                        "overreach",
                        "however",
                        "but ",
                        "risk",
                        "?",
                    ]
                ):
                    self._append_unique_argument(
                        key_conflicts,
                        f"Cross-advisor challenge: {challenge_prev}",
                        advisor=advisor,
                        protocol=protocol,
                        max_items=8,
                    )
            if key_tradeoff:
                append_labeled(comparative_claims, "KeyTradeoff", key_tradeoff, advisor=advisor, protocol=protocol)
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Key tradeoff: {key_tradeoff}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=16,
                )
            if comparison_reason:
                append_labeled(comparative_claims, "ComparisonReason", comparison_reason, advisor=advisor, protocol=protocol)
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Comparison reason: {comparison_reason}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=16,
                )
            if correction:
                append_labeled(correction_claims, "Correction", correction, advisor=advisor, protocol=protocol)
                self._append_unique_argument(
                    key_conflicts,
                    f"Advisor correction: {correction}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=8,
                )
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Advisor correction: {correction}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=16,
                )

            if stance == "unresolved" and (reason_for or reason_against):
                answer_text = reason_for or reason_against
                unresolved_advisor_answers.append(
                    {
                        "reason": self._compact_text(answer_text, 220),
                        "advisor": advisor,
                        "protocol": protocol,
                    }
                )
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"Unresolved advisor answer: {answer_text}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=12,
                )

            if evidence_item and supported_assumption:
                entry = by_candidate[evidence_item]
                entry["defended_count"] += 1
                if advisor and advisor not in entry["support_advisors"]:
                    entry["support_advisors"].append(advisor)
                self._append_unique_argument(entry["support"], supported_assumption, advisor=advisor, protocol=protocol, removed=removed)
                self._append_unique_argument(
                    useful_unique_arguments,
                    f"{evidence_item}: {supported_assumption}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=12,
                )

            if evidence_item and questioned_assumption:
                entry = by_candidate[evidence_item]
                entry["attacked_count"] += 1
                if advisor and advisor not in entry["against_advisors"]:
                    entry["against_advisors"].append(advisor)
                self._append_unique_argument(entry["against"], questioned_assumption, advisor=advisor, protocol=protocol, removed=removed)
                self._append_unique_argument(
                    key_conflicts,
                    f"{evidence_item} questioned: {questioned_assumption}",
                    advisor=advisor,
                    protocol=protocol,
                    max_items=8,
                )

            for item in support_set:
                entry = by_candidate[item]
                entry["defended_count"] += 1
                if advisor and advisor not in entry["support_advisors"]:
                    entry["support_advisors"].append(advisor)
                support_text = reason_for or task_answer or evidence_can_add or friend_basis
                self._append_unique_argument(entry["support"], support_text, advisor=advisor, protocol=protocol, removed=removed)
                if support_text:
                    self._append_unique_argument(
                        useful_unique_arguments,
                        f"{item}: {support_text}",
                        advisor=advisor,
                        protocol=protocol,
                        max_items=12,
                    )

            for item in against_set:
                entry = by_candidate[item]
                entry["attacked_count"] += 1
                if advisor and advisor not in entry["against_advisors"]:
                    entry["against_advisors"].append(advisor)
                against_text = reason_against or task_answer or still_missing or evidence_can_add or friend_basis
                self._append_unique_argument(entry["against"], against_text, advisor=advisor, protocol=protocol, removed=removed)
                if against_text:
                    self._append_unique_argument(
                        key_conflicts,
                        f"{item} down-ranked: {against_text}",
                        advisor=advisor,
                        protocol=protocol,
                        max_items=8,
                    )

            for item in unclear_set:
                entry = by_candidate[item]
                entry["unclear_count"] += 1
                if advisor and advisor not in entry["unclear_advisors"]:
                    entry["unclear_advisors"].append(advisor)
                unclear_text = still_missing or task_answer or reason_for or reason_against
                self._append_unique_argument(entry["unclear"], unclear_text, advisor=advisor, protocol=protocol, max_items=4, removed=removed)

            if endorsed and stance not in {"unresolved", "caution"} and not (
                evidence_item
                and self._norm_name(endorsed) == self._norm_name(evidence_item)
                and supported_assumption
            ):
                entry = by_candidate[endorsed]
                entry["defended_count"] += 1
                if advisor and advisor not in entry["support_advisors"]:
                    entry["support_advisors"].append(advisor)
                self._append_unique_argument(entry["support"], reason_for, advisor=advisor, protocol=protocol, removed=removed)
                if reason_for:
                    self._append_unique_argument(
                        useful_unique_arguments,
                        f"{endorsed}: {reason_for}",
                        advisor=advisor,
                        protocol=protocol,
                        max_items=12,
                    )

            if attacked and reason_against and not (
                evidence_item
                and self._norm_name(attacked) == self._norm_name(evidence_item)
                and questioned_assumption
            ):
                entry = by_candidate[attacked]
                entry["attacked_count"] += 1
                if advisor and advisor not in entry["against_advisors"]:
                    entry["against_advisors"].append(advisor)
                self._append_unique_argument(entry["against"], reason_against, advisor=advisor, protocol=protocol, removed=removed)
                if endorsed:
                    self._append_unique_argument(
                        key_conflicts,
                        f"{endorsed} over {attacked}: {reason_against}",
                        advisor=advisor,
                        protocol=protocol,
                        max_items=8,
                    )
            elif endorsed and endorsed != proposal_name and reason_against:
                attacked = ensure_item(proposal_name)
                if attacked:
                    entry = by_candidate[attacked]
                    entry["attacked_count"] += 1
                    if advisor and advisor not in entry["against_advisors"]:
                        entry["against_advisors"].append(advisor)
                    self._append_unique_argument(entry["against"], reason_against, advisor=advisor, protocol=protocol, removed=removed)
                    self._append_unique_argument(
                        key_conflicts,
                        f"{endorsed} challenges {attacked}: {reason_against}",
                        advisor=advisor,
                        protocol=protocol,
                        max_items=8,
                    )

        protocols = {str(row.get("protocol", "") or "") for row in advisor_arguments or []}
        defended_items = {
            str(row.get("defended_item") or row.get("endorsed_item") or "").strip()
            for row in advisor_arguments or []
            if str(row.get("defended_item") or row.get("endorsed_item") or "").strip()
        }
        attacked_items = {
            str(row.get("attacked_item") or "").strip()
            for row in advisor_arguments or []
            if str(row.get("attacked_item") or "").strip()
        }
        cooperative_protocols = {p for p in protocols if "cooperative" in p}
        competitive_protocols = {p for p in protocols if "competitive" in p}
        if cooperative_protocols and len(defended_items) <= 1 and not attacked_items:
            protocol_issues.append(
                {
                    "advisor": "system",
                    "issue": "cooperative_monologue",
                    "item": str(next(iter(defended_items), proposal_name) or proposal_name),
                }
            )
        if competitive_protocols and len(focus) >= 3:
            non_proposal_defended = {item for item in defended_items if self._norm_name(item) != self._norm_name(proposal_name)}
            only_proposal_attacked = bool(attacked_items) and {
                self._norm_name(item) for item in attacked_items
            } <= {self._norm_name(proposal_name)}
            if len(non_proposal_defended) >= 2 and only_proposal_attacked:
                protocol_issues.append(
                    {
                        "advisor": "system",
                        "issue": "no_cross_challenge",
                        "item": str(proposal_name or ""),
                    }
                )
            claim_signatures = []
            for row in advisor_arguments or []:
                views = []
                for view in list(row.get("candidate_views", []) or [])[:6]:
                    views.append(
                        "|".join(
                            [
                                self._norm_name((view or {}).get("candidate", "")),
                                str((view or {}).get("view", "") or "").strip().lower(),
                                self._argument_key((view or {}).get("reason", ""))[:80],
                            ]
                        )
                    )
                signature = "||".join(views) or self._argument_key(
                    row.get("reason_for", "") or row.get("reason_against", "") or row.get("task_answer", "")
                )[:160]
                if signature:
                    claim_signatures.append(signature)
            if len(claim_signatures) >= 3:
                counts = defaultdict(int)
                for signature in claim_signatures:
                    counts[signature] += 1
                if counts and max(counts.values()) >= min(3, len(claim_signatures)):
                    protocol_issues.append(
                        {
                            "advisor": "system",
                            "issue": "advisor_homogeneous_claims",
                            "item": str(proposal_name or ""),
                        }
                    )
            if by_candidate and len(by_candidate) >= 3:
                has_risk = any((entry or {}).get("against") or (entry or {}).get("attacked_count") for entry in by_candidate.values())
                has_key_tradeoff = any(str(row.get("key_tradeoff", "") or row.get("comparison_reason", "") or row.get("correction", "") or "").strip() for row in advisor_arguments or [])
                if not has_risk and not has_key_tradeoff and len(represented if "represented" in locals() else []) >= 3:
                    protocol_issues.append(
                        {
                            "advisor": "system",
                            "issue": "no_discrimination_reason_missing",
                            "item": str(proposal_name or ""),
                        }
                    )

        represented = [
            item
            for item, entry in by_candidate.items()
            if entry["defended_count"] > 0 or entry["attacked_count"] > 0 or entry["unclear_count"] > 0 or entry["support"] or entry["against"] or entry["unclear"]
        ]
        if competitive_protocols and len(focus) >= 3 and len(represented) >= 3:
            has_risk = any((entry or {}).get("against") or (entry or {}).get("attacked_count") for entry in by_candidate.values())
            has_key_tradeoff = any(
                str(row.get("key_tradeoff", "") or row.get("comparison_reason", "") or row.get("correction", "") or "").strip()
                for row in advisor_arguments or []
            )
            already_marked = any(
                isinstance(row, dict) and row.get("issue") == "no_discrimination_reason_missing"
                for row in protocol_issues
            )
            if not has_risk and not has_key_tradeoff and not already_marked:
                protocol_issues.append(
                    {
                        "advisor": "system",
                        "issue": "no_discrimination_reason_missing",
                        "item": str(proposal_name or ""),
                    }
                )
        silent_focus_candidates = [
            item
            for item, entry in by_candidate.items()
            if not entry["defended_count"] and not entry["attacked_count"] and not entry["unclear_count"] and not entry["support"] and not entry["against"] and not entry["unclear"]
        ]
        support_only_candidates = [
            item
            for item, entry in by_candidate.items()
            if (entry["support"] or entry["defended_count"]) and not (entry["against"] or entry["attacked_count"])
        ]
        eliminated_candidates = []
        retained_candidates = []
        for item, entry in by_candidate.items():
            has_support = bool(entry["support"] or entry["defended_count"])
            has_against = bool(entry["against"] or entry["attacked_count"])
            if has_against and not has_support:
                eliminated_candidates.append(
                    {
                        "candidate": item,
                        "reason": "challenged by advisor evidence and no extracted support remained",
                    }
                )
            else:
                retained_candidates.append(item)
        unresolved_questions = []
        if not represented:
            if unresolved_advisor_answers:
                unresolved_questions.append("advisor answers were unresolved but contain usable uncertainty evidence")
            else:
                unresolved_questions.append("no usable advisor evidence was extracted")
        if len(represented) >= 2:
            unresolved_questions.append("multiple candidates retain non-identical evidence; user should decide using UserReasoningSkill")
        if silent_focus_candidates:
            unresolved_questions.append(
                "some HesitationSet candidates received no advisor evidence; silence means missing evidence, not negative evidence"
            )
        if support_only_candidates:
            unresolved_questions.append(
                "some candidates were supported but not challenged; unopposed support may reflect advisor assignment rather than true superiority"
            )
        if protocol_issues:
            unresolved_questions.append("some advisor messages had protocol issues; treat affected evidence cautiously")

        candidate_evidence = []
        for item, entry in by_candidate.items():
            support_count = len(entry.get("support_advisors", []) or [])
            risk_count = len(entry.get("against_advisors", []) or [])
            unclear_advisors = len(entry.get("unclear_advisors", []) or [])
            unclear_count = unclear_advisors if unclear_advisors else int(entry.get("unclear_count", 0) or 0)
            if support_count > risk_count and support_count >= max(1, unclear_count):
                status = "mostly support"
            elif risk_count > support_count and risk_count >= max(1, unclear_count):
                status = "mostly risk"
            elif support_count or risk_count:
                status = "mixed"
            elif unclear_count or entry.get("unclear"):
                status = "unclear"
            else:
                status = "silent"
            candidate_evidence.append(
                {
                    "candidate": item,
                    "status": status,
                    "counts": {
                        "support": support_count,
                        "risk": risk_count,
                        "unclear": unclear_count,
                    },
                    "reason": self._candidate_reason_text(entry),
                }
            )
        advisor_questions = []
        for row in advisor_arguments or []:
            q = str(row.get("ask_user", "") or "").strip()
            if q and q.lower() not in {"none", "null", "n/a", "na"} and q not in advisor_questions:
                advisor_questions.append(q)
        main_agreement = ""
        main_conflict = ""
        supported = [x["candidate"] for x in candidate_evidence if x["status"] == "mostly support"]
        risky = [x["candidate"] for x in candidate_evidence if x["status"] == "mostly risk"]
        mixed = [x["candidate"] for x in candidate_evidence if x["status"] == "mixed"]
        if supported:
            main_agreement = "stronger support for: " + ", ".join(supported[:4])
        elif candidate_evidence:
            main_agreement = "advisors produced candidate-level evidence but no clear support consensus"
        if risky:
            main_conflict = "risk evidence for: " + ", ".join(risky[:4])
        elif mixed:
            main_conflict = "mixed evidence for: " + ", ".join(mixed[:4])
        remaining_gap = "; ".join(unresolved_questions[:3]) if unresolved_questions else "none"

        return {
            "summary_type": "distilled_advisor_evidence",
            "decision_policy": "non_binding; user agent must decide with UserReasoningSkill",
            "proposal_item": str(proposal_name or ""),
            "focus_candidates": list(focus),
            "by_candidate": by_candidate,
            "silent_focus_candidates": silent_focus_candidates,
            "missing_advisor_evidence": [
                {
                    "candidate": item,
                    "reason": "HesitationSet candidate was not evaluated by any advisor; do not treat silence as weak preference fit",
                }
                for item in silent_focus_candidates
            ],
            "support_only_candidates": support_only_candidates,
            "eliminated_candidates": eliminated_candidates,
            "retained_candidates": retained_candidates,
            "unchallenged_support_warning": (
                "support without opposition may be an artifact of advisor assignment, especially in competitive promotion"
                if support_only_candidates
                else ""
            ),
            "key_conflicts": key_conflicts,
            "useful_unique_arguments": useful_unique_arguments,
            "advisor_interactions": advisor_interactions[:8],
            "comparative_claims": comparative_claims[:8],
            "correction_claims": correction_claims[:8],
            "candidate_evidence": candidate_evidence,
            "discussion_summary": {
                "main_agreement": main_agreement or "none",
                "main_conflict": main_conflict or "none",
                "remaining_gap": remaining_gap,
                "advisor_questions_for_user": advisor_questions[:6],
            },
            "advisor_questions_for_user": advisor_questions[:6],
            "unresolved_advisor_answers": unresolved_advisor_answers[:8],
            "repeated_or_weak_arguments_removed": removed[:12],
            "unresolved_questions": unresolved_questions,
            "protocol_issues": list(protocol_issues or []),
        }

    @staticmethod
    def _discussion_result_from_information(evidence_summary):
        summary = dict(evidence_summary or {})
        silent = list(summary.get("silent_focus_candidates", []) or [])
        protocol_issues = list(summary.get("protocol_issues", []) or [])
        key_conflicts = list(summary.get("key_conflicts", []) or [])
        useful_arguments = list(summary.get("useful_unique_arguments", []) or [])
        by_candidate = dict(summary.get("by_candidate", {}) or {})
        covered = [
            item
            for item, entry in by_candidate.items()
            if (entry or {}).get("support")
            or (entry or {}).get("against")
            or (entry or {}).get("unclear")
            or (entry or {}).get("defended_count")
            or (entry or {}).get("attacked_count")
            or (entry or {}).get("unclear_count")
        ]
        severe_issues = set()
        for row in protocol_issues:
            if isinstance(row, dict):
                issue = str(row.get("issue", "") or "")
            else:
                issue = str(row or "")
            if issue:
                severe_issues.add(issue)
        has_real_comparison = bool(key_conflicts) or any(
            (entry or {}).get("against") or (entry or {}).get("attacked_count")
            for entry in by_candidate.values()
        )
        if "cooperative_monologue" in severe_issues:
            return "partially_resolved" if useful_arguments else "unresolved"
        if "no_cross_challenge" in severe_issues:
            return "partially_resolved"
        if silent:
            return "partially_resolved" if useful_arguments or key_conflicts else "unresolved"
        if protocol_issues:
            return "partially_resolved"
        if len(by_candidate) >= 2 and not has_real_comparison:
            return "partially_resolved" if len(covered) >= 2 else "unresolved"
        return "partially_resolved" if covered or useful_arguments else "unresolved"

    def _aggregate_feedback(self, proposal_name, feedbacks, focus_candidates=None, tree=None, path=None):
        allowed_items = [str(x) for x in (focus_candidates or []) if str(x or "").strip()]
        support_count = 0
        oppose_count = 0
        proposal_support_reasons = []
        proposal_oppose_reasons = []
        proposal_additional_info = []
        advisor_arguments = []
        protocol_issues = []

        for fb in feedbacks or []:
            advisor = str(fb.get("advisor_id", "") or "advisor")
            stance = str(fb.get("stance", "") or "")
            endorsed_item = str(fb.get("endorsed_item", "") or "")
            if not endorsed_item and stance not in {"evidence_only"}:
                endorsed_item = str(proposal_name or "")
            if allowed_items and endorsed_item and endorsed_item not in allowed_items:
                protocol_issues.append(
                    {
                        "advisor": advisor,
                        "issue": "out_of_focus_candidate_filtered",
                        "item": endorsed_item or str(fb.get("reviewed_item", "") or fb.get("evidence_item", "") or ""),
                    }
                )
                continue
            for issue in list(fb.get("protocol_issues", []) or []):
                protocol_issues.append(
                    {
                        "advisor": advisor,
                        "issue": str(issue),
                        "item": endorsed_item,
                    }
                )
            advisor_arguments.append(
                {
                    "advisor": advisor,
                    "stance": stance,
                    "vote_action": str(fb.get("vote_action", stance) or stance),
                    "endorsed_item": endorsed_item,
                    "defended_item": "" if stance == "evidence_only" else str(fb.get("defended_item", endorsed_item) or endorsed_item),
                    "attacked_item": str(fb.get("attacked_item", "") or ""),
                    "protocol": str(fb.get("protocol", "") or ""),
                    "decision_value": str(fb.get("decision_value", "") or ""),
                    "advisor_guidance": dict(fb.get("advisor_guidance", {}) or {}),
                    "friend_basis": str(fb.get("friend_basis", "") or ""),
                    "evidence_item": str(fb.get("evidence_item", "") or fb.get("reviewed_item", "") or ""),
                    "reviewed_item": str(fb.get("reviewed_item", "") or fb.get("evidence_item", "") or ""),
                    "supported_assumption": str(fb.get("supported_assumption", "") or ""),
                    "questioned_assumption": str(fb.get("questioned_assumption", "") or ""),
                    "remove_set": list(fb.get("remove_set", []) or []),
                    "keep_set": list(fb.get("keep_set", []) or []),
                    "interested_set": list(fb.get("interested_set", []) or []),
                    "weak_fit_set": list(fb.get("weak_fit_set", []) or []),
                    "stronger_candidates": list(fb.get("stronger_candidates", []) or []),
                    "weaker_candidates": list(fb.get("weaker_candidates", []) or []),
                    "unclear_set": list(fb.get("unclear_set", []) or []),
                    "covered_set": list(fb.get("covered_set", []) or []),
                    "missing_set": list(fb.get("missing_set", []) or []),
                    "evidence_gap_set": list(fb.get("evidence_gap_set", []) or []),
                    "candidate_views": list(fb.get("candidate_views", []) or []),
                    "candidate_evidence_sources": list(fb.get("candidate_evidence_sources", []) or []),
                    "task_answer": str(fb.get("task_answer", "") or ""),
                    "ask_user": str(fb.get("ask_user", "") or ""),
                    "response_to_previous": str(fb.get("response_to_previous", "") or ""),
                    "challenge_or_support_previous": str(fb.get("challenge_or_support_previous", "") or ""),
                    "key_tradeoff": str(fb.get("key_tradeoff", "") or ""),
                    "comparison_reason": str(fb.get("comparison_reason", "") or ""),
                    "correction": str(fb.get("correction", "") or ""),
                    "still_missing": str(fb.get("still_missing", "") or ""),
                    "evidence_can_add": str(fb.get("evidence_can_add", "") or ""),
                    "supplement_reason": str(fb.get("supplement_reason", "") or ""),
                    "reason_against": str(fb.get("oppose_reason", "") or ""),
                    "reason_for": str(fb.get("support_reason", "") or ""),
                    "protocol_issues": list(fb.get("protocol_issues", []) or []),
                    "task_type": str(fb.get("task_type", "") or fb.get("what", "") or ""),
                }
            )
            if stance in {"unresolved", "caution", "evidence_only"}:
                proposal_additional_info.append(str(fb.get("support_reason", "") or fb.get("oppose_reason", "") or "unresolved"))
            elif endorsed_item == proposal_name:
                support_count += 1
                if fb.get("support_reason"):
                    proposal_support_reasons.append(str(fb.get("support_reason")))
            else:
                oppose_count += 1
                if fb.get("oppose_reason"):
                    proposal_oppose_reasons.append(str(fb.get("oppose_reason")))
                if fb.get("support_reason"):
                    proposal_additional_info.append(str(fb.get("support_reason")))

        evidence_summary = self._build_evidence_summary(
            proposal_name=proposal_name,
            advisor_arguments=advisor_arguments,
            focus_candidates=allowed_items,
            protocol_issues=protocol_issues,
        )
        protocol_issues = list(evidence_summary.get("protocol_issues", []) or [])
        discussion_result = self._discussion_result_from_information(evidence_summary)
        evidence_summary["discussion_result"] = discussion_result
        evidence_summary["retained_candidates"] = list(evidence_summary.get("retained_candidates", []) or []) or [
            item
            for item, entry in (evidence_summary.get("by_candidate", {}) or {}).items()
            if entry.get("support")
            or entry.get("against")
            or entry.get("unclear")
            or entry.get("defended_count")
            or entry.get("attacked_count")
            or entry.get("unclear_count")
        ] or list(allowed_items)
        evidence_summary["recommended_next_state"] = (
            "final_decision" if discussion_result == "resolved"
            else "multi-competitive" if len(evidence_summary["retained_candidates"]) == 2
            else "multi-cooperative" if discussion_result == "unresolved"
            else "user_decides_or_continue"
        )
        summary_agent_input = self._build_summary_agent_input(
            proposal_name=proposal_name,
            feedbacks=feedbacks,
            focus_candidates=allowed_items,
            path=path,
            evidence_summary=evidence_summary,
        )
        advisor_synthesis_packet, summary_parse_status, summary_raw_response = self._run_advisor_summary_agent(
            summary_agent_input,
            focus_candidates=allowed_items,
            evidence_summary=evidence_summary,
        )
        return {
            "aggregation_mode": "summary_agent_v1",
            "summary_agent_input": summary_agent_input,
            "advisor_synthesis_packet": advisor_synthesis_packet,
            "summary_agent_parse_status": str(summary_parse_status or ""),
            "summary_agent_raw_response": self._compact_text(summary_raw_response, 1800),
            "legacy_aggregation_used": False,
            "raw_advisor_feedbacks": list(feedbacks or []),
            "decision_policy": "information_only_no_vote",
            "proposal_support_count": int(support_count),
            "proposal_oppose_count": int(oppose_count),
            "discussion_result": discussion_result,
            "advisor_pool_empty": bool(not feedbacks),
            "proposal_support_reasons": proposal_support_reasons,
            "proposal_oppose_reasons": proposal_oppose_reasons,
            "proposal_additional_info": proposal_additional_info,
            "alternative_candidates": [
                item
                for item, entry in (evidence_summary.get("by_candidate", {}) or {}).items()
                if item != proposal_name and any((entry or {}).get(bucket) for bucket in ["support", "against", "unclear"])
            ],
            "advisor_arguments": advisor_arguments,
            "evidence_summary": evidence_summary,
            "evidence_by_candidate": dict(evidence_summary.get("by_candidate", {}) or {}),
            "silent_focus_candidates": list(evidence_summary.get("silent_focus_candidates", []) or []),
            "support_only_candidates": list(evidence_summary.get("support_only_candidates", []) or []),
            "missing_advisor_evidence": list(evidence_summary.get("missing_advisor_evidence", []) or []),
            "answered_user_feedback": [],
            "unanswered_user_feedback": [],
            "protocol_issues": protocol_issues,
            "focus_candidates": list(allowed_items),
        }

    def _direct_advisor_packet(self, proposal_name, feedback, focus_candidates=None, tree=None, path=None):
        allowed_items = [str(x) for x in (focus_candidates or []) if str(x or "").strip()]
        fb = dict(feedback or {})
        advisor = str(fb.get("advisor_id", "") or "advisor")
        stance = str(fb.get("stance", "") or "")
        endorsed_item = str(fb.get("endorsed_item", "") or "")
        if not endorsed_item and stance not in {"evidence_only"}:
            endorsed_item = str(proposal_name or "")
        if allowed_items and endorsed_item and endorsed_item not in allowed_items:
            endorsed_item = str(proposal_name or "")
        defended_item = "" if stance == "evidence_only" else str(fb.get("defended_item", "") or endorsed_item or proposal_name)
        attacked_item = str(fb.get("attacked_item", "") or "")
        support_reason = str(fb.get("support_reason", "") or "")
        oppose_reason = str(fb.get("oppose_reason", "") or "")
        raw_text = str(fb.get("raw_text", "") or "")
        confidence = int(fb.get("confidence", 0) or 0)
        is_support_proposal = self._norm_name(endorsed_item) == self._norm_name(proposal_name)
        advisor_argument = {
            "advisor": advisor,
            "stance": stance,
            "vote_action": str(fb.get("vote_action", "") or fb.get("stance", "") or ""),
            "endorsed_item": endorsed_item,
            "defended_item": defended_item,
            "attacked_item": attacked_item,
            "protocol": str(fb.get("protocol", "") or ""),
            "reason_against": oppose_reason,
            "reason_for": support_reason,
            "friend_basis": str(fb.get("friend_basis", "") or ""),
            "remove_set": list(fb.get("remove_set", []) or []),
            "keep_set": list(fb.get("keep_set", []) or []),
            "interested_set": list(fb.get("interested_set", []) or []),
            "weak_fit_set": list(fb.get("weak_fit_set", []) or []),
            "stronger_candidates": list(fb.get("stronger_candidates", []) or []),
            "weaker_candidates": list(fb.get("weaker_candidates", []) or []),
            "unclear_set": list(fb.get("unclear_set", []) or []),
            "covered_set": list(fb.get("covered_set", []) or []),
            "missing_set": list(fb.get("missing_set", []) or []),
            "evidence_gap_set": list(fb.get("evidence_gap_set", []) or []),
            "candidate_views": list(fb.get("candidate_views", []) or []),
            "candidate_evidence_sources": list(fb.get("candidate_evidence_sources", []) or []),
            "task_answer": str(fb.get("task_answer", "") or ""),
            "ask_user": str(fb.get("ask_user", "") or ""),
            "response_to_previous": str(fb.get("response_to_previous", "") or ""),
            "challenge_or_support_previous": str(fb.get("challenge_or_support_previous", "") or ""),
            "key_tradeoff": str(fb.get("key_tradeoff", "") or ""),
            "comparison_reason": str(fb.get("comparison_reason", "") or ""),
            "correction": str(fb.get("correction", "") or ""),
            "still_missing": str(fb.get("still_missing", "") or ""),
            "evidence_can_add": str(fb.get("evidence_can_add", "") or ""),
            "supplement_reason": str(fb.get("supplement_reason", "") or ""),
            "supported_assumption": str(fb.get("supported_assumption", "") or ""),
            "questioned_assumption": str(fb.get("questioned_assumption", "") or ""),
            "raw_text": raw_text,
            "protocol_issues": list(fb.get("protocol_issues", []) or []),
            "task_type": str(fb.get("task_type", "") or fb.get("what", "") or ""),
        }
        by_candidate = {
            item: {
                "support": [],
                "against": [],
                "unclear": [],
                "support_advisors": [],
                "against_advisors": [],
                "unclear_advisors": [],
                "defended_count": 0,
                "attacked_count": 0,
                "unclear_count": 0,
            }
            for item in allowed_items
        }

        def ensure_item(item):
            item = str(item or "").strip()
            if not item:
                return ""
            if allowed_items and item not in by_candidate:
                return ""
            by_candidate.setdefault(
                item,
                {
                    "support": [],
                    "against": [],
                    "unclear": [],
                    "support_advisors": [],
                    "against_advisors": [],
                    "unclear_advisors": [],
                    "defended_count": 0,
                    "attacked_count": 0,
                    "unclear_count": 0,
                },
            )
            return item

        defended_key = ensure_item(defended_item or endorsed_item)
        attacked_key = ensure_item(attacked_item)
        if defended_key:
            row = by_candidate[defended_key]
            row["defended_count"] += 1
            row["support_advisors"].append(advisor)
            if support_reason:
                row["support"].append({"reason": support_reason, "advisor": advisor, "protocol": str(fb.get("protocol", "") or "")})
        if attacked_key:
            row = by_candidate[attacked_key]
            row["attacked_count"] += 1
            row["against_advisors"].append(advisor)
            if oppose_reason:
                row["against"].append({"reason": oppose_reason, "advisor": advisor, "protocol": str(fb.get("protocol", "") or "")})
        support_text = support_reason or str(fb.get("evidence_can_add", "") or "") or str(fb.get("friend_basis", "") or "")
        against_text = oppose_reason or support_reason or str(fb.get("evidence_can_add", "") or "") or str(fb.get("friend_basis", "") or "")
        gap_text = str(fb.get("supplement_reason", "") or fb.get("still_missing", "") or fb.get("task_answer", "") or "")
        for item in list(fb.get("keep_set", []) or []) + list(fb.get("interested_set", []) or []) + list(fb.get("stronger_candidates", []) or []):
            key = ensure_item(item)
            if not key:
                continue
            row = by_candidate[key]
            row["defended_count"] += 1
            if advisor not in row["support_advisors"]:
                row["support_advisors"].append(advisor)
            if support_text:
                row["support"].append({"reason": support_text, "advisor": advisor, "protocol": str(fb.get("protocol", "") or "")})
        for item in list(fb.get("remove_set", []) or []) + list(fb.get("weak_fit_set", []) or []) + list(fb.get("weaker_candidates", []) or []):
            key = ensure_item(item)
            if not key:
                continue
            row = by_candidate[key]
            row["attacked_count"] += 1
            if advisor not in row["against_advisors"]:
                row["against_advisors"].append(advisor)
            if against_text:
                row["against"].append({"reason": against_text, "advisor": advisor, "protocol": str(fb.get("protocol", "") or "")})
        for item in list(fb.get("evidence_gap_set", []) or []):
            key = ensure_item(item)
            if not key:
                continue
            row = by_candidate[key]
            row["unclear_count"] += 1
            if advisor not in row["unclear_advisors"]:
                row["unclear_advisors"].append(advisor)
            if gap_text:
                row["unclear"].append({"reason": gap_text, "advisor": advisor, "protocol": str(fb.get("protocol", "") or "")})
        direct_candidate_views = list(fb.get("candidate_views", []) or [])
        if not direct_candidate_views:
            for source in list(fb.get("candidate_evidence_sources", []) or []):
                source = dict(source or {})
                direct_candidate_views.append(
                    {
                        "candidate": str(source.get("candidate", "") or ""),
                        "view": str(source.get("source_label", "") or source.get("label", "") or "unclear"),
                        "reason": str(source.get("reason", "") or ""),
                    }
                )
        for view in direct_candidate_views:
            key = ensure_item((view or {}).get("candidate", ""))
            if not key:
                continue
            row = by_candidate[key]
            bucket = self._candidate_view_bucket((view or {}).get("view", ""))
            reason = str((view or {}).get("reason", "") or fb.get("task_answer", "") or "").strip()
            if bucket == "support":
                row["defended_count"] += 1
                if advisor not in row["support_advisors"]:
                    row["support_advisors"].append(advisor)
                if reason:
                    row["support"].append({"reason": reason, "advisor": advisor, "protocol": str(fb.get("protocol", "") or "")})
            elif bucket == "risk":
                row["attacked_count"] += 1
                if advisor not in row["against_advisors"]:
                    row["against_advisors"].append(advisor)
                if reason:
                    row["against"].append({"reason": reason, "advisor": advisor, "protocol": str(fb.get("protocol", "") or "")})
            else:
                row["unclear_count"] += 1
                if advisor not in row["unclear_advisors"]:
                    row["unclear_advisors"].append(advisor)
                if reason:
                    row["unclear"].append({"reason": reason, "advisor": advisor, "protocol": str(fb.get("protocol", "") or "")})

        silent_focus_candidates = [
            item
            for item, row in by_candidate.items()
            if not row["defended_count"] and not row["attacked_count"] and not row["unclear_count"] and not row["support"] and not row["against"] and not row["unclear"]
        ]
        candidate_evidence = []
        for item, row in by_candidate.items():
            support_count = len(row.get("support_advisors", []) or [])
            risk_count = len(row.get("against_advisors", []) or [])
            unclear_advisors = len(row.get("unclear_advisors", []) or [])
            unclear_count = unclear_advisors if unclear_advisors else int(row.get("unclear_count", 0) or 0)
            status = "mostly support" if support_count > risk_count and support_count >= max(1, unclear_count) else (
                "mostly risk" if risk_count > support_count and risk_count >= max(1, unclear_count) else (
                    "mixed" if support_count or risk_count else "unclear" if unclear_count else "silent"
                )
            )
            candidate_evidence.append(
                {
                    "candidate": item,
                    "status": status,
                    "counts": {"support": support_count, "risk": risk_count, "unclear": unclear_count},
                    "reason": self._candidate_reason_text(row),
                }
            )
        advisor_interactions = []
        comparative_claims = []
        correction_claims = []

        def add_direct_summary(bucket, label, text, max_items=6):
            text = self._compact_text(text, 320)
            if not text or len(bucket) >= max_items:
                return
            key = (str(label or "").lower(), self._argument_key(text))
            existing = {
                (str(row.get("type", "") or "").lower(), self._argument_key(row.get("reason", "")))
                for row in bucket
            }
            if key in existing:
                return
            bucket.append(
                {
                    "type": str(label or ""),
                    "reason": text,
                    "advisor": advisor,
                    "protocol": str(fb.get("protocol", "") or ""),
                }
            )

        for label, field in [
            ("ResponseToPrevious", "response_to_previous"),
            ("ChallengeOrSupportPrevious", "challenge_or_support_previous"),
        ]:
            add_direct_summary(advisor_interactions, label, str(fb.get(field, "") or ""))
        for label, field in [
            ("KeyTradeoff", "key_tradeoff"),
            ("ComparisonReason", "comparison_reason"),
        ]:
            add_direct_summary(comparative_claims, label, str(fb.get(field, "") or ""))
        for label, field in [
            ("Correction", "correction"),
            ("QuestionedAssumption", "questioned_assumption"),
        ]:
            add_direct_summary(correction_claims, label, str(fb.get(field, "") or ""))
        direct_key_conflicts = []
        if defended_key and attacked_key and oppose_reason:
            direct_key_conflicts.append(
                {
                    "reason": f"{defended_key} over {attacked_key}: {oppose_reason}",
                    "advisor": advisor,
                    "protocol": str(fb.get("protocol", "") or ""),
                }
            )
        for row in correction_claims:
            direct_key_conflicts.append(
                {
                    "reason": f"{row.get('type', 'Correction')}: {row.get('reason', '')}",
                    "advisor": advisor,
                    "protocol": str(fb.get("protocol", "") or ""),
                }
            )
        challenge_text = str(fb.get("challenge_or_support_previous", "") or "")
        if challenge_text:
            direct_key_conflicts.append(
                {
                    "reason": f"Cross-advisor response: {self._compact_text(challenge_text, 320)}",
                    "advisor": advisor,
                    "protocol": str(fb.get("protocol", "") or ""),
                }
            )
        direct_useful_arguments = []
        if defended_key and support_reason:
            direct_useful_arguments.append(
                {
                    "reason": f"{defended_key}: {support_reason}",
                    "advisor": advisor,
                    "protocol": str(fb.get("protocol", "") or ""),
                }
            )
        for row in advisor_interactions + comparative_claims + correction_claims:
            direct_useful_arguments.append(
                {
                    "reason": f"{row.get('type', 'AdvisorSignal')}: {row.get('reason', '')}",
                    "advisor": advisor,
                    "protocol": str(fb.get("protocol", "") or ""),
                }
            )
        evidence_summary = {
            "summary_type": "direct_single_advisor_evidence",
            "decision_policy": "non_binding; user reads the single advisor message directly",
            "discussion_result": "single_advisor_direct",
            "recommended_next_state": "user_decides_or_continue",
            "proposal_item": str(proposal_name or ""),
            "focus_candidates": list(allowed_items),
            "single_advisor": advisor,
            "single_advisor_message": {
                "defended_item": defended_item,
                "challenged_item": attacked_item,
                "friend_basis": str(fb.get("friend_basis", "") or ""),
                "remove_set": list(fb.get("remove_set", []) or []),
                "keep_set": list(fb.get("keep_set", []) or []),
                "interested_set": list(fb.get("interested_set", []) or []),
                "weak_fit_set": list(fb.get("weak_fit_set", []) or []),
                "support_reason": support_reason,
                "oppose_reason": oppose_reason,
                "task_answer": str(fb.get("task_answer", "") or ""),
                "ask_user": str(fb.get("ask_user", "") or ""),
                "response_to_previous": str(fb.get("response_to_previous", "") or ""),
                "challenge_or_support_previous": str(fb.get("challenge_or_support_previous", "") or ""),
                "key_tradeoff": str(fb.get("key_tradeoff", "") or ""),
                "comparison_reason": str(fb.get("comparison_reason", "") or ""),
                "correction": str(fb.get("correction", "") or ""),
                "still_missing": str(fb.get("still_missing", "") or ""),
                "supported_assumption": str(fb.get("supported_assumption", "") or ""),
                "questioned_assumption": str(fb.get("questioned_assumption", "") or ""),
                "raw_text": raw_text,
            },
            "by_candidate": by_candidate,
            "candidate_evidence": candidate_evidence,
            "discussion_summary": {
                "main_agreement": "single advisor produced candidate-level evidence",
                "main_conflict": (
                    self._compact_text(str(fb.get("correction", "") or fb.get("questioned_assumption", "") or ""), 220)
                    or "none"
                ),
                "remaining_gap": str(fb.get("still_missing", "") or fb.get("ask_user", "") or "only one advisor spoke"),
                "advisor_questions_for_user": [str(fb.get("ask_user", ""))] if str(fb.get("ask_user", "") or "").strip() else [],
            },
            "advisor_questions_for_user": [str(fb.get("ask_user", ""))] if str(fb.get("ask_user", "") or "").strip() else [],
            "silent_focus_candidates": silent_focus_candidates,
            "missing_advisor_evidence": [
                {
                    "candidate": item,
                    "reason": "single advisor did not evaluate this HesitationSet candidate; treat as missing evidence, not negative evidence",
                }
                for item in silent_focus_candidates
            ],
            "support_only_candidates": [defended_key] if defended_key and not attacked_key else [],
            "key_conflicts": direct_key_conflicts[:8],
            "useful_unique_arguments": direct_useful_arguments[:12],
            "advisor_interactions": advisor_interactions[:6],
            "comparative_claims": comparative_claims[:6],
            "correction_claims": correction_claims[:6],
            "repeated_or_weak_arguments_removed": [],
            "unresolved_questions": [
                "only one advisor spoke; use the advisor message as evidence, not as committee consensus",
            ] + (
                ["some HesitationSet candidates received no advisor evidence; silence means missing evidence, not negative evidence"]
                if silent_focus_candidates
                else []
            ),
            "protocol_issues": list(fb.get("protocol_issues", []) or []),
        }
        summary_agent_input = self._build_summary_agent_input(
            proposal_name=proposal_name,
            feedbacks=[fb],
            focus_candidates=allowed_items,
            path=path,
            evidence_summary=evidence_summary,
        )
        advisor_synthesis_packet, summary_parse_status, summary_raw_response = self._run_advisor_summary_agent(
            summary_agent_input,
            focus_candidates=allowed_items,
            evidence_summary=evidence_summary,
        )
        return {
            "aggregation_mode": "summary_agent_v1",
            "summary_agent_input": summary_agent_input,
            "advisor_synthesis_packet": advisor_synthesis_packet,
            "summary_agent_parse_status": str(summary_parse_status or ""),
            "summary_agent_raw_response": self._compact_text(summary_raw_response, 1800),
            "legacy_aggregation_used": False,
            "raw_advisor_feedbacks": [fb],
            "decision_policy": "information_only_no_vote",
            "proposal_support_count": 1 if is_support_proposal else 0,
            "proposal_oppose_count": 0 if is_support_proposal else 1,
            "proposal_support_reasons": [support_reason] if is_support_proposal and support_reason else [],
            "proposal_oppose_reasons": [oppose_reason] if (not is_support_proposal) and oppose_reason else [],
            "proposal_additional_info": [support_reason] if (not is_support_proposal) and support_reason else [],
            "alternative_candidates": [] if is_support_proposal else [endorsed_item],
            "advisor_arguments": [advisor_argument],
            "discussion_result": "single_advisor_direct",
            "evidence_summary": evidence_summary,
            "protocol_issues": list(fb.get("protocol_issues", []) or []),
            "focus_candidates": list(allowed_items),
            "single_advisor_direct": True,
        }

    def execute(
        self,
        host,
        path,
        advisor_profiles,
        advisor_agent,
        history_str,
        target_profile,
        proposal_name,
        proposal_reason,
        shortlist_names,
        cands_int,
        prior_hint,
        shared_memory,
        candidate_evidence=None,
        hesitation_reason="",
        target_user_skill=None,
        round_type="open_candidate_review",
        previous_user_feedback=None,
        previous_discussion_memory=None,
        previous_round_summary=None,
    ):
        tree_engine = getattr(host, "tree_engine", None)
        tree = getattr(tree_engine, "public_tree", {}) if tree_engine is not None else {}
        shortlist_len = len([x for x in (shortlist_names or []) if str(x or "").strip()])
        how = self._canonical_how((path or {}).get("how", "") or "single-advisor", focus_count=shortlist_len)
        task_type = str((path or {}).get("what", "") or "none")
        user_task = str((path or {}).get("user_task", "") or "")
        secondary_what = [str(x) for x in list((path or {}).get("secondary_what", []) or []) if str(x).strip()]
        criteria = [str(x) for x in list((path or {}).get("criteria", []) or []) if str(x).strip()]
        if path is not None:
            path = dict(path or {})
            path["how"] = how
        else:
            path = {}
            path["how"] = how
        shape = infer_communication_shape(how)
        family = infer_communication_family(how)
        intent = infer_communication_intent(how)
        default_limit = 3 if shape == "multi" else 3
        focus_candidates = self._focus_candidates(
            host,
            cands_int,
            shortlist_names,
            proposal_name,
            limit=max(default_limit, shortlist_len),
        )
        focus_candidates, scope_reasons = self._task_scoped_focus_candidates(focus_candidates, path, round_type)
        if scope_reasons and path is not None:
            path["path_reason"] = list(path.get("path_reason", []) or []) + list(scope_reasons)
            path["task_scoped_focus_candidates"] = list(focus_candidates)
        advisor_profiles = list(advisor_profiles or [])
        if shape == "single":
            advisor_profiles = advisor_profiles[:1]
        elif shape == "multi":
            advisor_profiles = advisor_profiles[: max(2, int(getattr(self.args, "com_max_advisors", 2) or 2))]
        if shape == "multi" and len(advisor_profiles) < 2 and intent != "feedback-repair":
            path = dict(path or {})
            original_how = how
            how = "single-advisor"
            path["how"] = how
            path["path_reason"] = list(path.get("path_reason", []) or []) + [
                f"execution downgraded from {original_how} because only {len(advisor_profiles)} advisor was available",
                f"single advisor cannot execute multi-advisor communication; use {how} instead",
            ]
            path["risk_marks"] = list(path.get("risk_marks", []) or []) + ["advisor_count_insufficient_for_debate"]
            shape = "single"
            family = infer_communication_family(how)
            intent = infer_communication_intent(how)
        advisor_discussion_rounds = self._advisor_discussion_rounds(shape, path)
        path["advisor_discussion_rounds_requested"] = max(1, int(getattr(self.args, "com_advisor_discussion_rounds", 1) or 1))
        path["advisor_discussion_rounds_effective"] = int(advisor_discussion_rounds)
        protocol_instruction = self._protocol_instruction(path, advisor_count=len(advisor_profiles))
        initial_choice_context = self._compact_initial_choice_context(
            proposal_reason=proposal_reason,
            hesitation_reason=hesitation_reason,
            candidate_evidence=candidate_evidence,
            focus_candidates=focus_candidates,
            proposal_name=proposal_name,
        )
        feedbacks = []
        requester_brief_mode = isinstance(target_user_skill, dict) and "requester_shareable_item_brief" in target_user_skill
        advisor_visible_history = (
            "privacy_filtered; use RequesterShareableItemBrief and HesitationSet instead of requester full history"
            if requester_brief_mode
            else history_str
        )
        if not advisor_profiles:
            committee_packet = self._aggregate_feedback(proposal_name, feedbacks, focus_candidates=focus_candidates, tree=tree, path=path)
            committee_packet["protocol_issues"] = list(committee_packet.get("protocol_issues", []) or []) + [
                {
                    "advisor": "system",
                    "issue": "advisor_pool_empty",
                    "item": str(proposal_name),
                }
            ]
            committee_packet["advisor_pool_empty"] = True
            return {
                "path": dict(path or {}),
                "interaction_shape": "one-to-one" if shape == "single" else "one-to-many",
                "communication_protocol": str(how or ""),
                "round_type": str(round_type or "open_candidate_review"),
                "previous_user_feedback": previous_user_feedback if previous_user_feedback else {},
                "focus_candidates": list(focus_candidates),
                "advisor_profiles": [],
                "advisor_feedbacks": [],
                "advisor_discussion_rounds": int(advisor_discussion_rounds),
                "committee_packet": committee_packet,
            }

        discussion_memory = []
        previous_round_memory = self._compact_previous_round_summary(previous_round_summary)
        final_pass_feedbacks = []
        speaker_profiles = list(advisor_profiles or [])
        for discussion_pass in range(1, advisor_discussion_rounds + 1):
            pass_feedbacks = []
            for idx, profile in enumerate(speaker_profiles, start=1):
                advisor_guidance = self._advisor_friend_guidance(
                    how,
                    idx,
                    len(speaker_profiles),
                    focus_candidates,
                    proposal_name,
                    previous_user_feedback=previous_user_feedback if previous_user_feedback else None,
                    user_task=user_task,
                    task_type=task_type,
                )
                advisor_guidance["advisor_discussion_pass"] = int(discussion_pass)
                advisor_guidance["advisor_discussion_rounds"] = int(advisor_discussion_rounds)
                output_contract_fields = self._output_contract_fields_for_path(tree, path)
                advisor_guidance["output_contract_fields"] = output_contract_fields
                advisor_guidance["required_output_fields"] = output_contract_fields
                friend_context = self._advisor_context(host, profile)
                advisor_own_skill = self._advisor_own_skill_context(host, profile)
                full_instruction = (
                    f"{self._advisor_skill_payload(tree, path, advisor_profile=profile)}\n\n"
                    f"{protocol_instruction}\n\n"
                    f"{advisor_guidance.get('how_execution_checklist', '')}\n\n"
                    f"AdvisorDiscussionPass: {discussion_pass}/{advisor_discussion_rounds}. "
                    "Read the prior advisor discussion memory, then revise, challenge, or strengthen your own view with concrete evidence.\n\n"
                    "Output must stay structured and recommendation-focused."
                ).strip()
                round_memory = self._compact_discussion_memory(discussion_memory)
                update_llm_prompt_trace_context(
                    phase="advisor_communication",
                    advisor_index=int(idx),
                    advisor_id=str(profile.get("u_raw", "") or ""),
                    advisor_type=str(profile.get("advisor_type", "") or ""),
                    path_how=str(how or ""),
                    advisor_discussion_pass=int(discussion_pass),
                    advisor_discussion_rounds=int(advisor_discussion_rounds),
                )
                decision, speech, alt_name, pos = advisor_agent.run_skill_native_review(
                    skill_instruction=full_instruction,
                    history_str=advisor_visible_history,
                    candidate_names=focus_candidates,
                    proposal=proposal_name,
                    proposal_reason=proposal_reason,
                    friend_context=friend_context,
                    friend_history_summary="",
                    friend_top_rated="",
                    candidate_matches="",
                    candidate_suggestions="",
                    group_memory=round_memory,
                    prior_hint="",
                    target_profile="",
                    shared_memory={
                        "user_task": user_task,
                        "task_type": task_type,
                        "secondary_what": list(secondary_what),
                        "criteria": list(criteria),
                        "task_source": str((path or {}).get("task_source", "") or ""),
                        "user_initial_choice_context": initial_choice_context,
                        "previous_user_feedback": previous_user_feedback if previous_user_feedback else "none",
                        "previous_round_summary": previous_round_memory,
                        "current_round_discussion_memory": round_memory,
                        "advisor_guidance": advisor_guidance,
                        "how_execution_checklist": advisor_guidance.get("how_execution_checklist", ""),
                    },
                    friend_speaker=f"Advisor{idx}(raw={profile.get('u_raw', '')})",
                    revote_round="false",
                    target_user_skill=target_user_skill,
                    advisor_own_skill=advisor_own_skill,
                )
                if not alt_name:
                    alt_name = proposal_name
                feedback = self._parse_feedback(
                    host,
                    profile,
                    decision,
                    speech,
                    alt_name,
                    pos,
                    proposal_name,
                    allowed_items=focus_candidates,
                    protocol=how,
                    advisor_guidance=advisor_guidance,
                    group_memory=round_memory,
                )
                feedback["advisor_guidance"] = dict(advisor_guidance)
                feedback["advisor_discussion_pass"] = int(discussion_pass)
                feedback["advisor_discussion_rounds"] = int(advisor_discussion_rounds)
                feedbacks.append(feedback)
                pass_feedbacks.append(feedback)
                self._append_discussion_memory(discussion_memory, profile, feedback)
            final_pass_feedbacks = pass_feedbacks

        summary_feedbacks = final_pass_feedbacks or feedbacks
        if len(summary_feedbacks) == 1:
            committee_packet = self._direct_advisor_packet(proposal_name, summary_feedbacks[0], focus_candidates=focus_candidates, tree=tree, path=path)
        else:
            committee_packet = self._aggregate_feedback(proposal_name, summary_feedbacks, focus_candidates=focus_candidates, tree=tree, path=path)
        committee_packet["advisor_discussion_rounds"] = int(advisor_discussion_rounds)
        committee_packet["advisor_discussion_final_pass_feedbacks"] = list(summary_feedbacks)
        return {
            "path": dict(path or {}),
            "interaction_shape": "one-to-one" if shape == "single" else "one-to-many",
            "communication_protocol": str(how or ""),
            "round_type": str(round_type or "open_candidate_review"),
            "previous_user_feedback": previous_user_feedback if previous_user_feedback else {},
            "focus_candidates": list(focus_candidates),
            "advisor_profiles": list(advisor_profiles or []),
            "advisor_feedbacks": feedbacks,
            "discussion_memory": list(discussion_memory),
            "advisor_discussion_rounds": int(advisor_discussion_rounds),
            "committee_packet": committee_packet,
        }
