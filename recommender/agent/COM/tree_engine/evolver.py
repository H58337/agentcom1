import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from recommender.agent.COM.tree_engine.evolution_analyzer import EvolutionAnalyzer
from recommender.agent.COM.tree_engine.user_policy import UserPolicyStore, DEFAULT_TRIGGER_SIGNATURES, TRIGGER_SIGNATURE_PRIORITY
from recommender.agent.COM.tree_engine.utils import append_jsonl, dump_json, load_json, load_jsonl, safe_read_text
from recommender.agent.COM.utils.com_agent import get_last_llm_request_usage, llm_request, update_llm_prompt_trace_context


class TrainOnlyEvolver:
    def __init__(self):
        self.analyzer = EvolutionAnalyzer()

    @staticmethod
    def _norm(value):
        return " ".join(str(value or "").strip().lower().split())

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
    def _path_prefixes(path):
        parts = [
            str((path or {}).get('why', "") or ""),
            str((path or {}).get("what", "") or ""),
            str((path or {}).get("who", "") or ""),
            str((path or {}).get("how", "") or ""),
        ]
        prefixes = []
        for idx in range(1, len(parts) + 1):
            if all(parts[:idx]):
                prefixes.append(" -> ".join(parts[:idx]))
        return prefixes

    @staticmethod
    def _path_prefixes_fine(path):
        parts = [
            str((path or {}).get('why', "") or ""),
            str((path or {}).get("what", "") or ""),
            str((path or {}).get("who_fine", "") or (path or {}).get("who", "") or ""),
            str((path or {}).get("how", "") or ""),
        ]
        prefixes = []
        for idx in range(1, len(parts) + 1):
            if all(parts[:idx]):
                prefixes.append(" -> ".join(parts[:idx]))
        return prefixes

    def _log_communication_evolution_gate(self, engine, user_raw, trace_context, outcome, event, **fields):
        trace_context = dict(trace_context or {})
        path = dict(trace_context.get("path", {}) or {})
        payload = {
            "event": str(event or ""),
            "ts": int(time.time()),
            "user_id": str(trace_context.get("user_id", "") or user_raw or ""),
            "outcome_signal": str(outcome or ""),
            "path": {
                'why': str(path.get('why', "") or ""),
                "what": str(path.get("what", "") or ""),
                "who": str(path.get("who", "") or ""),
                "how": str(path.get("how", "") or ""),
                "user_task": self._short_text(path.get("user_task", ""), 220),
                "unmapped_task": bool(path.get("unmapped_task", False)),
            },
            **fields,
        }
        try:
            append_jsonl(engine.public_tree_store.index_dir / "communication_evolution_gate.jsonl", self._json_safe(payload))
        except Exception:
            pass
        decision = str(fields.get("decision", "") or "")
        reason = str(fields.get("reason", "") or fields.get("skip_reason", "") or "")
        suffix = f" decision={decision}" if decision else ""
        if reason:
            suffix += f" reason={self._short_text(reason, 140)}"
        print(
            f"[com][evolution-gate] user={payload['user_id']} outcome={payload['outcome_signal']} "
            f"event={payload['event']}{suffix}"
        )

    @staticmethod
    def _rule_bucket_for_status(status):
        status = str(status or "candidate")
        return {
            "active": "active_rules",
            "candidate": "candidate_rules",
            "risky": "risky_rules",
            "inactive": "inactive_rules",
        }.get(status, "candidate_rules")

    def _find_rule(self, layer_obj, rule_text):
        target = self._norm(rule_text)
        if not target:
            return None, None, None
        for bucket in ["active_rules", "candidate_rules", "risky_rules", "inactive_rules"]:
            for idx, row in enumerate(list(layer_obj.get(bucket, []) or [])):
                if self._norm(row.get("rule", "")) == target:
                    return bucket, idx, row
        return None, None, None

    def _move_rule(self, layer_obj, source_bucket, idx, row):
        status = str(row.get("status", "") or "candidate")
        target_bucket = self._rule_bucket_for_status(status)
        if source_bucket == target_bucket:
            layer_obj[source_bucket][idx] = row
            return
        rows = list(layer_obj.get(source_bucket, []) or [])
        if 0 <= idx < len(rows):
            rows.pop(idx)
        layer_obj[source_bucket] = rows
        layer_obj.setdefault(target_bucket, []).append(row)

    def _apply_single_user_rule_update(self, policy, user_diag, diagnosis):
        user_diag = dict(user_diag or {})
        layer_name = UserPolicyStore.canonical_skill_layer(
            user_diag.get("target_layer", "item_selection_skill")
        )
        operation = str(user_diag.get("operation", "") or "")
        rule_text = str(user_diag.get("rule", "") or "").strip()
        if not rule_text:
            return policy

        layer_obj = dict(policy.get(layer_name, {}) or {})
        for bucket in ["active_rules", "candidate_rules", "risky_rules", "inactive_rules"]:
            layer_obj.setdefault(bucket, [])

        bucket, idx, existing = self._find_rule(layer_obj, rule_text)
        if operation == "reinforce":
            if existing is None:
                confidence = self._confidence_value_for_label("medium")
                existing = {
                    "rule": rule_text,
                    "confidence": confidence,
                    "status": self._status_for_confidence(confidence),
                    "reinforce_count": 0,
                    "weaken_count": 0,
                }
                bucket = self._rule_bucket_for_status(existing["status"])
                layer_obj[bucket].append(existing)
                idx = len(layer_obj[bucket]) - 1
            existing = dict(existing)
            existing["confidence"] = self._promote_confidence(existing.get("confidence", 0.50))
            existing["reinforce_count"] = int(existing.get("reinforce_count", 0) or 0) + 1
            existing["status"] = self._status_for_confidence(existing["confidence"])
            self._move_rule(layer_obj, bucket, idx, existing)
        elif operation in ["discover", "weaken"]:
            if existing is None:
                confidence = self._confidence_value_for_label("low")
                status = self._status_for_confidence(confidence)
                bucket = self._rule_bucket_for_status(status)
                layer_obj[bucket].append(
                    {
                        "rule": rule_text,
                        "confidence": confidence,
                        "status": status,
                        "reinforce_count": 0,
                        "weaken_count": 1 if operation == "weaken" else 0,
                    }
                )
            else:
                existing = dict(existing)
                if operation == "discover":
                    existing["confidence"] = self._promote_confidence(existing.get("confidence", 0.50))
                    existing["reinforce_count"] = int(existing.get("reinforce_count", 0) or 0) + 1
                    existing["weaken_count"] = max(0, int(existing.get("weaken_count", 0) or 0) - 1)
                    existing["status"] = self._status_for_confidence(existing["confidence"])
                else:
                    existing["confidence"] = self._demote_confidence(existing.get("confidence", 0.50))
                    existing["weaken_count"] = int(existing.get("weaken_count", 0) or 0) + 1
                    existing["status"] = self._status_for_confidence(existing["confidence"])
                self._move_rule(layer_obj, bucket, idx, existing)

        compressed = dict(layer_obj)
        compressed.update(self._compress_layer(layer_obj))
        policy[layer_name] = compressed
        return policy

    def _sync_item_preferences_from_rules(self, original_layer, compressed_layer):
        updated = dict(original_layer or {})
        updated.update(dict(compressed_layer or {}))
        preferences = list(updated.get("preferences", []) or [])
        seen = {" ".join(str(row.get("attribute", "") or "").lower().split()) for row in preferences if isinstance(row, dict)}
        for bucket in ["active_rules", "candidate_rules"]:
            for row in list(updated.get(bucket, []) or []):
                rule = " ".join(str((row or {}).get("rule", "") or "").split())
                if not rule:
                    continue
                cleaned = self._clean_preference_attribute(rule)
                if not cleaned:
                    continue
                if len(cleaned) > 60:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                conf = float((row or {}).get("confidence", 0.50) or 0.50)
                preferences.append(
                    {
                        "attribute": cleaned,
                        "confidence": conf,
                        "confidence_label": "high" if conf >= 0.66 else ("medium" if conf >= 0.45 else "low"),
                        "source": "evolution",
                        "evidence_artists": [],
                        "evidence": "",
                        "status": str((row or {}).get("status", "candidate") or "candidate"),
                        "reinforce_count": int((row or {}).get("reinforce_count", 0) or 0),
                        "weaken_count": int((row or {}).get("weaken_count", 0) or 0),
                    }
                )
        updated["preferences"] = self._consolidate_preferences(preferences, limit=12)
        updated.setdefault("recent_signals", [])
        updated.setdefault("decision_style", "history-cluster grounded selection with minority-cluster preservation")
        return updated

    @staticmethod
    def _clean_preference_attribute(text):
        text = " ".join(str(text or "").strip().split())
        if not text:
            return ""
        text = re.sub(
            r"^(Reinforce transferable item-selection signal|Add transferable item-selection signal|Weaken over-bias)\s*:\s*",
            "",
            text,
            flags=re.I,
        )
        text = re.split(r"\b(Evidence|Reason)\s*:", text, maxsplit=1, flags=re.I)[0]
        text = re.sub(r"\s+", " ", text).strip(" .;:")
        return text

    @staticmethod
    def _preference_family_key(text):
        text = TrainOnlyEvolver._clean_preference_attribute(text).lower()
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text)
        stop = {
            "a", "an", "and", "are", "as", "by", "for", "from", "in", "is", "of", "or", "the", "to", "with",
            "user", "users", "candidate", "candidates", "item", "items", "artist", "artists", "music",
            "preference", "preferences", "signal", "signals", "selection", "transferable", "positive",
            "future", "current", "same", 'why', "history", "evidence", "cluster", "clusters", "style", "styles",
            "grounded", "preserve", "supported",
        }
        kept = [tok for tok in tokens if tok not in stop and len(tok) > 1]
        if not kept:
            kept = [tok for tok in tokens if len(tok) > 1]
        return " ".join(kept[:8])

    @classmethod
    def _preferences_similar(cls, left, right):
        left_key = cls._preference_family_key(left)
        right_key = cls._preference_family_key(right)
        if not left_key or not right_key:
            return False
        if left_key == right_key:
            return True
        if len(left_key) >= 12 and len(right_key) >= 12 and (left_key in right_key or right_key in left_key):
            return True
        left_tokens = set(left_key.split())
        right_tokens = set(right_key.split())
        if not left_tokens or not right_tokens:
            return False
        inter = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        overlap = inter / max(1, min(len(left_tokens), len(right_tokens)))
        jaccard = inter / max(1, union)
        return overlap >= 0.70 or jaccard >= 0.55

    @staticmethod
    def _confidence_label(value):
        try:
            value = float(value)
        except Exception:
            value = 0.50
        if value >= 0.66:
            return "high"
        if value >= 0.45:
            return "medium"
        return "low"

    @staticmethod
    def _bounded_confidence(value, default=0.42, low=0.05, high=0.95):
        try:
            value = float(value)
        except Exception:
            value = float(default)
        try:
            low = float(low)
            high = float(high)
        except Exception:
            low, high = 0.05, 0.95
        if low > high:
            low, high = high, low
        return round(max(low, min(high, value)), 3)

    @classmethod
    def _confidence_value_for_label(cls, label):
        label = str(label or "").strip().lower()
        if label == "high":
            return 0.75
        if label == "medium":
            return 0.55
        return 0.35

    @classmethod
    def _promote_confidence(cls, value):
        label = cls._confidence_label(value)
        if label == "low":
            return cls._confidence_value_for_label("medium")
        if label == "medium":
            return cls._confidence_value_for_label("high")
        return cls._confidence_value_for_label("high")

    @classmethod
    def _demote_confidence(cls, value):
        label = cls._confidence_label(value)
        if label == "high":
            return cls._confidence_value_for_label("medium")
        if label == "medium":
            return cls._confidence_value_for_label("low")
        return cls._confidence_value_for_label("low")

    @classmethod
    def _status_for_confidence(cls, value):
        return "active" if cls._confidence_label(value) in ["medium", "high"] else "candidate"

    @staticmethod
    def _merge_text(left, right, max_len=420):
        left = " ".join(str(left or "").split())
        right = " ".join(str(right or "").split())
        if not left:
            return right[:max_len]
        if not right or right.lower() in left.lower():
            return left[:max_len]
        if left.lower() in right.lower():
            return right[:max_len]
        return f"{left}; {right}"[:max_len]

    @staticmethod
    def _merge_artists(left, right, limit=10):
        out = []
        seen = set()
        for value in list(left or []) + list(right or []):
            value = str(value or "").strip()
            key = value.lower()
            if not value or key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= limit:
                break
        return out

    def _find_preference_index(self, preferences, attribute):
        for idx, row in enumerate(list(preferences or [])):
            if not isinstance(row, dict):
                continue
            existing = str(row.get("attribute", "") or row.get("rule", "") or "")
            if self._preferences_similar(attribute, existing):
                return idx
        return None

    def _merge_preference_row(self, existing, incoming, operation):
        existing = dict(existing or {})
        incoming = dict(incoming or {})
        old_conf = float(existing.get("confidence", 0.50) or 0.50)
        new_conf = float(incoming.get("confidence", 0.50) or 0.50)
        if operation == "reinforce":
            confidence = max(self._promote_confidence(old_conf), new_conf)
            existing["reinforce_count"] = int(existing.get("reinforce_count", 0) or 0) + 1
            existing["weaken_count"] = max(0, int(existing.get("weaken_count", 0) or 0) - 1)
            existing["learning_priority"] = max(int(existing.get("learning_priority", 0) or 0), int(incoming.get("learning_priority", 0) or 0))
            existing["status"] = self._status_for_confidence(confidence)
        elif operation == "discover":
            confidence = max(old_conf, new_conf)
            existing["reinforce_count"] = int(existing.get("reinforce_count", 0) or 0) + 1
            existing["weaken_count"] = max(0, int(existing.get("weaken_count", 0) or 0) - 1)
            existing["learning_priority"] = max(int(existing.get("learning_priority", 0) or 0), int(incoming.get("learning_priority", 0) or 0))
            existing["status"] = self._status_for_confidence(confidence)
        else:
            confidence = self._demote_confidence(old_conf)
            existing["weaken_count"] = int(existing.get("weaken_count", 0) or 0) + 1
            existing["status"] = self._status_for_confidence(confidence)
        existing["confidence"] = confidence
        existing["confidence_label"] = self._confidence_label(confidence)
        existing["source"] = self._merge_text(existing.get("source", ""), incoming.get("source", ""), max_len=120)
        existing["evidence"] = self._merge_text(existing.get("evidence", ""), incoming.get("evidence", ""))
        existing["evidence_artists"] = self._merge_artists(existing.get("evidence_artists", []), incoming.get("evidence_artists", []))
        if len(str(incoming.get("attribute", "") or "")) < len(str(existing.get("attribute", "") or "")) or not existing.get("attribute"):
            existing["attribute"] = incoming.get("attribute", existing.get("attribute", ""))
        return existing

    def _consolidate_preferences(self, preferences, limit=12):
        merged = []
        for row in list(preferences or []):
            if not isinstance(row, dict):
                continue
            row = dict(row)
            attribute = self._clean_preference_attribute(row.get("attribute", "") or row.get("rule", ""))
            if not attribute:
                continue
            if UserPolicyStore._is_generic_item_protocol_rule(attribute):
                continue
            if len(attribute) > 80:
                attribute = attribute[:77] + "..."
            row["attribute"] = attribute
            try:
                row["confidence"] = float(row.get("confidence", 0.50) or 0.50)
            except Exception:
                row["confidence"] = 0.50
            row["confidence_label"] = self._confidence_label(row["confidence"])
            idx = self._find_preference_index(merged, attribute)
            if idx is None:
                merged.append(row)
            else:
                merged[idx] = self._merge_preference_row(merged[idx], row, "reinforce")

        def weight(row):
            return (
                float(row.get("confidence", 0.0) or 0.0)
                + 0.035 * int(row.get("learning_priority", 0) or 0)
                + 0.010 * min(3, int(row.get("reinforce_count", 0) or 0))
                - 0.020 * min(3, int(row.get("weaken_count", 0) or 0))
            )

        merged.sort(key=weight, reverse=True)
        return merged[:limit]

    def _diff_row_to_preference(self, row, operation, default_confidence, source):
        row = dict(row or {})
        attribute = self._clean_preference_attribute(row.get("attribute") or row.get("preference") or row.get("signal") or row.get("rule") or "")
        if len(attribute) > 60:
            return None
        evidence = " ".join(str(row.get("evidence") or row.get("reason") or row.get("rationale") or "").split())
        if not attribute:
            return None
        confidence = self._bounded_confidence(row.get("confidence", default_confidence), default=default_confidence)
        if operation == "discover":
            confidence = self._confidence_value_for_label("low")
        elif operation == "reinforce":
            confidence = max(confidence, self._confidence_value_for_label("medium"))
        elif operation == "weaken":
            confidence = self._confidence_value_for_label("low")
        return {
            "attribute": attribute,
            "confidence": confidence,
            "confidence_label": self._confidence_label(confidence),
            "source": source,
            "evidence_artists": [str(x) for x in list(row.get("evidence_artists", []) or []) if str(x or "").strip()][:10],
            "evidence": evidence,
            "status": self._status_for_confidence(confidence),
            "reinforce_count": 1 if operation == "reinforce" else 0,
            "weaken_count": 1 if operation == "weaken" else 0,
            "learning_priority": 1 if operation == "reinforce" else 0,
        }

    def _apply_incremental_preference_update(self, policy, diagnosis):
        inc = dict((diagnosis or {}).get("incremental_update", {}) or {})
        inc = self._sanitize_incremental_update(inc, (diagnosis or {}).get("forbidden_item_names", []))
        if not any(list(inc.get(key, []) or []) for key in ["reinforced", "new_preferences", "recent_signals", "weakened"]):
            return policy, {"applied": False, "reason": "no_incremental_update"}

        updated = dict(policy or {})
        item_skill = dict(updated.get("item_selection_skill", {}) or {})
        preferences = list(item_skill.get("preferences", []) or [])
        recent_signals = list(item_skill.get("recent_signals", []) or [])
        summary = {
            "applied": True,
            "reinforced": [],
            "new_preferences": [],
            "recent_signals": [],
            "weakened": [],
            "merged_similar_preferences": 0,
        }

        def upsert_preference(row, operation, bucket_name):
            nonlocal preferences
            incoming = self._diff_row_to_preference(
                row,
                operation=operation,
                default_confidence=0.55 if operation == "reinforce" else (0.35 if operation == "discover" else 0.35),
                source=f"evolution_{bucket_name}",
            )
            if not incoming:
                return
            idx = self._find_preference_index(preferences, incoming["attribute"])
            if operation == "weaken" and idx is None:
                summary[bucket_name].append({"attribute": incoming["attribute"], "action": "skipped_missing_existing_preference", "confidence": 0.0})
                return
            if idx is None:
                preferences.append(incoming)
                summary[bucket_name].append({"attribute": incoming["attribute"], "action": "added", "confidence": incoming["confidence"]})
            else:
                merge_operation = "reinforce" if operation == "discover" else operation
                preferences[idx] = self._merge_preference_row(preferences[idx], incoming, merge_operation)
                summary[bucket_name].append({"attribute": incoming["attribute"], "action": "merged", "confidence": preferences[idx].get("confidence", 0.0)})

        for row in list(inc.get("reinforced", []) or []):
            upsert_preference(row, "reinforce", "reinforced")
        for row in list(inc.get("new_preferences", []) or []):
            upsert_preference(row, "discover", "new_preferences")
        for row in list(inc.get("weakened", []) or []):
            upsert_preference(row, "weaken", "weakened")

        for row in list(inc.get("recent_signals", []) or []):
            incoming = self._diff_row_to_preference(row, operation="discover", default_confidence=0.35, source="evolution_recent_signal")
            if not incoming:
                continue
            idx = self._find_preference_index(recent_signals, incoming["attribute"])
            if idx is None:
                recent_signals.append(incoming)
                summary["recent_signals"].append({"attribute": incoming["attribute"], "action": "added", "confidence": incoming["confidence"]})
            else:
                recent_signals[idx] = self._merge_preference_row(recent_signals[idx], incoming, "reinforce")
                summary["recent_signals"].append({"attribute": incoming["attribute"], "action": "merged", "confidence": recent_signals[idx].get("confidence", 0.0)})

        before_count = len(preferences)
        preferences = self._consolidate_preferences(preferences, limit=12)
        recent_signals = self._consolidate_preferences(recent_signals, limit=6)
        summary["merged_similar_preferences"] = max(0, before_count - len(preferences))
        item_skill["preferences"] = preferences
        item_skill["recent_signals"] = recent_signals
        item_skill.setdefault("decision_style", "history-cluster grounded selection with minority-cluster preservation")
        updated["item_selection_skill"] = item_skill
        return updated, summary

    def _apply_user_rule_update(self, policy, diagnosis):
        user_diag = dict((diagnosis or {}).get("user_skill_diagnosis", {}) or {})
        policy = self._apply_single_user_rule_update(policy, user_diag, diagnosis)
        for extra in list(user_diag.get("additional_updates", []) or []):
            if isinstance(extra, dict):
                policy = self._apply_single_user_rule_update(policy, extra, diagnosis)
        return policy

    def _apply_absorption_case_update(self, policy, diagnosis, trace_context):
        updated = dict(policy or {})
        skill = dict(updated.get("communication_absorption_skill", {}) or {})
        skill.setdefault("version", 1)
        skill.setdefault("active_rules", [])
        skill.setdefault("candidate_rules", [])
        skill.setdefault("risky_rules", [])
        skill.setdefault("inactive_rules", [])
        skill.setdefault("ignored_advisor_signal_cases", [])
        decision = self._normalize_evolution_decision((diagnosis or {}).get("evolution_decision", {}))
        specific_rule = self._short_text(
            decision.get("user_absorption_rule", "")
            or ((diagnosis or {}).get("user_skill_diagnosis", {}) or {}).get("rule", "")
            or decision.get("reason", "")
            or 'Why advisors give clear candidate-specific guidance, compare it against the current choice before finalizing.',
            360,
        )
        active_rules = [dict(row) for row in list(skill.get("active_rules", []) or []) if isinstance(row, dict)]
        if specific_rule:
            key = " ".join(specific_rule.lower().split())
            existing = None
            for row in active_rules:
                if " ".join(str(row.get("rule", "") or "").lower().split()) == key:
                    existing = row
                    break
            if existing is None:
                active_rules.insert(
                    0,
                    {
                        "rule": specific_rule,
                        "confidence": 0.58,
                        "status": "active",
                        "reinforce_count": 1,
                        "weaken_count": 0,
                        "source": "llm0_user_absorption_update",
                    },
                )
            else:
                existing["reinforce_count"] = int(existing.get("reinforce_count", 0) or 0) + 1
                try:
                    existing["confidence"] = min(0.90, float(existing.get("confidence", 0.58) or 0.58) + 0.03)
                except Exception:
                    existing["confidence"] = 0.61
                existing["status"] = "active"
        skill["active_rules"] = active_rules[:8]
        case = {
            "diagnosis_id": str((diagnosis or {}).get("diagnosis_id", "") or ""),
            "outcome_signal": str((diagnosis or {}).get("outcome_signal", "") or ""),
            "primary_failure_level": str((diagnosis or {}).get("primary_failure_level", "") or ""),
            "failure_attribution": str(
                (diagnosis or {}).get("user_skill_failure_attribution")
                or (diagnosis or {}).get("failure_attribution", "")
                or ""
            ),
            "path": dict((trace_context or {}).get("path", {}) or {}),
            "reason": self._short_text(
                (diagnosis or {}).get("path_effect_explanation")
                or (diagnosis or {}).get("communication_reflection_summary")
                or decision.get("reason", "")
                or ((diagnosis or {}).get("user_skill_diagnosis", {}) or {}).get("problem", ""),
                220,
            ),
            "learned_rule": specific_rule,
        }
        rows = [dict(row) for row in list(skill.get("ignored_advisor_signal_cases", []) or []) if isinstance(row, dict)]
        key = "|".join([case["diagnosis_id"], case["outcome_signal"], case["primary_failure_level"]])
        existing_keys = {
            "|".join([
                str(row.get("diagnosis_id", "") or ""),
                str(row.get("outcome_signal", "") or ""),
                str(row.get("primary_failure_level", "") or ""),
            ])
            for row in rows
        }
        if key.strip("|") and key not in existing_keys:
            rows.append(case)
        skill["ignored_advisor_signal_cases"] = rows[-4:]
        try:
            skill["confidence"] = min(0.90, max(0.50, float(skill.get("confidence", 0.50) or 0.50) + 0.04))
        except Exception:
            skill["confidence"] = 0.54
        updated["communication_absorption_skill"] = skill
        return updated

    @staticmethod
    def _dedupe_rules(rows):
        out = []
        seen = set()
        for row in rows or []:
            key = " ".join(str(row.get("rule", "") or "").strip().lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _compress_layer(self, layer_obj):
        limits = {
            "active_rules": 8,
            "candidate_rules": 12,
            "risky_rules": 5,
            "inactive_rules": 30,
        }
        out = {}
        for bucket, limit in limits.items():
            rows = self._dedupe_rules(list(layer_obj.get(bucket, []) or []))
            rows.sort(key=lambda row: (-float(row.get("confidence", 0.0) or 0.0), -int(row.get("reinforce_count", 0) or 0)))
            out[bucket] = rows[:limit]
        return out

    @staticmethod
    def _route_trigger_signature_from_path(path):
        path = dict(path or {})
        rows = []
        for row in list(path.get("matched_why", []) or []):
            row = str(row or "").strip()
            if row and row not in ["skip", "none"] and row not in rows:
                rows.append(row)
        if not rows:
            row = str(path.get('why', "") or "").strip()
            if row and row not in ["skip", "none"]:
                rows.append(row)
        priority = {name: idx for idx, name in enumerate(TRIGGER_SIGNATURE_PRIORITY)}
        rows.sort(key=lambda x: priority.get(x, 99))
        return "+".join(rows) if rows else "default"

    @staticmethod
    def _route_scope(path, level):
        path = dict(path or {})
        signature = TrainOnlyEvolver._route_trigger_signature_from_path(path)
        what = str(path.get("what", "") or "none")
        how = str(path.get("how", "") or "")
        if level == "what":
            return signature
        if level == "how":
            return what or "none"
        if level == "who":
            return how or "default"
        return signature

    @staticmethod
    def _move_to_front(order, node):
        node = str(node or "").strip()
        rows = [str(x) for x in list(order or []) if str(x).strip() and str(x) != node]
        return ([node] if node else []) + rows

    @staticmethod
    def _move_to_back(order, node):
        node = str(node or "").strip()
        rows = [str(x) for x in list(order or []) if str(x).strip() and str(x) != node]
        return rows + ([node] if node else [])

    def _ensure_route_skill(self, policy):
        policy = dict(policy or {})
        route = dict(policy.get("communication_route_skill", {}) or {})
        if route:
            route["version"] = max(2, int(route.get("version", 1) or 1) if str(route.get("version", 1) or 1).isdigit() else 2)
            route.setdefault("template_id", "migrated-route")
            route.setdefault("template_features", {})
            route.setdefault("signature_order", list(DEFAULT_TRIGGER_SIGNATURES))
            old_what = dict(route.get("what_by_signature", {}) or {})
            old_how = dict(route.get("how_by_signature_what", {}) or {})
            old_who = dict(route.get("who_by_signature_what_how", {}) or {})
            route["what_by_why"] = dict(route.get("what_by_why", {}) or old_what or {})
            if not route.get("how_by_what"):
                migrated_how = {}
                for key, value in old_how.items():
                    parts = str(key or "").split("|")
                    what_key = parts[-1] if parts else str(key or "")
                    if what_key and what_key not in migrated_how:
                        migrated_how[what_key] = list(value or [])
                route["how_by_what"] = migrated_how
            if not route.get("who_by_how"):
                migrated_who = {}
                for key, value in old_who.items():
                    parts = str(key or "").split("|")
                    how_key = parts[-1] if parts else str(key or "")
                    if how_key and how_key not in migrated_who:
                        migrated_who[how_key] = list(value or [])
                route["who_by_how"] = migrated_who
            route.setdefault("child_order_memory", {})
            route["what_by_signature"] = {}
            route["how_by_signature_what"] = {}
            route["who_by_signature_what_how"] = {}
            route.setdefault("demotions", [])
            route.setdefault("unmapped_task_memory", [])
            route.setdefault("exploration_slots", [])
            route.setdefault("exploration_history", [])
            policy["communication_route_skill"] = route
            return policy, route
        # Minimal fallback route skill; full initialization is handled by UserPolicyStore.
        route = {
            "version": 2,
            "template_id": "runtime-minimal",
            "template_features": {},
            "signature_order": list(DEFAULT_TRIGGER_SIGNATURES),
            "what_by_why": {
                "internal-prior-conflict+candidate-conflict": ["reasoning_check", "compare_remaining_candidates", "reduce_hesitation_set", "none"],
                "candidate-conflict": ["reduce_hesitation_set", "compare_remaining_candidates", "reasoning_check", "none"],
                "default": ["reduce_hesitation_set", "compare_remaining_candidates", "none"],
            },
            "how_by_what": {
                "reasoning_check": ["multi-competitive", "multi-cooperative", "single-advisor"],
                "compare_remaining_candidates": ["multi-competitive", "multi-cooperative", "single-advisor"],
                "reduce_hesitation_set": ["multi-cooperative", "multi-competitive", "single-advisor"],
                "none": ["multi-cooperative", "single-advisor", "multi-competitive"],
                "default": ["multi-cooperative", "multi-competitive", "single-advisor"],
            },
            "who_by_how": {
                "multi-competitive": ["similar-users", "experienced-users", "trusted-advisors", "topk-advisors"],
                "multi-cooperative": ["similar-users", "experienced-users", "trusted-advisors", "topk-advisors"],
                "single-advisor": ["similar-users", "experienced-users", "trusted-advisors", "topk-advisors"],
                "default": ["similar-users", "experienced-users", "trusted-advisors", "topk-advisors"],
            },
            "child_order_memory": {},
            "what_by_signature": {},
            "how_by_signature_what": {},
            "who_by_signature_what_how": {},
            "demotions": [],
            "unmapped_task_memory": [],
            "exploration_slots": [],
            "exploration_history": [],
        }
        policy["communication_route_skill"] = route
        return policy, route

    def _upsert_route_exploration_row(self, rows, row, limit=32):
        rows = [dict(x) for x in list(rows or []) if isinstance(x, dict)]
        row = dict(row or {})
        key = "|".join(str(row.get(field, "") or "").strip().lower() for field in ["level", "scope", "node"])
        if not key.strip("|"):
            return rows[-limit:]
        for existing in rows:
            existing_key = "|".join(str(existing.get(field, "") or "").strip().lower() for field in ["level", "scope", "node"])
            if existing_key != key:
                continue
            existing["trial_count"] = int(existing.get("trial_count", 0) or 0) + int(row.get("trial_count", 1) or 1)
            for field in ["helpful_count", "harmful_count", "ineffective_count"]:
                existing[field] = int(existing.get(field, 0) or 0) + int(row.get(field, 0) or 0)
            for field in ["last_effect", "status", "parent_node", "reason", "source", "placement", "trial_priority"]:
                if str(row.get(field, "") or "").strip():
                    existing[field] = row.get(field)
            return rows[-limit:]
        rows.append(row)
        return rows[-limit:]

    def _apply_route_operation(self, route, operation):
        route = dict(route or {})
        op = str((operation or {}).get("operation", "") or "").strip().lower()
        level = str((operation or {}).get("level", "") or "").strip().lower()
        scope = str((operation or {}).get("scope", "") or "").strip()
        node = str((operation or {}).get("node", "") or "").strip()
        if "|" in scope:
            parts = [x for x in scope.split("|") if x]
            if level == "what" and parts:
                scope = parts[0]
            elif level in ["how", "who"] and parts:
                scope = parts[-1]
        if not op:
            return route, False
        if op == "record_unmapped_task":
            route["unmapped_task_memory"] = self._append_unique_rows(
                route.get("unmapped_task_memory", []),
                [{
                    "raw_task": str((operation or {}).get("raw_task", "") or (operation or {}).get("task_text_summary", "") or "")[:320],
                    "task_text_summary": str((operation or {}).get("task_text_summary", "") or (operation or {}).get("reason", "") or "")[:240],
                    "round_index": int((operation or {}).get("round_index", 0) or 0),
                    "trigger_signature": str((operation or {}).get("trigger_signature", "") or ""),
                    "mapped_what": "none",
                    "suggested_future_what": str((operation or {}).get("suggested_future_what", "") or (operation or {}).get("node", "") or "unmapped-followup-task"),
                    "why_unmapped": str((operation or {}).get("why_unmapped", "") or (operation or {}).get("reason", "") or ""),
                    "count": 1,
                    "source": "communication_route_evolution",
                }],
                key_fields=["mapped_what", "suggested_future_what", "task_text_summary", "trigger_signature"],
                limit=24,
            )
            return route, True
        if op == "record_exploration_result":
            effect = str((operation or {}).get("effect", "") or "")
            level = str((operation or {}).get("level", "") or "").strip() or ("how" if node.startswith("multi-") else "")
            parent_node = str((operation or {}).get("parent_node", "") or "")
            if not parent_node and "/" in node:
                parent_node = node.split("/", 1)[0]
            row = {
                "level": level,
                "scope": scope,
                "node": node,
                "parent_node": parent_node,
                "status": str((operation or {}).get("status", "") or ""),
                "trial_count": 1,
                "helpful_count": int(effect == "helpful"),
                "harmful_count": int(effect == "harmful"),
                "ineffective_count": int(effect == "ineffective"),
                "last_effect": effect,
                "reason": str((operation or {}).get("reason", "") or "sprout/child node trial result"),
                "source": "communication_route_evolution",
            }
            if effect == "helpful":
                row["placement"] = "before_parent"
                row["trial_priority"] = "high"
            route["exploration_history"] = self._upsert_route_exploration_row(
                route.get("exploration_history", []),
                row,
                limit=32,
            )
            route["exploration_slots"] = self._upsert_route_exploration_row(
                route.get("exploration_slots", []),
                row,
                limit=32,
            )
            return route, True
        if op == "record_child_order_inheritance":
            bucket = str((operation or {}).get("bucket", "") or "").strip()
            child_scope = str((operation or {}).get("child_scope", "") or "").strip()
            parent_scope = str((operation or {}).get("parent_scope", "") or "").strip()
            if bucket not in ["how_by_what", "who_by_how"] or not child_scope or not parent_scope:
                return route, False
            table = dict(route.get(bucket, {}) or {})
            if table.get(child_scope):
                return route, False
            inherited_order = [
                str(x)
                for x in list((operation or {}).get("inherited_order", []) or table.get(parent_scope, []) or [])
                if str(x).strip()
            ]
            if not inherited_order:
                return route, False
            table[child_scope] = inherited_order
            route[bucket] = table
            memory = dict(route.get("child_order_memory", {}) or {})
            memory.setdefault(
                f"{bucket}|{child_scope}",
                {
                    "inherited_from": parent_scope,
                    "created_by": "child_route_inheritance",
                    "order": list(inherited_order),
                },
            )
            route["child_order_memory"] = memory
            return route, True
        if level not in ["what", "how", "who"] or not scope or not node:
            return route, False
        bucket = {
            "what": "what_by_why",
            "how": "how_by_what",
            "who": "who_by_how",
        }[level]
        table = dict(route.get(bucket, {}) or {})
        order = list(table.get(scope, []) or [])
        if op in ["promote", "insert_before"]:
            before = str((operation or {}).get("before", "") or "").strip()
            order = [x for x in order if x != node]
            if before and before in order:
                idx = order.index(before)
                order = order[:idx] + [node] + order[idx:]
            else:
                order = self._move_to_front(order, node)
            table[scope] = order
            route[bucket] = table
            return route, True
        if op == "append_if_missing":
            if node not in order:
                order.append(node)
                table[scope] = order
                route[bucket] = table
                return route, True
            return route, False
        if op == "demote":
            parent_node = node.split("/", 1)[0] if "/" in node else ""
            order = [x for x in order if x != node]
            if parent_node and parent_node in order:
                idx = order.index(parent_node) + 1
                table[scope] = order[:idx] + [node] + order[idx:]
            else:
                table[scope] = order + ([node] if node else [])
            route[bucket] = table
            route["demotions"] = self._append_unique_rows(
                route.get("demotions", []),
                [{
                    "level": level,
                    "scope": scope,
                    "node": node,
                    "reason": str((operation or {}).get("reason", "") or ""),
                    "source": "communication_route_evolution",
                    "count": 1,
                }],
                key_fields=["level", "scope", "node"],
                limit=32,
            )
            return route, True
        return route, False

    def _derive_route_operations(self, path, diagnosis):
        path = dict(path or {})
        diagnosis = dict(diagnosis or {})
        outcome = str(diagnosis.get("outcome_signal", "") or "")
        operations = []
        if outcome not in ["WT", "TW", "WW"]:
            return operations
        what = str(path.get("what", "") or "none")
        how = str(path.get("how", "") or "")
        who = str(path.get("who", "") or "")
        if outcome == "WT":
            if what and "/" in what:
                operations.append({"operation": "promote", "level": "what", "scope": self._route_scope(path, "what"), "node": what, "reason": "WT promoted explored what child for this trigger signature"})
            if how:
                operations.append({"operation": "promote", "level": "how", "scope": self._route_scope(path, "how"), "node": how, "reason": "WT communication path helped final decision"})
            if who:
                operations.append({"operation": "promote", "level": "who", "scope": self._route_scope(path, "who"), "node": who, "reason": "WT advisor source helped final decision"})
        elif outcome in ["TW", "WW"]:
            effect = "harmful" if outcome == "TW" else "ineffective"
            if what and "/" in what:
                operations.append({"operation": "demote", "level": "what", "scope": self._route_scope(path, "what"), "node": what, "reason": f"{outcome} explored what child was {effect}"})
            if how:
                operations.append({"operation": "demote", "level": "how", "scope": self._route_scope(path, "how"), "node": how, "reason": f"{outcome} route was {effect}"})
            if outcome == "TW" and who:
                operations.append({"operation": "demote", "level": "who", "scope": self._route_scope(path, "who"), "node": who, "reason": "advisor source contributed to harmful communication"})
        if bool(path.get("unmapped_task", False)) and what == "none":
            operations.append({
                "operation": "record_unmapped_task",
                "raw_task": str(path.get("user_task", "") or "")[:320],
                "task_text_summary": str(path.get("user_task", "") or "")[:240],
                "trigger_signature": self._route_trigger_signature_from_path(path),
                "suggested_future_what": "unmapped-followup-task",
                "reason": "FeedbackToAdvisors did not map to any active what node",
            })
        for row in list(diagnosis.get("unmapped_followup_tasks", []) or []):
            if not isinstance(row, dict):
                continue
            operations.append({
                "operation": "record_unmapped_task",
                "raw_task": str(row.get("raw_task", "") or "")[:320],
                "task_text_summary": str(row.get("raw_task", "") or "")[:240],
                "round_index": int(row.get("round_index", 0) or 0),
                "trigger_signature": str(row.get("trigger_signature", "") or ""),
                "suggested_future_what": "unmapped-followup-task",
                "why_unmapped": str(row.get("why_unmapped", "") or "FeedbackToAdvisors did not map to any active what node"),
                "reason": str(row.get("why_unmapped", "") or "FeedbackToAdvisors did not map to any active what node"),
            })
        if path.get("trial_flag") or path.get("sprout_nodes"):
            for node in list(path.get("sprout_nodes", []) or []):
                raw_node = str(node)
                level_hint = raw_node.split("/", 1)[0] if "/" in raw_node else ""
                node_id = raw_node.split("/", 1)[1] if "/" in raw_node else raw_node
                scope = self._route_scope(path, level_hint) if level_hint in ["what", "how", "who"] else self._route_trigger_signature_from_path(path)
                operations.append({
                    "operation": "record_exploration_result",
                    "level": level_hint,
                    "scope": scope,
                    "node": node_id,
                    "parent_node": self._parent_node_id(node_id),
                    "status": "sprout",
                    "effect": "helpful" if outcome == "WT" else ("harmful" if outcome == "TW" else "ineffective"),
                    "reason": f"{outcome} outcome after trying public-tree child node",
                })
        return operations

    def _derive_child_order_inheritance_operations(self, path):
        operations = []
        for row in list((path or {}).get("child_order_inheritance", []) or []):
            if not isinstance(row, dict):
                continue
            operations.append(
                {
                    "operation": "record_child_order_inheritance",
                    "bucket": str(row.get("bucket", "") or ""),
                    "child_scope": str(row.get("child_scope", "") or ""),
                    "parent_scope": str(row.get("parent_scope", "") or ""),
                    "inherited_order": list(row.get("inherited_order", []) or []),
                }
            )
        return operations

    def _apply_route_diff(self, policy, path, diagnosis):
        updated, route = self._ensure_route_skill(policy)
        outcome = str((diagnosis or {}).get("outcome_signal", "") or "")
        operations = []
        inherited_applied = 0
        inherited_operations = []
        for op in self._derive_child_order_inheritance_operations(path):
            route, changed = self._apply_route_operation(route, op)
            inherited_applied += int(bool(changed))
            if changed:
                inherited_operations.append(dict(op))
        if outcome == "TT":
            updated["communication_route_skill"] = route
            return updated, {
                "applied": bool(inherited_applied),
                "operations": inherited_applied,
                "applied_operations": inherited_operations[:12],
            }
        decision = self._normalize_evolution_decision((diagnosis or {}).get("evolution_decision", {}))
        if decision.get("decision") == "existing_route_reorder":
            operations.extend(self._derive_existing_route_reorder_operations(route, path, diagnosis))
        else:
            operations.extend(self._derive_route_operations(path, diagnosis))
        applied = 0
        applied_operations = list(inherited_operations)
        for op in operations:
            route, changed = self._apply_route_operation(route, op)
            applied += int(bool(changed))
            if changed:
                applied_operations.append(dict(op))
        updated["communication_route_skill"] = route
        total_applied = inherited_applied + applied
        return updated, {"applied": bool(total_applied), "operations": total_applied, "applied_operations": applied_operations[:12]}

    @staticmethod
    def _append_unique_rows(rows, new_rows, key_fields, limit=12):
        out = [dict(x) for x in list(rows or []) if isinstance(x, dict)]
        seen = {
            "|".join(str((row or {}).get(field, "") or "").strip().lower() for field in key_fields)
            for row in out
        }
        for row in list(new_rows or []):
            if not isinstance(row, dict):
                continue
            key = "|".join(str(row.get(field, "") or "").strip().lower() for field in key_fields)
            if not key.strip("|") or key in seen:
                continue
            out.append(dict(row))
            seen.add(key)
        return out[-limit:]

    def _communication_specific_failure(self, trace_context):
        evaluation = dict((trace_context or {}).get("evaluation_result", {}) or {})
        execution = dict((trace_context or {}).get("execution_packet", {}) or {})
        committee = dict(execution.get("committee_packet", {}) or {})
        evidence_summary = dict(committee.get("evidence_summary", {}) or {})
        return bool(
            evaluation.get("advisor_pool_empty")
            or evaluation.get("advisor_pool_rerouted")
            or evidence_summary.get("silent_focus_candidates")
            or evidence_summary.get("missing_advisor_evidence")
            or evidence_summary.get("protocol_issues")
            or committee.get("protocol_issues")
        )

    @staticmethod
    def _normalize_evolution_decision(decision):
        decision = dict(decision or {})
        raw = str(decision.get("decision", "") or "").strip().lower()
        if raw not in ["user_absorption_update", "existing_route_reorder", "public_tree_need"]:
            raw = "public_tree_need"
        confidence = str(decision.get("confidence", "") or "").strip().lower()
        if confidence not in ["high", "medium", "low"]:
            confidence = "medium"
        level = str(decision.get("level", "") or "").strip().lower()
        if level not in ["what", "who", "how", "none"]:
            level = "none"
        return {
            "decision": raw,
            "confidence": confidence,
            "level": level,
            "reason": TrainOnlyEvolver._short_text(decision.get("reason", ""), 420),
            "reference_node": str(decision.get("reference_node", "") or ""),
            "suggested_node_hint": str(decision.get("suggested_node_hint", "") or ""),
            "missing_capability": TrainOnlyEvolver._short_text(decision.get("missing_capability", ""), 360),
            "route_reorder_hint": TrainOnlyEvolver._short_text(decision.get("route_reorder_hint", ""), 300),
            "user_absorption_rule": TrainOnlyEvolver._short_text(decision.get("user_absorption_rule", ""), 360),
            "selected_existing_node": str(
                decision.get("selected_existing_node", "")
                or dict(decision.get("route_reorder", {}) or {}).get("promote_existing_node", "")
                or dict(decision.get("existing_route_reorder", {}) or {}).get("selected_existing_node", "")
                or ""
            ),
            "tree_need_signal": dict(decision.get("tree_need_signal", {}) or {}),
            "not_user_absorption_reason": TrainOnlyEvolver._short_text(decision.get("not_user_absorption_reason", ""), 320),
            "rule_source": str(decision.get("rule_source", "") or ""),
        }

    def _quality_issue_level(self, issue, fallback_level="how"):
        issue = str(issue or "").strip().lower()
        if any(token in issue for token in ["unmapped", "task_mapping", "wrong_what", "task_does_not"]):
            return "what"
        if any(token in issue for token in ["advisor_pool", "advisor_source", "wrong_who", "source_overlap", "homogeneity"]):
            return "who"
        if any(token in issue for token in ["protocol", "counterargument", "rebuttal", "challenge", "candidate_view", "no_candidate"]):
            return "how"
        return fallback_level

    def _quality_issue_family(self, value, fallback="protocol"):
        text = " ".join(str(x or "") for x in (value if isinstance(value, list) else [value])).lower()
        if any(token in text for token in ["unmapped", "task_mapping", "wrong_what", "task_does_not"]):
            return "task_mapping"
        if any(token in text for token in ["advisor_pool", "advisor_source", "wrong_who", "source_overlap", "homogeneity", "diversity"]):
            return "advisor_selection"
        if any(token in text for token in ["counterargument", "rebuttal", "challenge", "protocol", "candidate_view", "no_candidate"]):
            return "protocol"
        if any(token in text for token in ["answer", "speaking", "task_answer"]):
            return "advisor_speaking"
        if any(token in text for token in ["aggregation", "evidence_lost", "summary"]):
            return "aggregation"
        if any(token in text for token in ["candidate_context", "target_missing", "shortlist"]):
            return "candidate_context"
        return str(fallback or "protocol")

    def _public_tree_need_anchor_from_decision(self, decision, path=None, diagnosis=None):
        decision = self._normalize_evolution_decision(decision)
        path = dict(path or {})
        diagnosis = dict(diagnosis or {})
        llm_signal = dict(decision.get("tree_need_signal", {}) or {})
        level = str(decision.get("level", "") or "how")
        if level == "none":
            level = self._quality_issue_level(
                self._diagnosis_quality_issues(diagnosis),
                fallback_level="how",
            )
        reference = str(llm_signal.get("reference_node", "") or decision.get("reference_node", "") or self._reference_node_for_layer(level, path))
        missing = str(llm_signal.get("missing_capability", "") or decision.get("missing_capability", "") or decision.get("reason", "") or "")
        issue_family = self._quality_issue_family(
            [llm_signal.get("suggested_node_hint", ""), decision.get("suggested_node_hint", ""), missing] + self._diagnosis_quality_issues(diagnosis),
            fallback={
                "what": "task_mapping",
                "who": "advisor_selection",
                "how": "protocol",
            }.get(level, "protocol"),
        )
        return self._json_safe(
            {
                "level": level,
                "reference_node": reference,
                "issue_family": issue_family,
                "suggested_node_hint": str(llm_signal.get("suggested_node_hint", "") or decision.get("suggested_node_hint", "") or f"{level}-repair"),
                "missing_capability": self._short_text(missing or issue_family, 420),
                "rule_source": str(decision.get("rule_source", "") or ""),
                "confidence": str(decision.get("confidence", "") or ""),
            }
        )

    def _public_tree_need_anchor_from_signal(self, signal, path=None, diagnosis=None):
        signal = dict(signal or {})
        level = str(signal.get("level", "") or signal.get("layer", "") or "how")
        if level not in {"what", "who", "how"}:
            level = "how"
        path = dict(path or {})
        missing = str(signal.get("missing_capability", "") or signal.get("why_current_nodes_insufficient", "") or signal.get("evidence_pattern", "") or "")
        return self._json_safe(
            {
                "level": level,
                "reference_node": str(signal.get("reference_node", "") or self._reference_node_for_layer(level, path)),
                "issue_family": self._quality_issue_family(
                    [signal.get("suggested_node_hint", ""), signal.get("failure_type", ""), signal.get("failed_stage", ""), missing],
                    fallback={
                        "what": "task_mapping",
                        "who": "advisor_selection",
                        "how": "protocol",
                    }.get(level, "protocol"),
                ),
                "suggested_node_hint": str(signal.get("suggested_node_hint", "") or f"{level}-repair"),
                "missing_capability": self._short_text(missing or str(signal.get("suggested_node_hint", "") or f"{level}-repair"), 420),
                "rule_source": str(signal.get("source", "") or ""),
                "confidence": str(signal.get("confidence", "") or signal.get("support_strength", "") or ""),
            }
        )

    def _public_tree_need_signal_from_decision(self, decision, path, diagnosis):
        decision = self._normalize_evolution_decision(decision)
        path = dict(path or {})
        diagnosis = dict(diagnosis or {})
        llm_signal = dict(decision.get("tree_need_signal", {}) or {})
        level = str(decision.get("level", "") or "how")
        if level == "none":
            level = self._quality_issue_level(
                " ".join(str(x) for x in self._diagnosis_quality_issues(diagnosis)),
                fallback_level="how",
            )
        reference = str(
            llm_signal.get("reference_node", "")
            or decision.get("reference_node", "")
            or self._reference_node_for_layer(level, path)
        )
        hint = str(llm_signal.get("suggested_node_hint", "") or decision.get("suggested_node_hint", "") or "")
        if not hint:
            hint = {
                "what": "unmapped-followup-task",
                "who": "advisor-source-repair",
                "how": "protocol-repair",
            }.get(level, "communication-tree-repair")
        missing = str(llm_signal.get("missing_capability", "") or decision.get("missing_capability", "") or decision.get("reason", "") or "")
        anchor = self._public_tree_need_anchor_from_decision(decision, path, diagnosis)
        return {
            "level": level,
            "suggested_node_hint": hint,
            "why_current_nodes_insufficient": self._short_text(missing or "current public tree route did not express the needed communication behavior", 360),
            "evidence_pattern": self._short_text(
                llm_signal.get("failure_pattern", "")
                or llm_signal.get("observed_behavior", "")
                or decision.get("reason", "")
                or diagnosis.get("primary_failure_level", "")
                or hint,
                360,
            ),
            "expected_public_tree_behavior": self._short_text(llm_signal.get("expected_public_tree_behavior", ""), 360),
            "observed_behavior": self._short_text(llm_signal.get("observed_behavior", ""), 360),
            "evidence_for_batch": [self._short_text(x, 180) for x in list(llm_signal.get("evidence_for_batch", []) or [])[:4]],
            "support_strength": "single_user_medium" if decision.get("confidence") == "high" else "single_user_low",
            "failure_type": str(diagnosis.get("primary_failure_level", "") or hint),
            "failed_stage": {
                "what": "task_mapping",
                "who": "advisor_selection",
                "how": "advisor_interaction",
            }.get(level, "advisor_interaction"),
            "reference_node": reference,
            "source": str(decision.get("rule_source", "") or "communication_evolution_decision"),
            "public_tree_need_anchor": anchor,
        }

    def _route_order_for_level(self, route, path, level):
        route = dict(route or {})
        path = dict(path or {})
        level = str(level or "")
        if level == "what":
            scope = self._route_scope(path, "what")
            bucket = "what_by_why"
        elif level == "who":
            scope = self._route_scope(path, "who")
            bucket = "who_by_how"
        else:
            level = "how"
            scope = self._route_scope(path, "how")
            bucket = "how_by_what"
        order = [str(x) for x in list(dict(route.get(bucket, {}) or {}).get(scope, []) or []) if str(x).strip()]
        node = str(path.get(level, "") or "")
        return bucket, scope, node, order

    def _diagnosis_quality_issues(self, diagnosis):
        diagnosis = dict(diagnosis or {})
        issues = []
        sources = [
            diagnosis.get("communication_quality_issues", []),
            dict(diagnosis.get("interaction_trace", {}) or {}).get("communication_quality_issues", []),
            dict(diagnosis.get("tree_diagnosis", {}) or {}).get("communication_quality_issues", []),
        ]
        for source in sources:
            if isinstance(source, str):
                source = [source]
            for issue in list(source or []):
                issue = str(issue or "").strip()
                if issue and issue not in issues:
                    issues.append(issue)
        return issues

    def _extract_json_object(self, text):
        if isinstance(text, dict):
            return text
        raw = str(text or "").strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            pass
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidate = raw[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return {}
        return {}

    def _tree_node_contract(self, engine, level, node_id, path=None):
        level = str(level or "")
        node_id = str(node_id or "")
        path = dict(path or {})
        if not level or not node_id:
            return {}
        node = {}
        payload_node = dict(dict(path.get("path_skill_payload", {}) or {}).get(level, {}) or {})
        if str(payload_node.get("node_id", "") or "") == node_id:
            node.update(payload_node)
        try:
            tree = engine.public_tree_store.load_tree(force_reload=True)
            node.update(dict((tree.get(level, {}) or {}).get(node_id, {}) or {}))
        except Exception:
            pass
        out = {
            "level": level,
            "node_id": node_id,
            "status": str(node.get("status", "") or ""),
            "use_why": self._short_text(node.get("use_why", "") or node.get("description", ""), 260),
            "if_selected": self._short_text(node.get("if_selected", ""), 260),
        }
        selection_profile = dict(node.get("selection_profile", {}) or {})
        if selection_profile:
            out["selection_profile"] = {
                "requires": list(selection_profile.get("requires", []) or [])[:3],
                "prefers": list(selection_profile.get("prefers", []) or [])[:3],
                "do_not_use_why": list(selection_profile.get("do_not_use_why", []) or [])[:3],
            }
        fmt = list(node.get("task_output_format", []) or node.get("advisor_output_format", []) or [])
        if fmt:
            out["output_contract"] = [self._short_text(x, 160) for x in fmt[:4]]
        return self._json_safe(out)

    def _available_existing_route_contracts(self, engine, route, path):
        rows = []
        for level in ["what", "how", "who"]:
            _, scope, current_node, order = self._route_order_for_level(route, path, level)
            candidates = []
            if current_node:
                candidates.append(("current", current_node))
            for node in order:
                node = str(node or "").strip()
                if node and node != current_node and node not in ["none", "skip"]:
                    candidates.append(("sibling", node))
            for relation, node in candidates[:4]:
                contract = self._tree_node_contract(engine, level, node, path=path)
                if not contract:
                    continue
                contract["relation"] = relation
                contract["scope"] = scope
                rows.append(contract)
        return rows[:12]

    def _last_round_trace_context(self, trace_context):
        contexts = [ctx for ctx in list((trace_context or {}).get("round_trace_contexts", []) or []) if isinstance(ctx, dict)]
        return dict(contexts[-1]) if contexts else dict(trace_context or {})

    def _compact_advisor_summary_for_decision(self, trace_context):
        ctx = self._last_round_trace_context(trace_context)
        execution = dict(ctx.get("execution_packet", {}) or (trace_context or {}).get("execution_packet", {}) or {})
        committee = dict(execution.get("committee_packet", {}) or {})
        synthesis = dict(committee.get("advisor_synthesis_packet", {}) or {})
        interaction = dict(synthesis.get("interaction_summary", {}) or {})
        candidate_summaries = {}
        for key, value in list(dict(synthesis.get("candidate_summaries", {}) or {}).items())[:4]:
            candidate_summaries[str(key)] = self._short_text(value, 160)
        def compact_value(value, depth=0):
            if depth >= 2:
                return self._short_text(value, 140)
            if isinstance(value, dict):
                return {
                    str(k): compact_value(v, depth + 1)
                    for k, v in list(value.items())[:5]
                    if str(k).strip()
                }
            if isinstance(value, list):
                return [compact_value(v, depth + 1) for v in value[:5]]
            return self._short_text(value, 160)
        return self._json_safe(
            {
                "task": self._short_text(execution.get("task") or execution.get("user_task", ""), 240),
                "what_was_answered": self._short_text(synthesis.get("what_was_answered", ""), 220),
                "candidate_summaries": candidate_summaries,
                "task_specific_summary": compact_value(dict(synthesis.get("task_specific_summary", {}) or {})),
                "main_disagreements": [
                    self._short_text(x, 160)
                    for x in list(interaction.get("main_disagreements", []) or [])[:3]
                    if str(x).strip()
                ],
                "corrections_or_rebuttals": [
                    self._short_text(x, 160)
                    for x in list(interaction.get("corrections_or_rebuttals", []) or [])[:3]
                    if str(x).strip()
                ],
                "unresolved_conflicts": [
                    self._short_text(x, 160)
                    for x in list(interaction.get("unresolved_conflicts", []) or [])[:3]
                    if str(x).strip()
                ],
                "remaining_uncertainty": [
                    self._short_text(x, 160)
                    for x in list(synthesis.get("remaining_uncertainty", []) or [])[:4]
                    if str(x).strip()
                ],
            }
        )

    def _compact_user_redecision_for_decision(self, trace_context):
        ctx = self._last_round_trace_context(trace_context)
        packet = dict(ctx.get("redecision_packet", {}) or (trace_context or {}).get("redecision_packet", {}) or {})
        response = dict(ctx.get("user_response_after_round", {}) or {})
        arbitration = dict(packet.get("arbitration", {}) or response.get("arbitration", {}) or {})
        def as_list(value):
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, str):
                return [x.strip() for x in value.split(",") if x.strip()]
            return []
        return self._json_safe(
            {
                "selected_item": str(
                    packet.get("revised_name", "")
                    or packet.get("revised_item", "")
                    or packet.get("decision_item", "")
                    or packet.get("item", "")
                    or packet.get("selected_item", "")
                    or arbitration.get("decision_item", "")
                    or response.get("item", "")
                    or response.get("selected_item", "")
                    or ""
                ),
                "reason": self._short_text(
                    packet.get("revised_reason", "")
                    or packet.get("reason", "")
                    or response.get("reason", ""),
                    260,
                ),
                "decision_state": str(
                    arbitration.get("decision_state", "")
                    or packet.get("decision_state", "")
                    or packet.get("DecisionState", "")
                    or response.get("decision_state", "")
                    or ""
                ),
                "current_decision": str(
                    arbitration.get("current_decision", "")
                    or packet.get("current_decision", "")
                    or response.get("current_decision", "")
                    or ""
                ),
                "decision_confidence": str(
                    arbitration.get("decision_confidence", "")
                    or packet.get("decision_confidence", "")
                    or response.get("decision_confidence", "")
                    or ""
                ),
                "next_round_task": self._short_text(
                    arbitration.get("feedback_to_advisors", "")
                    or packet.get("feedback_to_advisors", "")
                    or packet.get("FeedbackToAdvisors", "")
                    or response.get("feedback_to_advisors", ""),
                    240,
                ),
                "next_round_hesitation_set": as_list(
                    arbitration.get("next_round_focus", [])
                    or arbitration.get("next_round_hesitation_set", [])
                    or packet.get("next_round_hesitation_set", [])
                    or packet.get("NextRoundHesitationSet", [])
                    or []
                )[:5],
                "removed_from_hesitation_set": as_list(
                    arbitration.get("removed_from_hesitation", [])
                    or arbitration.get("removed_from_hesitation_set", [])
                    or packet.get("removed_from_hesitation_set", [])
                    or packet.get("RemovedFromHesitationSet", [])
                    or []
                )[:5],
            }
        )

    def _target_evidence_signal_for_decision(self, diagnosis):
        tree_diag = dict((diagnosis or {}).get("tree_diagnosis", {}) or {})
        interaction = dict((diagnosis or {}).get("interaction_trace", {}) or {})
        def compact_value(value, depth=0):
            if depth >= 2:
                return self._short_text(value, 180)
            if isinstance(value, dict):
                return {
                    str(k): compact_value(v, depth + 1)
                    for k, v in list(value.items())[:8]
                    if str(k).strip()
                }
            if isinstance(value, list):
                return [compact_value(v, depth + 1) for v in value[:8]]
            return self._short_text(value, 220)
        for source in [diagnosis, tree_diag, interaction]:
            for key in ["target_evidence_signal", "target_summary_signal", "advisor_target_signal"]:
                value = dict((source or {}).get(key, {}) or {})
                if value:
                    return self._json_safe(compact_value(value))
        return {}

    def _communication_evolution_decision_payload(self, engine, full_user_policy, diagnosis, trace_context, path):
        _, route = self._ensure_route_skill(dict(full_user_policy or {}))
        path = dict(path or {})
        advisor_summary = self._compact_advisor_summary_for_decision(trace_context)
        user_redecision = self._compact_user_redecision_for_decision(trace_context)
        detected = {
            "primary_failure_level": str((diagnosis or {}).get("primary_failure_level", "") or ""),
            "failure_attribution": str((diagnosis or {}).get("failure_attribution", "") or ""),
            "communication_quality_issues": self._diagnosis_quality_issues(diagnosis)[:6],
            "target_evidence_signal": self._target_evidence_signal_for_decision(diagnosis),
        }
        return self._json_safe(
            {
                "route_used": {
                    'why': str(path.get('why', "") or ""),
                    "what": str(path.get("what", "") or ""),
                    "who": str(path.get("who", "") or ""),
                    "how": str(path.get("how", "") or ""),
                    "user_task": self._short_text(path.get("user_task", ""), 420),
                },
                "available_existing_routes": self._available_existing_route_contracts(engine, route, path),
                "advisor_summary": advisor_summary,
                "user_redecision": user_redecision,
                "detected_signals": detected,
            }
        )

    def _route_decision_selected_node_valid(self, decision, full_user_policy, path):
        _, route = self._ensure_route_skill(dict(full_user_policy or {}))
        level = str((decision or {}).get("level", "") or "")
        node = str((decision or {}).get("selected_existing_node", "") or "")
        if level not in {"what", "who", "how"} or not node:
            return False
        _, _, current, order = self._route_order_for_level(route, path, level)
        return bool(node in order and node != current and node not in ["none", "skip"])

    def _llm_decide_communication_evolution_target(self, engine, full_user_policy, diagnosis, trace_context, path):
        payload = self._communication_evolution_decision_payload(engine, full_user_policy, diagnosis, trace_context, path)
        system_prompt = (
            "You judge one failed communication training case. Choose exactly one: "
            "user_absorption_update, existing_route_reorder, or public_tree_need. "
            "Use only the compact evidence; do not generate nodes or rewrite skills. "
            "user_absorption_update requires clear actionable advisor guidance covering the relevant HesitationSet and a user choice against it. "
            "If advisor evidence is unresolved, incomplete, ask-user, or lacks executable comparison, choose public_tree_need. "
            "existing_route_reorder requires a clearly better available sibling route and selected_existing_node. "
            "Choose public_tree_need when current routes lack the needed behavior/protocol/task/advisor capability, or when uncertain. "
            "Return strict JSON only."
        )
        user_prompt = (
            "CommunicationEvolutionDecisionInput:\n"
            f"{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "decision": "user_absorption_update|existing_route_reorder|public_tree_need",\n'
            '  "confidence": "high|medium|low",\n'
            '  "level": "what|who|how|none",\n'
            '  "reason": "",\n'
            '  "not_user_absorption_reason": "",\n'
            '  "user_absorption_rule": "Only for user_absorption_update: one concrete reusable rule for how this user should use clear advisor evidence next time; otherwise empty.",\n'
            '  "selected_existing_node": "",\n'
            '  "route_reorder_hint": "",\n'
            '  "tree_need_signal": {\n'
            '    "missing_capability": "",\n'
            '    "failure_pattern": "",\n'
            '    "expected_public_tree_behavior": "",\n'
            '    "observed_behavior": "",\n'
            '    "reference_node": "",\n'
            '    "suggested_node_hint": "",\n'
            '    "evidence_for_batch": []\n'
            "  }\n"
            "}\n"
        )
        raw_response = None
        parsed = {}
        llm_usage = {}
        try:
            if not hasattr(engine.args, "max_retry_num"):
                setattr(engine.args, "max_retry_num", 3)
            if not hasattr(engine.args, "temperature"):
                setattr(engine.args, "temperature", 0.2)
            update_llm_prompt_trace_context(
                phase="communication_evolution_decision",
                advisor_index="",
                advisor_id="",
                advisor_type="",
                path_why=str((path or {}).get('why', "") or ""),
                path_who=str((path or {}).get("who", "") or ""),
                path_how=str((path or {}).get("how", "") or ""),
            )
            raw_response = llm_request(system_prompt, user_prompt, engine.args)
            llm_usage = get_last_llm_request_usage()
            parsed = self._extract_json_object(raw_response)
        except Exception as exc:
            parsed = {"decision": "public_tree_need", "confidence": "low", "reason": f"llm_decision_failed: {exc}"}
        if not isinstance(parsed, dict) or not parsed:
            parsed = {"decision": "public_tree_need", "confidence": "low", "reason": "llm_decision_returned_no_json"}
        decision = self._normalize_evolution_decision(parsed)
        decision["rule_source"] = "llm_communication_evolution_decision"
        if decision.get("decision") == "existing_route_reorder" and not self._route_decision_selected_node_valid(decision, full_user_policy, path):
            original = dict(decision)
            decision = self._normalize_evolution_decision(
                {
                    "decision": "public_tree_need",
                    "confidence": "low",
                    "level": original.get("level") if original.get("level") in {"what", "who", "how"} else "how",
                    "reason": "LLM chose route reorder but did not name a valid existing sibling route; treat as public tree need.",
                    "reference_node": original.get("reference_node") or self._reference_node_for_layer(original.get("level", "how"), path),
                    "suggested_node_hint": "route-decision-repair",
                    "missing_capability": original.get("reason", ""),
                    "tree_need_signal": original.get("tree_need_signal", {}),
                    "rule_source": "llm_route_reorder_invalid_as_tree_need",
                }
            )
        append_jsonl(
            engine.public_tree_store.index_dir / "communication_evolution_decision_prompt_io.jsonl",
            {
                "event": "communication_evolution_decision_prompt_io",
                "ts": int(time.time()),
                "user_id": str((trace_context or {}).get("user_id", "") or ""),
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "llm_usage": llm_usage,
                "payload": payload,
                "raw_response_preview": self._short_text(raw_response, 1800),
                "parsed_json": parsed,
                "decision": decision,
            },
        )
        return decision

    def _derive_existing_route_reorder_operations(self, route, path, diagnosis):
        decision = self._normalize_evolution_decision((diagnosis or {}).get("evolution_decision", {}))
        if decision.get("decision") != "existing_route_reorder":
            return []
        preferred_levels = []
        level = str(decision.get("level", "") or "")
        if level in ["what", "who", "how"]:
            preferred_levels.append(level)
        preferred_levels.extend([x for x in ["how", "who", "what"] if x not in preferred_levels])
        operations = []
        for one_level in preferred_levels:
            _, scope, node, order = self._route_order_for_level(route, path, one_level)
            alternatives = [x for x in order if x and x != node and x not in ["none", "skip"]]
            if not node or not alternatives:
                continue
            selected = str(decision.get("selected_existing_node", "") or "")
            if selected:
                if selected not in alternatives:
                    continue
                promote_node = selected
            else:
                continue
            operations.append(
                {
                    "operation": "demote",
                    "level": one_level,
                    "scope": scope,
                    "node": node,
                    "reason": decision.get("reason") or f"failed communication route should rank behind {promote_node}",
                }
            )
            operations.append(
                {
                    "operation": "promote",
                    "level": one_level,
                    "scope": scope,
                    "node": promote_node,
                    "before": node,
                    "reason": decision.get("reason") or f"existing public-tree {one_level} alternative after failed communication route",
                }
            )
            break
        return operations

    @staticmethod
    def _short_text(value, limit=360):
        text = " ".join(str(value or "").split())
        if len(text) <= int(limit):
            return text
        return text[: max(0, int(limit) - 3)].rstrip() + "..."

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): TrainOnlyEvolver._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrainOnlyEvolver._json_safe(v) for v in value]
        if isinstance(value, set):
            return [TrainOnlyEvolver._json_safe(v) for v in sorted(value, key=lambda x: str(x))]
        if isinstance(value, Counter):
            return {str(k): int(v) for k, v in value.items()}
        return value

    @staticmethod
    def _as_dict(value):
        return dict(value or {}) if isinstance(value, dict) else {}

    def _safe_path_dict(self, value, fallback=None):
        if isinstance(value, dict):
            return {
                'why': str(value.get('why', "") or ""),
                "what": str(value.get("what", "") or ""),
                "who": str(value.get("who", "") or ""),
                "how": str(value.get("how", "") or ""),
            }
        fallback = fallback if isinstance(fallback, dict) else {}
        return {
            'why': str(fallback.get('why', "") or ""),
            "what": str(fallback.get("what", "") or ""),
            "who": str(fallback.get("who", "") or ""),
            "how": str(fallback.get("how", "") or ""),
        }

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    def _policy_field_for_level(self, level):
        level = str(level or "").strip().lower()
        return {
            'why': ("trigger_policy", 'why'),
            "trigger": ("trigger_policy", 'why'),
            "what": ("what_policy", "preferred_what"),
            "who": ("who_policy", "preferred_who"),
            "how": ("how_policy", "preferred_how"),
        }.get(level, ("open_condition_memory", "suggested_node_hint"))

    def _communication_confidence_value(self, value, default=0.38):
        if isinstance(value, str):
            return {"high": 0.72, "medium": 0.55, "low": 0.35}.get(value.strip().lower(), default)
        try:
            return max(0.05, min(0.95, float(value)))
        except Exception:
            return default

    def _merge_planning_policy_row(self, rows, incoming, id_fields, confidence_delta=0.0, limit=20):
        out = [dict(row) for row in list(rows or []) if isinstance(row, dict)]
        incoming = dict(incoming or {})
        if not incoming:
            return out[:limit]
        key = "|".join(str(incoming.get(field, "") or "").strip().lower() for field in id_fields)
        if not key.strip("|"):
            out.append(incoming)
            return out[-limit:]
        for idx, row in enumerate(out):
            row_key = "|".join(str(row.get(field, "") or "").strip().lower() for field in id_fields)
            if row_key != key:
                continue
            cur = self._communication_confidence_value(row.get("confidence", 0.40), 0.40)
            inc = self._communication_confidence_value(incoming.get("confidence", cur), cur)
            row.update({k: v for k, v in incoming.items() if v not in ["", None, [], {}]})
            row["confidence"] = round(max(0.05, min(0.95, max(cur, inc) + confidence_delta)), 3)
            row["reinforce_count"] = int(row.get("reinforce_count", 0) or 0) + (1 if confidence_delta > 0 else 0)
            row["weaken_count"] = int(row.get("weaken_count", 0) or 0) + (1 if confidence_delta < 0 else 0)
            out[idx] = row
            return out[:limit]
        incoming["confidence"] = round(max(0.05, min(0.95, self._communication_confidence_value(incoming.get("confidence", 0.35)) + confidence_delta)), 3)
        out.append(incoming)
        return out[-limit:]

    def _condition_signature_from_trace(self, trace_context, path=None):
        trace_context = dict(trace_context or {})
        path = dict(path or trace_context.get("path") or {})
        planner_log = dict(path.get("planner_log", {}) or {})
        signature = dict(planner_log.get("state_signature", {}) or {})
        if signature:
            return signature
        decision_state = dict(trace_context.get("decision_state") or {})
        confidence = int(decision_state.get("self_confidence", 0) or 0)
        focus_size = len(list(decision_state.get("candidate_shortlist", []) or decision_state.get("shortlist", []) or []))
        primary = str(decision_state.get("primary_trigger", "") or path.get('why', "") or "")
        uncertainty = [str(x) for x in list(decision_state.get("uncertainty_points", []) or []) if str(x).strip()]
        if "candidate_comparison" in uncertainty or focus_size >= 2:
            shape = "candidate-conflict"
        elif "internal_prior_conflict" in uncertainty or primary == "internal-prior-conflict":
            shape = "internal-prior-conflict"
        elif "novelty_justification" in uncertainty or primary == "novelty-uncertainty":
            shape = "novelty-check"
        else:
            shape = primary or "proposal-risk-check"
        return {
            'why': primary,
            "round_type": str(decision_state.get("round_type", "") or "initial"),
            "uncertainty_shape": shape,
            "confidence_band": "high" if confidence >= 75 else ("medium" if confidence >= 50 else "low"),
            "focus_set_size": int(focus_size),
            "previous_feedback_exists": bool(decision_state.get("previous_user_feedback")),
        }

    def _communication_execution_quality(self, trace_context):
        trace_context = dict(trace_context or {})
        execution = dict(trace_context.get("execution_packet") or {})
        committee = dict(execution.get("committee_packet", {}) or {})
        evidence_summary = dict(committee.get("evidence_summary", {}) or {})
        advisor_feedbacks = [row for row in list(execution.get("advisor_feedbacks", []) or []) if isinstance(row, dict)]
        try:
            quality_issues = list(self.analyzer._communication_quality_issues(trace_context) or [])
        except Exception:
            quality_issues = []
        issue_text = " ".join(str(x) for x in quality_issues).lower()
        silent = list(evidence_summary.get("silent_focus_candidates", []) or [])
        missing = list(evidence_summary.get("missing_advisor_evidence", []) or [])
        candidate_rows = list(evidence_summary.get("candidate_evidence", []) or evidence_summary.get("candidate_evidence_rows", []) or [])
        candidate_coverage = "full"
        if silent or missing:
            candidate_coverage = "partial"
        if not advisor_feedbacks:
            candidate_coverage = "none"
        evidence_specificity = "concrete" if candidate_rows else ("generic" if advisor_feedbacks else "missing")
        if any(token in issue_text for token in ["generic", "no_candidate", "evidence_lost"]):
            evidence_specificity = "generic"
        advisor_signals = set()
        for row in advisor_feedbacks:
            advisor_signals.add(
                "|".join(
                    [
                        str(row.get("advice", "") or row.get("stance", "") or ""),
                        str(row.get("suggested_item", "") or row.get("endorsed_item", "") or row.get("attacked_item", "") or ""),
                    ]
                )
            )
        advisor_diversity = "unknown"
        if len(advisor_feedbacks) >= 2:
            advisor_diversity = "low" if len(advisor_signals) <= 1 else ("medium" if len(advisor_signals) <= 3 else "high")
        return {
            "protocol_enforced": not any(token in issue_text for token in ["protocol", "no_candidate_set_signal"]),
            "candidate_coverage": candidate_coverage,
            "evidence_specificity": evidence_specificity,
            "advisor_diversity": advisor_diversity,
            "advisor_count": int(len(advisor_feedbacks)),
            "quality_issues": [str(x) for x in quality_issues[:10]],
        }

    def _advisor_turn_reflection_summary(self, feedback, max_excerpt=360):
        feedback = dict(feedback or {})
        raw = str(feedback.get("raw_text", "") or "")
        excerpt_parts = [
            str(feedback.get("task_answer", "") or ""),
            str(feedback.get("ask_user", "") or ""),
            str(feedback.get("response_to_previous", "") or ""),
            str(feedback.get("challenge_or_support_previous", "") or ""),
        ]
        excerpt = " | ".join([x for x in excerpt_parts if str(x).strip()])
        if not excerpt:
            excerpt = raw
        return {
            "advisor_type": str(feedback.get("advisor_type", "") or ""),
            "candidate_view": [
                {
                    "candidate": str(row.get("candidate", "") or ""),
                    "view": str(row.get("view", "") or ""),
                    "reason": self._short_text(row.get("reason", ""), 180),
                }
                for row in list(feedback.get("candidate_views", []) or [])[:8]
                if isinstance(row, dict)
            ],
            "task_answer": self._short_text(feedback.get("task_answer", ""), 420),
            "ask_user": self._short_text(feedback.get("ask_user", ""), 300),
            "response_to_previous": self._short_text(feedback.get("response_to_previous", ""), 300),
            "challenge_or_support_previous": self._short_text(feedback.get("challenge_or_support_previous", ""), 300),
            "protocol_issues": [str(x) for x in list(feedback.get("protocol_issues", []) or [])[:8]],
            "raw_excerpt": self._short_text(excerpt, max_excerpt),
        }

    def _advisor_raw_speech(self, feedback, limit=1200):
        raw = str((feedback or {}).get("raw_text", "") or "")
        if raw.startswith("speech="):
            speech = raw[len("speech="):]
            marker = speech.find("; pos=")
            if marker >= 0:
                speech = speech[:marker]
            return self._short_text(speech, limit)
        return self._short_text(raw, limit)

    def _advisor_dialogue_log(self, advisor_feedbacks, max_advisors=6):
        rows = []
        for idx, feedback in enumerate(list(advisor_feedbacks or [])[:max_advisors], start=1):
            if not isinstance(feedback, dict):
                continue
            candidate_views = []
            for row in self._candidate_view_rows(feedback)[:8]:
                candidate_views.append(
                    {
                        "candidate": str(row.get("candidate", "") or ""),
                        "view": str(row.get("view", "") or row.get("label", "") or row.get("status", "") or ""),
                        "reason": self._short_text(row.get("reason", "") or row.get("evidence", "") or row.get("summary", ""), 260),
                    }
                )
            rows.append(
                {
                    "advisor_index": idx,
                    "advisor_id": str(feedback.get("advisor_id", "") or ""),
                    "advisor_type": str(feedback.get("advisor_type", "") or feedback.get("advisor_role", "") or ""),
                    "advisor_speech": self._advisor_raw_speech(feedback, 1200),
                    "task_answer": self._short_text(feedback.get("task_answer", ""), 520),
                    "candidate_views": candidate_views,
                    "response_to_previous": self._short_text(feedback.get("response_to_previous", ""), 420),
                    "challenge_or_support_previous": self._short_text(feedback.get("challenge_or_support_previous", ""), 420),
                    "correction": self._short_text(feedback.get("correction", ""), 360),
                    "ask_user": self._short_text(feedback.get("ask_user", ""), 360),
                    "protocol_issues": [str(x) for x in list(feedback.get("protocol_issues", []) or [])[:8]],
                    "raw_parse_record": self._short_text(feedback.get("raw_text", ""), 1500),
                }
            )
        return self._json_safe(rows)

    def _summary_agent_log(self, packet):
        packet = dict(packet or {})
        synthesis = dict(packet.get("advisor_synthesis_packet", {}) or {})
        interaction = dict(synthesis.get("interaction_summary", {}) or {})
        summary_input = dict(packet.get("summary_agent_input", {}) or {})
        input_feedbacks = []
        for idx, feedback in enumerate(list(summary_input.get("advisor_feedbacks", []) or [])[:6], start=1):
            if not isinstance(feedback, dict):
                continue
            input_feedbacks.append(
                {
                    "advisor_index": idx,
                    "advisor_id": str(feedback.get("advisor_id", "") or ""),
                    "advisor_type": str(feedback.get("advisor_type", "") or ""),
                    "task_answer": self._short_text(feedback.get("task_answer", ""), 360),
                    "response_to_previous": self._short_text(feedback.get("response_to_previous", ""), 240),
                    "challenge_or_support_previous": self._short_text(feedback.get("challenge_or_support_previous", ""), 240),
                    "raw_text_excerpt": self._short_text(feedback.get("raw_text_excerpt", "") or feedback.get("raw_text", ""), 520),
                    "protocol_issues": [str(x) for x in list(feedback.get("protocol_issues", []) or [])[:6]],
                }
            )
        return self._json_safe(
            {
                "aggregation_mode": str(packet.get("aggregation_mode", "") or "summary_agent_v1"),
                "decision_policy": str(packet.get("decision_policy", "") or synthesis.get("decision_policy", "") or "information_only_no_vote"),
                "summary_agent_parse_status": str(packet.get("summary_agent_parse_status", "") or ""),
                "legacy_aggregation_used": bool(packet.get("legacy_aggregation_used", False)),
                "summary_agent_raw_response": self._short_text(packet.get("summary_agent_raw_response", ""), 2200),
                "advisor_synthesis_packet": {
                    "source": str(synthesis.get("source", "") or ""),
                    "what_was_answered": self._short_text(synthesis.get("what_was_answered", ""), 520),
                    "candidate_summaries": dict(synthesis.get("candidate_summaries", {}) or {}),
                    "task_specific_summary": dict(synthesis.get("task_specific_summary", {}) or {}),
                    "extra_task_summary": dict(synthesis.get("extra_task_summary", {}) or {}),
                    "interaction_summary": {
                        "main_disagreements": list(interaction.get("main_disagreements", []) or [])[:6],
                        "corrections_or_rebuttals": list(interaction.get("corrections_or_rebuttals", []) or [])[:6],
                        "unresolved_conflicts": list(interaction.get("unresolved_conflicts", []) or [])[:6],
                    },
                    "extra_interaction_summary": dict(synthesis.get("extra_interaction_summary", {}) or {}),
                    "remaining_uncertainty": list(synthesis.get("remaining_uncertainty", []) or [])[:8],
                },
                "summary_agent_input_brief": {
                    "task": self._short_text(summary_input.get("task", "") or summary_input.get("user_task", ""), 420),
                    "proposal_item": str(summary_input.get("proposal_item", "") or ""),
                    "focus_candidates": list(summary_input.get("focus_candidates", []) or [])[:8],
                    "advisor_feedbacks": input_feedbacks,
                    "diagnostic_context": dict(summary_input.get("diagnostic_context", {}) or {}),
                },
            }
        )

    def _communication_round_dialogue_log(self, trace_context, diagnosis=None, path=None, outcome_signal=""):
        trace_context = dict(trace_context or {})
        diagnosis = dict(diagnosis or {})
        contexts = [ctx for ctx in list(trace_context.get("round_trace_contexts", []) or []) if isinstance(ctx, dict)]
        if not contexts and trace_context.get("execution_packet"):
            contexts = [trace_context]
        rounds = []
        for ctx in contexts[:8]:
            ctx = dict(ctx or {})
            ctx_path = dict(ctx.get("path", {}) or path or trace_context.get("path", {}) or {})
            execution = dict(ctx.get("execution_packet", {}) or {})
            committee = dict(execution.get("committee_packet", {}) or {})
            user_after_round = dict(ctx.get("user_response_after_round", {}) or {})
            rounds.append(
                {
                    "round_id": ctx.get("round_id"),
                    "path": {
                        'why': str(ctx_path.get('why', "") or ""),
                        "what": str(ctx_path.get("what", "") or ""),
                        "who": str(ctx_path.get("who", "") or ""),
                        "how": str(ctx_path.get("how", "") or ""),
                        "user_task": self._short_text(ctx_path.get("user_task", "") or execution.get("task", "") or execution.get("user_task", ""), 420),
                    },
                    "task": self._short_text(execution.get("task") or execution.get("user_task", "") or ctx_path.get("user_task", ""), 520),
                    "previous_user_feedback": self._short_text(execution.get("previous_user_feedback", ""), 420),
                    "advisor_dialogue_log": self._advisor_dialogue_log(execution.get("advisor_feedbacks", []) or []),
                    "summary_agent_log": self._summary_agent_log(committee),
                    "user_after_round": self._json_safe(user_after_round),
                }
            )
        return self._json_safe(
            {
                "event": "communication_round_dialogue",
                "ts": int(time.time()),
                "user_id": str(trace_context.get("user_id", "") or ""),
                "outcome_signal": str(outcome_signal or diagnosis.get("outcome_signal", "") or ""),
                "primary_failure_level": str(diagnosis.get("primary_failure_level", "") or ""),
                "failure_attribution": str(diagnosis.get("failure_attribution", "") or ""),
                "evolution_decision": dict(diagnosis.get("evolution_decision", {}) or {}),
                "route_diff_update": dict(diagnosis.get("route_diff_update", {}) or {}),
                "rounds": rounds,
            }
        )

    def _advisor_speech_summary_for_batch(self, feedback):
        feedback = dict(feedback or {})
        candidate_views = []
        for row in self._candidate_view_rows(feedback)[:3]:
            candidate_views.append(
                {
                    "view": self._short_text(row.get("view", "") or row.get("label", "") or row.get("status", ""), 80),
                    "reason": self._short_text(row.get("reason", "") or row.get("evidence", "") or row.get("summary", ""), 140),
                }
            )
        issues = [str(x) for x in list(feedback.get("protocol_issues", []) or []) if str(x).strip()]
        return self._json_safe(
            {
                "advisor_type": str(feedback.get("advisor_type", "") or feedback.get("advisor_role", "") or ""),
                "answer_summary": self._short_text(
                    feedback.get("task_answer", "") or feedback.get("raw_text", "") or self._feedback_text(feedback),
                    220,
                ),
                "response_or_challenge": self._short_text(
                    feedback.get("challenge_or_support_previous", "") or feedback.get("response_to_previous", ""),
                    180,
                ),
                "candidate_evidence": candidate_views,
                "protocol_issues": issues[:3],
            }
        )

    @staticmethod
    def _normalize_rule_text(text):
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        lines = [" ".join(line.split()) for line in raw.split("\n")]
        return "\n".join([line for line in lines if line])

    @staticmethod
    def _is_reasoning_strategy_rule(text):
        low = " ".join(str(text or "").strip().lower().split())
        if not low:
            return False
        strategy_markers = [
            "### evolvable strategy",
            "ranking criteria",
            "primary:",
            "secondary:",
            "tie breaker:",
            "hesitation set:",
            "choose_by=",
            "preserve=",
            "minority=",
            "avoid_overbias=",
            "when ",
        ]
        preference_only_markers = [
            "reinforce transferable item-selection signal",
            "add transferable item-selection signal",
            "weaken over-bias",
        ]
        if any(marker in low for marker in preference_only_markers):
            return False
        return any(marker in low for marker in strategy_markers)

    def _sanitize_user_update(self, row, fallback=None):
        row = dict(row or {})
        fallback = dict(fallback or {})
        layer = UserPolicyStore.canonical_skill_layer(row.get("target_layer") or fallback.get("target_layer") or "item_selection_skill")
        operation = str(row.get("operation") or fallback.get("operation") or "discover")
        if operation not in ["reinforce", "discover", "weaken", "record_only"]:
            operation = "discover"
        rule = self._normalize_rule_text(row.get("rule") or fallback.get("rule") or "")
        problem = " ".join(str(row.get("problem") or fallback.get("problem") or "").split())
        confidence = self._bounded_confidence(row.get("confidence", fallback.get("confidence", 0.42)))
        return {
            "target_layer": layer,
            "operation": operation,
            "problem": problem,
            "rule": rule,
            "confidence": confidence,
        }

    def _generalize_item_specific_update(self, update, forbidden_item_names=None):
        update = dict(update or {})
        forbidden_item_names = [str(x).strip() for x in (forbidden_item_names or []) if str(x or "").strip()]
        if not forbidden_item_names:
            return update
        rule = str(update.get("rule", "") or "")
        problem = str(update.get("problem", "") or "")

        for name in sorted(set(forbidden_item_names), key=len, reverse=True):
            if not name:
                continue
            pattern = re.compile(re.escape(name), flags=re.IGNORECASE)
            rule = pattern.sub("a candidate with the same user-relevant style/cluster signal", rule)
            problem = pattern.sub("the missed style/cluster signal", problem)

        item_specific_markers = [
            "this item",
            "that item",
            "the target item",
            "the correct item",
            "candidate like",
        ]
        low_rule = rule.lower()
        if any(marker in low_rule for marker in item_specific_markers):
            rule = (
                "choose_by=history-grounded domain preference and cluster match; "
                "preserve=target-like style/category/use-case/feature/cultural-signal candidates; "
                "minority=weak, recent, co-occurrence, language/region, niche category, or stable minority clusters; "
                "avoid_overbias=do not exclude those signals only because the current favorite matches a dominant cluster."
            )

        update["rule"] = self._normalize_rule_text(rule)
        update["problem"] = " ".join(problem.split())
        return update

    @staticmethod
    def _contains_forbidden_item_name(text, forbidden_item_names=None):
        low = str(text or "").lower()
        for name in forbidden_item_names or []:
            name = str(name or "").strip()
            if len(name) >= 2 and name.lower() in low:
                return True
        return False

    @staticmethod
    def _is_weak_meta_preference(text):
        low = " ".join(str(text or "").strip().lower().split())
        if not low:
            return False
        concrete_style_markers = [
            "rock", "metal", "pop", "punk", "emo", "hardcore", "indie", "folk", "jazz", "soul",
            "r&b", "hip-hop", "hip hop", "rap", "electronic", "ambient", "industrial", "gothic",
            "darkwave", "classical", "soundtrack", "latin", "j-pop", "jpop", "k-pop", "kpop",
            "vocal", "spoken-word", "spoken word", "instrumental", "acoustic", "dance", "trance",
            "product", "category", "brand", "manufacturer", "use case", "feature", "function",
            "price", "value", "quality", "durability", "reliability", "design", "form factor",
            "compatibility", "accessory", "review", "rating", "household", "consumer",
            "electronics", "home", "kitchen", "beauty", "book", "toy", "software", "hardware",
            "books", "author", "authors", "literary", "fiction", "nonfiction", "non-fiction",
            "memoir", "biography", "history", "novel", "fantasy", "mystery", "romance",
            "thriller", "poetry", "essay", "translated", "classic", "series", "young adult",
            "children", "academic",
        ]
        if any(re.search(r"(?<![a-z0-9])" + re.escape(marker) + r"(?![a-z0-9])", low) for marker in concrete_style_markers):
            return False
        weak_markers = [
            "mid-popularity", "low-popularity", "high-popularity", "popularity",
            "mainstream artist affinity", "popular/mainstream artist affinity", "mainstream affinity",
            "mainstream product affinity", "popular/mainstream product affinity",
            "mainstream book affinity", "popular/mainstream book affinity",
            "long-form band", "project-name", "name-shape", "naming marker", "name marker",
            "model-number", "technical-spec", "product-name signal", "product affinity",
            "book-title signal", "author-name signal", "book affinity",
            "regional-language", "regional script", "long-term anchor", "repeated listening anchor",
            "long-term listening anchor", "long-term listening anchors", "history anchor",
            "recent listening drift anchor", "recent listening drift anchors",
            "recent product-need drift anchor", "recent product-need drift anchors",
            "recent reading drift anchor", "recent reading drift anchors",
            "prior hint", "prior alignment", "preference alignment",
        ]
        return any(marker in low for marker in weak_markers)

    def _sanitize_incremental_update(self, incremental_update, forbidden_item_names=None):
        inc = dict(incremental_update or {})
        forbidden_item_names = [str(x).strip() for x in (forbidden_item_names or []) if str(x or "").strip()]
        cleaned = {}
        for bucket in ["reinforced", "new_preferences", "recent_signals", "weakened"]:
            rows = []
            for row in list(inc.get(bucket, []) or []):
                if not isinstance(row, dict):
                    continue
                row = dict(row)
                attribute = self._clean_preference_attribute(
                    row.get("attribute") or row.get("preference") or row.get("signal") or row.get("rule") or ""
                )
                if not attribute:
                    continue
                if len(attribute) > 60:
                    continue
                if UserPolicyStore._is_generic_item_protocol_rule(attribute):
                    continue
                if self._contains_forbidden_item_name(attribute, forbidden_item_names):
                    continue
                if bucket != "weakened" and self._is_weak_meta_preference(attribute):
                    continue
                row["attribute"] = attribute
                for text_key in ["evidence", "reason", "rationale"]:
                    if text_key not in row:
                        continue
                    text = str(row.get(text_key, "") or "")
                    for name in sorted(set(forbidden_item_names), key=len, reverse=True):
                        if name:
                            text = re.sub(re.escape(name), "a candidate with this transferable style signal", text, flags=re.I)
                    row[text_key] = " ".join(text.split())
                rows.append(row)
            cleaned[bucket] = rows
        return cleaned

    @staticmethod
    def _weaken_reason_causally_excludes_target(row):
        text = " ".join(
            str(row.get(key, "") or "")
            for key in ["reason", "evidence", "rationale"]
        ).lower()
        if not text:
            return False
        weak_noncausal_markers = [
            "not actively engaged",
            "not engaged",
            "was not used",
            "not used",
            "lower priority compared to",
            "may be a lower priority",
            "did not appear",
        ]
        if any(marker in text for marker in weak_noncausal_markers):
            return False
        causal_markers = [
            "caused",
            "led to",
            "excluded",
            "exclude",
            "omitted",
            "omit",
            "overshadowed",
            "crowded out",
            "narrowed too early",
            "over-reliance",
            "overreliance",
            "over-prioritized",
            "overprioritized",
            "dominant cluster",
            "filled the hesitation",
            "prevented",
        ]
        return any(marker in text for marker in causal_markers)

    def _gate_weakened_incremental_updates(self, incremental_update, target_in_shortlist, target_is_proposal):
        inc = dict(incremental_update or {})
        weakened = list(inc.get("weakened", []) or [])
        if target_in_shortlist or target_is_proposal:
            inc["weakened"] = []
            return inc, len(weakened)
        kept = [row for row in weakened if self._weaken_reason_causally_excludes_target(dict(row or {}))]
        inc["weakened"] = kept
        return inc, max(0, len(weakened) - len(kept))

    def _diff_row_to_user_update(self, row, operation, analysis="", default_confidence=0.42):
        row = dict(row or {})
        attribute = " ".join(str(row.get("attribute") or row.get("preference") or row.get("signal") or "").split())
        evidence = " ".join(str(row.get("evidence") or row.get("reason") or row.get("rationale") or "").split())
        if not attribute:
            return {}
        if operation == "reinforce":
            rule = (
                f"Reinforce transferable item-selection signal: {attribute}. "
                f"Evidence: {evidence or analysis or 'the proposal/target comparison supports this signal'}. "
                "Use this as a positive clue when future candidates express the same style or cluster."
            )
            problem = "The observed proposal/shortlist behavior confirms a user preference signal that should remain available during item selection."
            confidence = row.get("confidence", default_confidence)
        elif operation == "weaken":
            rule = (
                f"Weaken over-bias: {attribute}. "
                f"Reason: {evidence or analysis or 'this bias made the proposal salient while hiding a supervised target-style candidate'}. "
                "Do not let this bias exclude candidates with supported weak, minority, recent, or co-occurrence-neighbor signals."
            )
            problem = "The user's proposal reasoning over-weighted one signal and omitted a supervised target-style candidate from the choice space."
            confidence = row.get("confidence", default_confidence)
        else:
            rule = (
                f"Add transferable item-selection signal: {attribute}. "
                f"Evidence: {evidence or analysis or 'the omitted target carries this user-relevant signal'}. "
                "Preserve future candidates with this signal when it connects to the user's history, weak signals, minority clusters, recent drift, or co-occurrence neighbors."
            )
            problem = "The proposal-first reasoning missed a transferable positive signal that should be considered in future candidate selection."
            confidence = row.get("confidence", default_confidence)
        return self._sanitize_user_update(
            {
                "target_layer": "item_selection_skill",
                "operation": operation,
                "problem": problem,
                "rule": rule,
                "confidence": confidence,
            }
        )

    def _incremental_diff_to_updates(self, incremental_update, analysis=""):
        inc = dict(incremental_update or {})
        updates = []
        for row in list(inc.get("reinforced", []) or []):
            update = self._diff_row_to_user_update(row, "reinforce", analysis=analysis, default_confidence=0.55)
            if update.get("rule"):
                updates.append(update)
        for key in ["new_preferences", "recent_signals"]:
            for row in list(inc.get(key, []) or []):
                update = self._diff_row_to_user_update(row, "discover", analysis=analysis, default_confidence=0.42)
                if update.get("rule"):
                    updates.append(update)
        for row in list(inc.get("weakened", []) or []):
            update = self._diff_row_to_user_update(row, "weaken", analysis=analysis, default_confidence=0.40)
            if update.get("rule"):
                updates.append(update)
        return updates

    def _llm_refine_user_diagnosis(self, engine, full_user_policy, diagnosis, trace_context):
        if not bool(getattr(engine.args, "com_llm_evolve_user_skill", True)):
            return diagnosis
        if not bool(getattr(engine.args, "com_llm_evolve_item_skill", True)):
            diagnosis = dict(diagnosis or {})
            diagnosis["llm_reflection_used"] = False
            diagnosis["llm_reflection_skipped_reason"] = "com_llm_evolve_item_skill_false"
            return diagnosis
        trace_context = dict(trace_context or {})
        deterministic = dict((diagnosis or {}).get("user_skill_diagnosis", {}) or {})
        decision_state = dict(trace_context.get("decision_state") or {})
        candidate_evidence = list(decision_state.get("candidate_evidence", []) or [])
        target_names = [str(x) for x in trace_context.get("target_item_names", []) or []]
        target_norm = {self._norm(x) for x in target_names if self._norm(x)}
        target_evidence = [
            row for row in candidate_evidence
            if self._norm((row or {}).get("candidate", "")) in target_norm
        ]
        proposal_name = str(trace_context.get("proposal_item_name", "") or "")
        proposal_norm = self._norm(proposal_name)
        proposal_evidence = [
            row for row in candidate_evidence
            if proposal_norm and self._norm((row or {}).get("candidate", "")) == proposal_norm
        ]
        target_style_evidence = []
        for row in target_evidence:
            fit = " ".join(str((row or {}).get("fit", "") or "").split())
            reason = " ".join(str((row or {}).get("reason", "") or "").split())
            if fit:
                target_style_evidence.append({"type": "fit", "text": fit})
            if reason:
                target_style_evidence.append({"type": "reason", "text": reason})
        shortlist = list(decision_state.get("shortlist", []) or [])
        shortlist_norm = {self._norm(x) for x in shortlist if self._norm(x)}
        target_in_shortlist = any(self._norm(x) in shortlist_norm for x in target_names)
        target_is_proposal = any(self._norm(x) == proposal_norm for x in target_names if self._norm(x))
        candidate_items = list(trace_context.get("candidate_item_names", []) or [])
        unshortlisted = [str(x) for x in candidate_items if self._norm(x) not in shortlist_norm]
        shortlist_without_proposal = [str(x) for x in shortlist if self._norm(x) != proposal_norm]
        contrastive_context = {
            "easy_background_candidates_not_in_shortlist": unshortlisted[:12],
            "near_miss_hesitation_candidates": shortlist_without_proposal,
            "selected_proposal": proposal_name,
        }
        item_skill = dict(((full_user_policy or {}).get("item_selection_skill", {}) or {}))
        current_item_preferences = list(item_skill.get("preferences", []) or [])
        current_recent_signals = list(item_skill.get("recent_signals", []) or [])
        core_rules = list(item_skill.get("active_rules", []) or [])
        candidate_rules = list(item_skill.get("candidate_rules", []) or [])
        history_seed = str(((full_user_policy or {}).get("policy_evolution_state", {}) or {}).get("history_seed", "") or "")
        prompt_payload = {
            "user_id": str(trace_context.get("user_id", "")),
            "outcome_signal": str((trace_context.get("evaluation_result") or {}).get("outcome_signal", "")),
            "primary_failure_level": str((diagnosis or {}).get("primary_failure_level", "")),
            "history_seed": history_seed,
            "supervised_target_items": target_names,
            "target_is_supervision_not_user_choice": True,
            "candidate_items": candidate_items,
            "proposal_item_user_selected": proposal_name,
            "final_item": str(trace_context.get("final_item_name", "") or ""),
            "prior_hint": str((decision_state or {}).get("prior_item", "") or ""),
            "candidate_shortlist": shortlist,
            "target_in_hesitation_shortlist": bool(target_in_shortlist),
            "target_is_first_choice": bool(target_is_proposal),
            "proposal_reason": str((decision_state or {}).get("proposal_reason", "") or ""),
            "proposal_candidate_evidence": proposal_evidence,
            "target_candidate_evidence": target_evidence,
            "target_style_evidence_to_abstract": target_style_evidence[:6],
            "contrastive_context_for_attention_bias": contrastive_context,
            "all_candidate_evidence": candidate_evidence,
            "current_item_preferences": current_item_preferences[:10],
            "current_recent_signals": current_recent_signals[:6],
            "current_core_active_rules": core_rules[:10],
            "current_core_candidate_rules": candidate_rules[:6],
            "deterministic_diagnosis": deterministic,
        }
        dataset_slug = self._dataset_slug(engine).lower()
        if "librarything" in dataset_slug:
            domain_guidance = (
                "For books, use fiction/non-fiction genre, literary form, topic/subject, author style, narrative tone, "
                "era/setting, language/cultural signal, audience/age category, series/franchise relation, "
                "canon/award/niche level, adjacent theme bridge, recent reading drift, weak topic/author bridge, "
                "or stable minority reading cluster.\n"
            )
            domain_examples = (
                "for book examples: translated literary fiction, character-driven historical fiction, feminist nonfiction, "
                "survival memoir, middle-grade fantasy adventure, South African political biography, experimental short fiction, "
                "or classic mystery series signal.\n"
            )
            rule_example = (
                "choose_by=book genre/topic/author-style match to user history; "
                "preserve=translated literary fiction or historical biography candidates; "
                "minority=weak memoir/nonfiction/cultural-language signals or adjacent theme bridges; "
                "avoid_overbias=do not fill hesitation set only with the dominant genre/topic candidates"
            )
        elif "epinions" in dataset_slug:
            domain_guidance = (
                "For products, use product category, use case, brand/manufacturer family, feature/function, price/value tier, "
                "quality/durability, reliability, design/form factor, compatibility/accessory relation, review/rating sentiment, "
                "popularity/niche level, substitute/complement bridge, recent need drift, or stable minority category.\n"
            )
            domain_examples = (
                "for product examples: budget electronics, durable home tools, premium skincare, compatibility accessories, "
                "ergonomic household design, or high-rating quality signal.\n"
            )
            rule_example = (
                "choose_by=product category/use-case match to user history; "
                "preserve=durable home-tool or compatibility-accessory candidates; "
                "minority=niche product category, substitute/complement bridge, or recent need signal; "
                "avoid_overbias=do not fill hesitation set only with dominant category candidates"
            )
        else:
            domain_guidance = (
                "For music, use genre, scene, era, mood, vocal/instrumental style, language/cultural signal, energy, "
                "popularity level, co-occurrence-neighbor signal, recent drift, weak signal, or stable minority cluster.\n"
            )
            domain_examples = (
                "for music examples: regional folk, politically charged spoken-word, experimental emo edge, "
                "R&B/pop crossover, or world-music vocal signal.\n"
            )
            rule_example = (
                "choose_by=music taste cluster match to user history; "
                "preserve=regional folk or experimental emo candidates; "
                "minority=non-English cultural signal or recent co-occurrence bridge; "
                "avoid_overbias=do not fill hesitation set only with dominant genre candidates"
            )
        system_prompt = (
            "You are the COM train-stage User Reasoning Skill evolution analyzer.\n"
            "Your job is to analyze the user's observed Stage-1 item-selection reasoning and produce an incremental user-skill diff.\n"
            "Important distinction: the proposal item is what the user agent selected; the supervised target is NOT a user choice. Do not explain why the user selected the target.\n"
            "First explain why the user selected the proposal: which history, skill, prior-hint, dominant-cluster, recent, or easy-bridge signals made it salient.\n"
            "Then diagnose why that proposal-centered reasoning did or did not preserve the supervised target in the hesitation shortlist.\n"
            "Only after that, abstract any missed target evidence into transferable domain preference evidence. "
            f"{domain_guidance}"
            "The rule field must not contain the exact target item, proposal item, or final item name unless the name is an evidence item already present in history_seed.\n"
            "If current_item_preferences already contain overlapping signals, use one concise canonical attribute in incremental_update rather than creating another near-duplicate.\n"
            "Do not infer dislikes or must-avoid rules from unchosen candidates. Unchosen candidates are only contrastive context for understanding the proposal's attention bias.\n"
            "Use refine-not-replace: output only structured diff entries; never rewrite the whole skill.\n"
            "Confidence is three-level only: low, medium, high. New discovered preferences must start as low-confidence candidates. "
            "A preference may be weakened by one level only when it clearly caused the supervised target to be excluded from the hesitation shortlist. "
            "Do not weaken a preference merely because it was not used or was less salient in this one interaction.\n"
            "CRITICAL: incremental_update entries must be PREFERENCE-LEVEL signals only: short domain labels (2-5 words), not reasoning procedures.\n"
            "CRITICAL: user_skill_diagnosis.rule and additional_updates[].rule must be REASONING-LEVEL strategies, not plain preference labels.\n"
            "A good rule may carry preference-aware ranking criteria, but it must explain HOW the user decides and how the HesitationShortlist is protected.\n"
            "Prefer this compact slot format for item-selection rules:\n"
            "choose_by=[user-specific taste/history evidence for choosing the favorite]; "
            "preserve=[specific non-favorite style signals to keep in the hesitation set]; "
            "minority=[specific weak/minority/recent/language/cultural/co-occurrence signals not to drop]; "
            "avoid_overbias=[dominant-cluster or prior-hint bias to avoid].\n"
            "Do NOT write primary=... or vague preserve rules. preserve/minority must name concrete transferable signals, "
            f"{domain_examples}"
            "Avoid generic rules. Be specific to this user's preference clusters and the observed failure/success.\n"
            "Output strict JSON only."
        )
        user_prompt = (
            "InteractionEvidence:\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False)}\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '  "analysis": "2-4 sentences: why proposal was selected, and why this reasoning did/did not keep the supervised target in the hesitation set",\n'
            '  "proposal_selection_explanation": "which user-history/skill/prior signals made the proposal salient",\n'
            '  "target_omission_explanation": "why the target was omitted or only kept as hesitation; say none if target was first choice",\n'
            '  "incremental_update": {\n'
            '    "reinforced": [{"attribute": "short preference label 2-5 words", "evidence": "why this interaction supports it"}],\n'
            '    "new_preferences": [{"attribute": "short preference label 2-5 words", "evidence": "how target/proposal contrast reveals it"}],\n'
            '    "recent_signals": [{"attribute": "recent drift or short-term signal", "evidence": "supporting recent/history clue"}],\n'
            '    "weakened": [{"attribute": "short preference label 2-5 words", "reason": "why to demote it conservatively"}]\n'
            "  },\n"
            '  "user_skill_diagnosis": {\n'
            '    "target_layer": "item_selection_skill",\n'
            '    "operation": "discover or reinforce or weaken or record_only",\n'
            '    "problem": "specific failed reasoning step in Item selection or HesitationShortlist construction",\n'
            '    "rule_format": "Use slot format: choose_by=...; preserve=...; minority=...; avoid_overbias=...",\n'
            f'    "rule_example": "{rule_example}",\n'
            '    "rule": "choose_by=[...]; preserve=[...]; minority=[...]; avoid_overbias=[...]",\n'
            '    "confidence": 0.05\n'
            "  },\n"
            '  "additional_update_rule_note": "Use only extra reasoning strategies here; do not restate incremental_update taste labels as rules.",\n'
            '  "additional_updates": [\n'
            '    {"target_layer": "item_selection_skill", "operation": "discover or weaken", "problem": "...", "rule": "choose_by=[...]; preserve=[...]; minority=[...]; avoid_overbias=[...]", "confidence": 0.05}\n'
            "  ]\n"
            "}\n"
        )
        try:
            if not hasattr(engine.args, "max_retry_num"):
                setattr(engine.args, "max_retry_num", 3)
            if not hasattr(engine.args, "temperature"):
                setattr(engine.args, "temperature", 0.2)
            payload = self._extract_json_object(llm_request(system_prompt, user_prompt, engine.args))
        except Exception as exc:
            print(f"[com] LLM user skill evolution failed user={trace_context.get('user_id', '')}: {exc}")
            return diagnosis
        if not isinstance(payload, dict):
            return diagnosis

        refined = dict(diagnosis or {})
        analysis_text = " ".join(str(payload.get("analysis") or payload.get("reflection_summary") or "").split())
        forbidden_names = []
        forbidden_names.extend(target_names)
        forbidden_names.extend(candidate_items)
        prior_hint = str((decision_state or {}).get("prior_item", "") or "").strip()
        if prior_hint:
            forbidden_names.append(prior_hint)
        for key in ["proposal_item_name", "final_item_name"]:
            value = str(trace_context.get(key, "") or "").strip()
            if value:
                forbidden_names.append(value)
        forbidden_names = sorted({x for x in forbidden_names if x}, key=len, reverse=True)
        incremental_update = self._sanitize_incremental_update(payload.get("incremental_update"), forbidden_names)
        incremental_update, skipped_weakened = self._gate_weakened_incremental_updates(
            incremental_update,
            target_in_shortlist=target_in_shortlist,
            target_is_proposal=target_is_proposal,
        )
        user_update = self._generalize_item_specific_update(
            self._sanitize_user_update(payload.get("user_skill_diagnosis"), fallback={}),
            forbidden_item_names=forbidden_names,
        )
        if user_update.get("rule") and not self._is_reasoning_strategy_rule(user_update.get("rule")):
            user_update["rule"] = ""
        if not user_update.get("rule") and deterministic.get("rule"):
            user_update = self._generalize_item_specific_update(
                self._sanitize_user_update(deterministic),
                forbidden_item_names=forbidden_names,
            )
        if user_update.get("rule") and not self._is_reasoning_strategy_rule(user_update.get("rule")):
            user_update["rule"] = ""
        additional = []
        for row in list(payload.get("additional_updates", []) or []):
            clean = self._generalize_item_specific_update(
                self._sanitize_user_update(row),
                forbidden_item_names=forbidden_names,
            )
            if clean.get("rule") and self._is_reasoning_strategy_rule(clean.get("rule")):
                additional.append(clean)
        if not additional:
            for row in list(deterministic.get("additional_updates", []) or []):
                clean = self._generalize_item_specific_update(
                    self._sanitize_user_update(row),
                    forbidden_item_names=forbidden_names,
                )
                if clean.get("rule") and self._is_reasoning_strategy_rule(clean.get("rule")):
                    additional.append(clean)
        user_update["additional_updates"] = additional[:4]
        refined["user_skill_diagnosis"] = user_update
        refined["llm_reflection_summary"] = analysis_text
        refined["proposal_selection_explanation"] = " ".join(str(payload.get("proposal_selection_explanation", "") or "").split())
        refined["target_omission_explanation"] = " ".join(str(payload.get("target_omission_explanation", "") or "").split())
        refined["incremental_update"] = incremental_update
        refined["skipped_weakened_updates"] = int(skipped_weakened)
        refined["forbidden_item_names"] = forbidden_names
        refined["llm_reflection_used"] = True
        return refined

    def evolve(
        self,
        engine,
        user_raw,
        full_user_policy,
        path,
        evaluation_result,
        trace_context=None,
    ):
        if not bool(getattr(engine.args, "com_llm_evolve_user_skill", True)):
            return dict(full_user_policy or {}), {
                "skipped": True,
                "reason": "com_llm_evolve_user_skill_false",
            }

        outcome = str((evaluation_result or {}).get("outcome_signal", "") or "")
        if outcome not in ["TT", "WT", "TW", "WW"]:
            return dict(full_user_policy or {}), {}

        trace_context = dict(trace_context or {})
        trace_context["user_id"] = str(user_raw)
        trace_context["path"] = dict(path or {})
        trace_context["evaluation_result"] = dict(evaluation_result or {})
        stage1_only_arg = bool(getattr(engine.args, "com_stage1_only", False))
        stage1_trace_only = bool(trace_context.get("stage1_only", False) or (evaluation_result or {}).get("stage1_only", False))
        train_item_during_communication = bool(getattr(engine.args, "com_train_item_during_communication", False))
        try:
            initial_hit = bool((evaluation_result or {}).get("initial_hit", outcome in ["TT", "TW"]))
        except Exception:
            initial_hit = outcome in ["TT", "TW"]
        final_hit = outcome in ["TT", "WT"]
        stage1_only = bool(stage1_only_arg or stage1_trace_only)
        decision_state = dict(trace_context.get("decision_state", {}) or {})
        communication_gate = dict(decision_state.get("communication_training_gate", {}) or {})
        communication_gate_eligible = bool(
            communication_gate.get(
                "eligible",
                (evaluation_result or {}).get("communication_train_eligible", True),
            )
        )
        target_in_focus = bool(
            communication_gate_eligible
            or communication_gate.get("target_in_stage1_decision_scope", False)
            or (evaluation_result or {}).get("communication_train_eligible", False)
            or (evaluation_result or {}).get("focus_target_overlap", False)
        )
        communication_target_gate_exempt = bool(
            trace_context.get("communication_target_gate_exempt", False)
            or trace_context.get("communication_continuation", False)
            or (evaluation_result or {}).get("communication_target_gate_exempt", False)
            or (evaluation_result or {}).get("communication_continuation", False)
            or decision_state.get("communication_target_gate_exempt", False)
            or decision_state.get("communication_continuation", False)
            or communication_gate.get("communication_target_gate_exempt", False)
            or communication_gate.get("communication_continuation", False)
        )
        communication_ineligible_for_training = (
            not stage1_only
            and bool(getattr(engine.args, "com_train_communication_eligible_only", True))
            and not communication_gate_eligible
            and not communication_target_gate_exempt
        )
        if communication_ineligible_for_training and not (train_item_during_communication and not initial_hit):
            gate_reason = str(communication_gate.get("reason", "") or "communication_training_gate_ineligible")
            diagnosis = {
                "skipped": True,
                "reason": gate_reason,
                "outcome_signal": outcome,
                "communication_train_eligible": False,
                "communication_target_gate_exempt": False,
                "communication_training_gate": dict(communication_gate or {}),
            }
            self._log_communication_evolution_gate(
                engine,
                user_raw,
                trace_context,
                outcome,
                "skip_before_evolution",
                skip_reason=gate_reason,
                communication_train_eligible=False,
                final_hit=bool(final_hit),
                initial_hit=bool(initial_hit),
            )
            engine.user_policy_store.append_interaction_diagnosis(user_raw, diagnosis)
            return dict(full_user_policy or {}), diagnosis
        diagnosis = {}
        path = dict(trace_context.get("path") or path or {})
        if (
            bool(path.get("unmapped_task", False))
            and str(path.get("what", "") or "") == "none"
            and str(path.get("user_task", "") or "").strip()
        ):
            tree_signal = {
                "level": "what",
                "suggested_node_hint": "unmapped-followup-task",
                "why_current_nodes_insufficient": "A non-empty user communication task could only run through the generic what=none template.",
                "evidence_pattern": "unmapped user task preserved as what=none during communication execution",
                "support_strength": "single_user_low",
                "source": "deterministic_task_mapping",
            }
            existing_signals = [
                row for row in list(path.get("tree_need_signals", []) or [])
                if isinstance(row, dict)
            ]
            if not any(
                str(row.get("suggested_node_hint", "") or "") == tree_signal["suggested_node_hint"]
                and str(row.get("level", "") or "") == "what"
                for row in existing_signals
            ):
                existing_signals.append(tree_signal)
            path["tree_need_signals"] = existing_signals
            trace_context["path"] = path
            diagnosis["unmapped_task_tree_signal"] = tree_signal
            quality_issues = [
                str(x) for x in list(diagnosis.get("communication_quality_issues", []) or [])
                if str(x).strip()
            ]
            if "unmapped_what_task" not in quality_issues:
                quality_issues.append("unmapped_what_task")
            diagnosis["communication_quality_issues"] = quality_issues
        unmapped_round_tasks = []
        for ctx in list(trace_context.get("round_trace_contexts", []) or []):
            ctx_path = dict((ctx or {}).get("path", {}) or {})
            if (
                bool(ctx_path.get("unmapped_task", False))
                and str(ctx_path.get("what", "") or "") == "none"
                and str(ctx_path.get("user_task", "") or "").strip()
            ):
                unmapped_round_tasks.append(
                    {
                        "round_index": int((ctx or {}).get("round_id", 0) or 0),
                        "trigger_signature": str(
                            dict(ctx_path.get("planner_log", {}) or {}).get("trigger_signature", "")
                            or self._route_trigger_signature_from_path(ctx_path)
                        ),
                        "raw_task": self._short_text(ctx_path.get("user_task", ""), 260),
                        "mapped_what": "none",
                        "why_unmapped": "FeedbackToAdvisors did not map to any active what node",
                    }
                )
        if unmapped_round_tasks:
            diagnosis["unmapped_followup_tasks"] = unmapped_round_tasks[:6]

        updated = engine.user_policy_store.normalize_policy(dict(full_user_policy or {}))
        item_diagnosis = None
        item_update_applied = False
        communication_diagnosis = None
        communication_update_applied = False

        def apply_stage1_item_update(source):
            item_trace = dict(trace_context or {})
            item_eval = dict((item_trace.get("evaluation_result") or evaluation_result or {}))
            item_eval["stage1_only"] = True
            if not initial_hit:
                item_eval["outcome_signal"] = "WW"
            item_trace["stage1_only"] = True
            item_trace["evaluation_result"] = item_eval
            diagnosis_local = self.analyzer.analyze(item_trace)
            diagnosis_local["item_training_source"] = str(source or "")
            diagnosis_local = self._llm_refine_user_diagnosis(engine, full_user_policy, diagnosis_local, item_trace)
            updated_policy = self._apply_user_rule_update(updated, diagnosis_local)
            updated_policy, structured_preference_update = self._apply_incremental_preference_update(updated_policy, diagnosis_local)
            diagnosis_local["structured_preference_update"] = structured_preference_update
            return updated_policy, diagnosis_local

        if stage1_only:
            if initial_hit:
                item_diagnosis = {
                    "skipped": True,
                    "reason": "stage1_item_training_skipped_initial_hit",
                    "outcome_signal": outcome,
                    "initial_hit": bool(initial_hit),
                    "final_hit": bool(final_hit),
                    "success": bool(final_hit),
                }
            elif stage1_only_arg or train_item_during_communication:
                updated, item_diagnosis = apply_stage1_item_update(
                    "stage1_only" if stage1_only_arg else "communication_gate_stage1_item_update"
                )
                item_update_applied = True
            else:
                item_diagnosis = {
                    "skipped": True,
                    "reason": "stage1_item_training_disabled_during_communication",
                    "outcome_signal": outcome,
                    "initial_hit": bool(initial_hit),
                    "final_hit": bool(final_hit),
                    "success": bool(final_hit),
                }
        else:
            if train_item_during_communication and not initial_hit:
                updated, item_diagnosis = apply_stage1_item_update("full_communication_train_initial_miss")
                item_update_applied = True
            if communication_ineligible_for_training:
                gate_reason = str(communication_gate.get("reason", "") or "communication_training_gate_ineligible")
                communication_diagnosis = {
                    "skipped": True,
                    "reason": gate_reason,
                    "outcome_signal": outcome,
                    "initial_hit": bool(initial_hit),
                    "final_hit": bool(final_hit),
                    "success": bool(final_hit),
                    "communication_train_eligible": False,
                    "communication_target_gate_exempt": False,
                    "communication_training_gate": dict(communication_gate or {}),
                    "structured_preference_update": dict((item_diagnosis or {}).get("structured_preference_update", {}) or {"skipped": True}),
                }
            elif final_hit:
                communication_diagnosis = {
                    "skipped": True,
                    "reason": "final_hit_keep_communication_unchanged",
                    "training_policy": "communication_failure_driven_only",
                    "outcome_signal": outcome,
                    "initial_hit": bool(initial_hit),
                    "final_hit": True,
                    "success": True,
                    "structured_preference_update": dict((item_diagnosis or {}).get("structured_preference_update", {}) or {"skipped": True}),
                }
                self._log_communication_evolution_gate(
                    engine,
                    user_raw,
                    trace_context,
                    outcome,
                    "skip_before_evolution",
                    skip_reason="final_hit_keep_communication_unchanged",
                    communication_train_eligible=True,
                    final_hit=True,
                    initial_hit=bool(initial_hit),
                )
            else:
                diagnosis = self.analyzer.analyze(trace_context)
                diagnosis["communication_train_eligible"] = True
                diagnosis["communication_target_gate_exempt"] = bool(communication_target_gate_exempt)
                diagnosis["structured_preference_update"] = {
                    "skipped": True,
                    "reason": (
                        "item_training_disabled_during_communication"
                        if not train_item_during_communication
                        else "item_training_not_needed_initial_hit"
                    ),
                }
                original_failure_attribution = str(diagnosis.get("failure_attribution", "") or "")
                self._log_communication_evolution_gate(
                    engine,
                    user_raw,
                    trace_context,
                    outcome,
                    "enter_llm0_decision",
                    communication_train_eligible=True,
                    final_hit=False,
                    initial_hit=bool(initial_hit),
                    primary_failure_level=str(diagnosis.get("primary_failure_level", "") or ""),
                    failure_attribution=original_failure_attribution,
                )
                evolution_decision = self._llm_decide_communication_evolution_target(
                    engine,
                    full_user_policy,
                    diagnosis,
                    trace_context,
                    path,
                )
                diagnosis["evolution_decision"] = dict(evolution_decision)
                diagnosis["evolution_decision_source"] = str(evolution_decision.get("rule_source", "") or "")
                decision_name = str(evolution_decision.get("decision", "") or "")
                user_diag = dict(diagnosis.get("user_skill_diagnosis", {}) or {})
                diagnosed_user_layer = UserPolicyStore.canonical_skill_layer(
                    user_diag.get("target_layer", "")
                )
                diagnosis["diagnosed_user_skill_layer"] = diagnosed_user_layer

                def apply_diagnosed_user_skill_update(reason):
                    nonlocal updated
                    if not str(user_diag.get("rule", "") or "").strip():
                        diagnosis["user_skill_update_applied"] = False
                        diagnosis["user_skill_update_reason"] = "empty_user_skill_rule"
                        return False
                    if diagnosed_user_layer == "communication_absorption_skill":
                        diagnosis["user_skill_failure_attribution"] = "user_absorption_failure"
                    updated = self._apply_user_rule_update(updated, diagnosis)
                    if diagnosed_user_layer == "communication_absorption_skill":
                        updated = self._apply_absorption_case_update(updated, diagnosis, trace_context)
                    diagnosis["user_skill_update_applied"] = True
                    diagnosis["user_skill_update_reason"] = str(reason or "")
                    return True

                if decision_name == "user_absorption_update":
                    if original_failure_attribution != "user_absorption_failure":
                        diagnosis["original_failure_attribution"] = original_failure_attribution
                    diagnosis["failure_attribution"] = "user_absorption_failure"
                    diagnosis["tree_signal_skipped"] = True
                    apply_diagnosed_user_skill_update("llm_decision_user_absorption_update")
                    route_diff_update = {
                        "applied": False,
                        "reason": "llm_decision_user_absorption_updates_communication_absorption_skill",
                    }
                elif decision_name == "public_tree_need":
                    diagnosis["original_failure_attribution"] = original_failure_attribution
                    diagnosis["failure_attribution"] = "tree_defect"
                    tree_signal = self._public_tree_need_signal_from_decision(evolution_decision, path, diagnosis)
                    comm_diff = dict(diagnosis.get("communication_diff", {}) or {})
                    tree_signals = [
                        row for row in list(comm_diff.get("tree_need_signals", []) or [])
                        if isinstance(row, dict)
                    ]
                    tree_signals.append(tree_signal)
                    comm_diff["tree_need_signals"] = tree_signals
                    diagnosis["communication_diff"] = comm_diff
                    diagnosis["tree_signal_skipped"] = False
                    if diagnosed_user_layer in ["communication_absorption_skill", "communication_selection_skill"]:
                        apply_diagnosed_user_skill_update(
                            f"diagnosed_{diagnosed_user_layer}_preserved_alongside_public_tree_need"
                        )
                    route_diff_update = {
                        "applied": False,
                        "reason": "public_tree_need_wait_for_batch_node_injection",
                        "user_skill_update_applied": bool(diagnosis.get("user_skill_update_applied", False)),
                        "user_skill_update_reason": str(diagnosis.get("user_skill_update_reason", "") or ""),
                    }
                else:
                    diagnosis["tree_signal_skipped"] = True
                    updated, route_diff_update = self._apply_route_diff(updated, path, diagnosis)
                diagnosis["route_diff_update"] = route_diff_update
                self._log_communication_evolution_gate(
                    engine,
                    user_raw,
                    trace_context,
                    outcome,
                    "llm0_decision_applied",
                    decision=decision_name,
                    confidence=str(evolution_decision.get("confidence", "") or ""),
                    level=str(evolution_decision.get("level", "") or ""),
                    reason=str(evolution_decision.get("reason", "") or ""),
                    communication_train_eligible=True,
                    final_hit=False,
                    initial_hit=bool(initial_hit),
                    route_diff_applied=bool(route_diff_update.get("applied", False)),
                    route_diff_reason=str(route_diff_update.get("reason", "") or ""),
                    tree_signal_skipped=bool(diagnosis.get("tree_signal_skipped", False)),
                )
                communication_diagnosis = diagnosis
                communication_update_applied = True
                try:
                    policy_paths = engine.user_policy_store._paths(user_raw)
                    dump_json(
                        policy_paths["dir"] / "references" / "communication_reflection_latest.json",
                        self._json_safe(diagnosis.get("communication_reflection", {}) or {}),
                    )
                    append_jsonl(
                        policy_paths["dir"] / "references" / "communication_route_diff_log.jsonl",
                        self._json_safe({
                            "outcome_signal": outcome,
                            "route_diff_update": route_diff_update,
                            "communication_reflection_summary": str(diagnosis.get("communication_reflection_summary", "") or ""),
                        }),
                    )
                    if bool(getattr(engine.args, "com_save_dialogue", False)):
                        dialogue_log = self._communication_round_dialogue_log(
                            trace_context,
                            diagnosis=diagnosis,
                            path=path,
                            outcome_signal=outcome,
                        )
                        dump_json(
                            policy_paths["dir"] / "references" / "communication_round_dialogue_latest.json",
                            dialogue_log,
                        )
                        append_jsonl(
                            policy_paths["dir"] / "references" / "communication_round_dialogue_log.jsonl",
                            dialogue_log,
                        )
                        append_jsonl(
                            engine.public_tree_store.index_dir / "communication_round_dialogues.jsonl",
                            dialogue_log,
                        )
                except Exception:
                    pass
        diagnosis = communication_diagnosis or item_diagnosis or {
            "skipped": True,
            "reason": "no_item_or_communication_update_needed",
            "outcome_signal": outcome,
            "initial_hit": bool(initial_hit),
            "final_hit": bool(final_hit),
            "success": bool(final_hit),
        }
        if item_diagnosis is not None or communication_diagnosis is not None:
            diagnosis = dict(diagnosis or {})
            diagnosis["item_training_applied"] = bool(item_update_applied)
            diagnosis["communication_training_applied"] = bool(communication_update_applied)
            if item_diagnosis is not None:
                diagnosis["item_training_diagnosis"] = dict(item_diagnosis or {})
            if communication_diagnosis is not None:
                diagnosis["communication_training_diagnosis"] = dict(communication_diagnosis or {})
            if item_update_applied and not communication_update_applied:
                diagnosis["skipped"] = False
        engine.user_policy_store.save_full_policy(updated, snapshot_reason=f"interaction_{outcome.lower()}")
        engine.user_policy_store.append_interaction_diagnosis(user_raw, diagnosis)
        engine.user_policy_store.append_evolution_log(
            user_raw,
            {
                "event": "user_reasoning_skill_update",
                "diagnosis_id": str(diagnosis.get("diagnosis_id", "") or ""),
                "outcome_signal": outcome,
                "primary_failure_level": str(diagnosis.get("primary_failure_level", "") or ""),
                "llm_reflection_used": bool(diagnosis.get("llm_reflection_used", False)),
                "llm_reflection_summary": str(diagnosis.get("llm_reflection_summary", "") or ""),
                "incremental_update": dict(diagnosis.get("incremental_update", {}) or {}),
                "skipped_weakened_updates": int(diagnosis.get("skipped_weakened_updates", 0) or 0),
                "structured_preference_update": dict(diagnosis.get("structured_preference_update", {}) or {}),
                "communication_reflection_summary": str(diagnosis.get("communication_reflection_summary", "") or ""),
                "path_effect_explanation": str(diagnosis.get("path_effect_explanation", "") or ""),
                "communication_diff": dict(diagnosis.get("communication_diff", {}) or {}),
                "communication_reflection": dict(diagnosis.get("communication_reflection", {}) or {}),
                "route_diff_update": dict(diagnosis.get("route_diff_update", {}) or {}),
                "communication_evolution_skipped": bool(diagnosis.get("communication_evolution_skipped", False)),
                "communication_evolution_skip_reason": str(diagnosis.get("communication_evolution_skip_reason", "") or ""),
                "evolution_decision": dict(diagnosis.get("evolution_decision", {}) or {}),
                "user_skill_diagnosis": dict(diagnosis.get("user_skill_diagnosis", {}) or {}),
            },
        )

        tree_signal_skipped = bool(diagnosis.get("tree_signal_skipped", False))
        if communication_update_applied and not tree_signal_skipped and not bool(getattr(engine.args, "com_stage1_only", False)):
            tree_diagnosis = self._build_route_aware_tree_diagnosis(diagnosis, trace_context, path)
            for signal in list(tree_diagnosis.get("non_tree_repair_signals", []) or []):
                append_jsonl(engine.public_tree_store.index_dir / "non_tree_repair_signals.jsonl", signal)
            attribution = str(tree_diagnosis.get("failure_attribution", "") or diagnosis.get("failure_attribution", "") or "")
            if attribution in ["aggregation_or_parser_defect", "user_absorption_failure"]:
                append_jsonl(
                    engine.public_tree_store.index_dir / "non_tree_repair_signals.jsonl",
                    {
                        "event": "failure_attribution_non_tree",
                        "ts": int(time.time()),
                        "user_id": str(trace_context.get("user_id", "") or ""),
                        "outcome_signal": str(diagnosis.get("outcome_signal", "") or ""),
                        "failure_attribution": attribution,
                        "primary_failure_level": str(diagnosis.get("primary_failure_level", "") or ""),
                        "reason": self._short_text(
                            diagnosis.get("path_effect_explanation")
                            or diagnosis.get("communication_reflection_summary")
                            or tree_diagnosis.get("failure_reason", ""),
                            700,
                        ),
                    },
                )
            self._record_sprout_trials(engine, tree_diagnosis)
            if attribution == "tree_defect" and self._is_effective_tree_diagnosis(tree_diagnosis):
                tree_diagnosis = self._with_dataset_evolution_metadata(engine, tree_diagnosis)
                append_jsonl(self._tree_evolution_buffer_path(engine), tree_diagnosis)
                self._log_communication_evolution_gate(
                    engine,
                    user_raw,
                    trace_context,
                    outcome,
                    "tree_buffer_written",
                    decision=str((diagnosis.get("evolution_decision", {}) or {}).get("decision", "") or ""),
                    attribution=attribution,
                    effective_tree_diagnosis=True,
                    pending_tree_signal_count=len(list((tree_diagnosis.get("tree_need_signals", []) or []))),
                )
            elif attribution == "tree_defect":
                self._log_communication_evolution_gate(
                    engine,
                    user_raw,
                    trace_context,
                    outcome,
                    "tree_buffer_skipped",
                    decision=str((diagnosis.get("evolution_decision", {}) or {}).get("decision", "") or ""),
                    attribution=attribution,
                    effective_tree_diagnosis=False,
                    skip_reason="ineffective_tree_diagnosis",
                )
            if not diagnosis.get("success"):
                append_jsonl(engine.public_tree_store.index_dir / "failure_diagnoses.jsonl", tree_diagnosis)
        return updated, diagnosis

    def _compact_tree_node_payload(self, node):
        node = dict(node or {})
        out = {}
        for key in ["node_id", "parent_node", "status", "use_why", "if_selected", "skill_path", "node_level", "level"]:
            value = node.get(key)
            if value not in [None, "", [], {}]:
                out[key] = value
        if not out and node:
            for key in ["name", "description", "status"]:
                value = node.get(key)
                if value not in [None, "", [], {}]:
                    out[key] = self._short_text(value, 220)
        return out

    def _compact_path_for_tree_buffer(self, path):
        path = dict(path or {})
        out = dict(path)
        if isinstance(out.get("planner_log"), dict):
            planner = dict(out.get("planner_log") or {})
            if "selected_from_order" in planner:
                planner["selected_from_order"] = self._json_safe(planner.get("selected_from_order"))
            out["planner_log"] = planner
        payload = dict(out.get("path_skill_payload", {}) or {})
        if payload:
            compact_payload = {}
            for layer, node in payload.items():
                compact_payload[layer] = self._compact_tree_node_payload(node)
            out["path_skill_payload"] = compact_payload
        for key in ["skill_body", "system_prompt", "user_prompt"]:
            if key in out:
                out[key] = self._short_text(out.get(key), 400)
        return self._json_safe(out)

    def _route_path_summary_for_prompt(self, path):
        path = dict(path or {})
        planner = dict(path.get("planner_log", {}) or {})
        signature = dict(planner.get("state_signature", {}) or {})
        out = {
            'why': str(path.get('why', "") or ""),
            "what": str(path.get("what", "") or ""),
            "who": str(path.get("who", "") or ""),
            "how": str(path.get("how", "") or ""),
            "user_task": self._short_text(path.get("user_task", ""), 260),
            "unmapped_task": bool(path.get("unmapped_task", False)),
        }
        task_type = str(path.get("task_type_hint", "") or signature.get("task_type_hint", "") or "")
        if task_type:
            out["task_type_hint"] = self._short_text(task_type, 120)
        targets = path.get("task_targets") or signature.get("task_targets") or []
        if isinstance(targets, (list, tuple)):
            clean_targets = [str(x) for x in targets if str(x).strip()]
            if clean_targets:
                out["task_targets"] = clean_targets[:6]
        expected = str(path.get("expected_output", "") or path.get("expected_behavior", "") or "")
        if expected:
            out["expected_output"] = self._short_text(expected, 220)
        return self._json_safe({k: v for k, v in out.items() if v not in ["", [], {}, None]})

    def _compact_round_context_for_tree_buffer(self, ctx):
        ctx = dict(ctx or {})
        out = {
            "round_id": ctx.get("round_id"),
            "path": self._compact_path_for_tree_buffer(ctx.get("path", {})),
            "evaluation_result": dict(ctx.get("evaluation_result", {}) or {}),
        }
        execution = dict(ctx.get("execution_packet", {}) or {})
        if execution:
            advisor_feedbacks = [
                row for row in list(execution.get("advisor_feedbacks", []) or [])
                if isinstance(row, dict)
            ]
            committee_packet = execution.get("committee_packet", {}) or {}
            out["execution_packet"] = {
                "task": self._short_text(execution.get("task") or execution.get("user_task", ""), 320),
                "previous_user_feedback": self._short_text(execution.get("previous_user_feedback", ""), 260),
                "advisor_feedbacks": [
                    self._advisor_turn_reflection_summary(row)
                    for row in advisor_feedbacks[:4]
                ],
                "committee_packet": self._compact_committee_packet(committee_packet),
            }
        for key in ["redecision_packet", "user_response_after_round"]:
            if isinstance(ctx.get(key), dict):
                out[key] = self._json_safe(ctx.get(key))
        return self._json_safe(out)

    def _compact_committee_packet(self, packet):
        packet = dict(packet or {})
        synthesis = dict(packet.get("advisor_synthesis_packet", {}) or {})
        interaction = dict(synthesis.get("interaction_summary", {}) or {})
        return {
            "aggregation_mode": str(packet.get("aggregation_mode", "") or "summary_agent_v1"),
            "decision_policy": str(packet.get("decision_policy", "") or synthesis.get("decision_policy", "") or "information_only_no_vote"),
            "summary_agent_parse_status": str(packet.get("summary_agent_parse_status", "") or ""),
            "legacy_aggregation_used": bool(packet.get("legacy_aggregation_used", False)),
            "advisor_synthesis_packet": {
                "what_was_answered": self._short_text(synthesis.get("what_was_answered", ""), 360),
                "candidate_summaries": dict(synthesis.get("candidate_summaries", {}) or {}),
                "task_specific_summary": dict(synthesis.get("task_specific_summary", {}) or {}),
                "interaction_summary": {
                    "main_disagreements": list(interaction.get("main_disagreements", []) or [])[:4],
                    "corrections_or_rebuttals": list(interaction.get("corrections_or_rebuttals", []) or [])[:4],
                    "unresolved_conflicts": list(interaction.get("unresolved_conflicts", []) or [])[:4],
                },
                "remaining_uncertainty": list(synthesis.get("remaining_uncertainty", []) or [])[:5],
            },
        }

    def _route_selection_evidence(self, trace_context, path):
        path = dict(path or {})
        planner = dict(path.get("planner_log", {}) or {})
        selected = dict(planner.get("selected_action", {}) or {})
        trigger_signature = str(
            planner.get("trigger_signature", "")
            or path.get("trigger_signature", "")
            or self._route_trigger_signature_from_path(path)
        )
        matched_why = list(planner.get("matched_why", []) or path.get("matched_why", []) or [])
        selected_order = dict(planner.get("selected_from_order", {}) or {})

        def as_list(value):
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, dict):
                return [value]
            if str(value or "").strip():
                return [str(value)]
            return []

        return self._json_safe(
            {
                "route_skill_used": bool(planner.get("route_skill_used", False)),
                "template_id": str(planner.get("template_id", "") or ""),
                "trigger_signature": trigger_signature,
                "matched_why": matched_why,
                "selected_order_snapshot": {
                    "what_order": as_list(selected_order.get("what_order", []))[:8],
                    "how_order": as_list(selected_order.get("how_order", []))[:8],
                    "who_order": as_list(selected_order.get("who_order", []))[:8],
                },
                "selected_nodes": {
                    "what": str(path.get("what", "") or selected.get("what", "") or ""),
                    "how": str(path.get("how", "") or selected.get("how", "") or ""),
                    "who": str(path.get("who", "") or selected.get("who", "") or ""),
                },
                "selection_reasons": as_list(planner.get("selection_reasons", []) or planner.get("path_reason", []) or [])[:8],
                "demotions_applied": as_list(planner.get("demotions_applied", []) or [])[:8],
                "exploration_nodes_considered": as_list(planner.get("exploration_nodes_considered", []) or [])[:8],
                "fallback_used": bool(planner.get("fallback_used", False)),
            }
        )

    def _advisor_execution_evidence(self, trace_context):
        contexts = list((trace_context or {}).get("round_trace_contexts", []) or [])
        execution_packets = [dict((ctx or {}).get("execution_packet", {}) or {}) for ctx in contexts]
        if not execution_packets:
            execution_packets = [dict((trace_context or {}).get("execution_packet", {}) or {})]
        advisor_turns = []
        protocol_issues = []
        candidate_views = 0
        ask_user_count = 0
        response_count = 0
        challenge_count = 0
        for packet in execution_packets:
            for fb in list(packet.get("advisor_feedbacks", []) or []):
                if not isinstance(fb, dict):
                    continue
                issues = [str(x) for x in list(fb.get("protocol_issues", []) or []) if str(x).strip()]
                protocol_issues.extend(issues)
                raw_cv = fb.get("candidate_view", fb.get("candidate_views", []))
                if isinstance(raw_cv, list):
                    cv = raw_cv
                elif isinstance(raw_cv, tuple):
                    cv = list(raw_cv)
                elif isinstance(raw_cv, dict):
                    cv = [raw_cv]
                elif str(raw_cv or "").strip():
                    cv = [str(raw_cv)]
                else:
                    cv = []
                ask_user = str(fb.get("ask_user", "") or "").strip()
                response = str(fb.get("response_to_previous", "") or "").strip()
                challenge = str(fb.get("challenge_or_support_previous", "") or "").strip()
                candidate_views += len(cv)
                ask_user_count += 1 if ask_user and ask_user.lower() != "none" else 0
                response_count += 1 if response and response.lower() != "none" else 0
                challenge_count += 1 if challenge and challenge.lower() != "none" else 0
                advisor_turns.append(
                    {
                        "advisor_id": str(fb.get("advisor_id", fb.get("advisor", "")) or ""),
                        "advisor_type": str(fb.get("advisor_type", fb.get("advisor_role", "")) or ""),
                        "candidate_view_count": len(cv),
                        "has_task_answer": bool(str(fb.get("task_answer", "") or "").strip()),
                        "has_ask_user": bool(ask_user and ask_user.lower() != "none"),
                        "has_response_to_previous": bool(response and response.lower() != "none"),
                        "has_challenge_or_support_previous": bool(challenge and challenge.lower() != "none"),
                        "protocol_issues": issues[:6],
                    }
                )
        advisor_count = len(advisor_turns)
        unique_claim_shapes = {
            (
                str(row.get("candidate_view_count", "")),
                str(row.get("has_ask_user", "")),
                str(row.get("has_response_to_previous", "")),
                str(row.get("has_challenge_or_support_previous", "")),
            )
            for row in advisor_turns
        }
        if advisor_count <= 1:
            diversity = "unknown"
        elif len(unique_claim_shapes) >= max(2, advisor_count // 2):
            diversity = "medium"
        else:
            diversity = "low"
        if candidate_views >= max(1, advisor_count):
            coverage = "full"
        elif candidate_views > 0:
            coverage = "partial"
        else:
            coverage = "none"
        specificity = "missing" if candidate_views == 0 else ("generic" if "generic_advisor_answer" in protocol_issues else "concrete")
        aggregation_loss = False
        for packet in execution_packets:
            committee = dict(packet.get("committee_packet", {}) or {})
            questions = list(committee.get("advisor_questions_for_user", []) or [])
            evidence_text = str(committee.get("candidate_evidence_text", "") or committee.get("discussion_summary", "") or "")
            if ask_user_count and not questions:
                aggregation_loss = True
            if candidate_views and not evidence_text.strip():
                aggregation_loss = True
        return self._json_safe(
            {
                "advisor_count": advisor_count,
                "candidate_coverage": coverage,
                "advisor_diversity": diversity,
                "protocol_enforced": not any("protocol" in str(x).lower() for x in protocol_issues),
                "evidence_specificity": specificity,
                "advisor_turns_summary": advisor_turns[:6],
                "aggregation_loss_signal": bool(aggregation_loss),
            }
        )

    def _build_route_aware_tree_diagnosis(self, diagnosis, trace_context, path):
        diagnosis = dict(diagnosis or {})
        trace_context = dict(trace_context or {})
        path = dict(path or trace_context.get("path", {}) or {})
        base = dict(diagnosis.get("tree_diagnosis", {}) or {})
        route_selection = self._route_selection_evidence(trace_context, path)
        advisor_execution = self._advisor_execution_evidence(trace_context)
        public_tree_need_anchor = dict(diagnosis.get("public_tree_need_anchor", {}) or {})
        if not public_tree_need_anchor:
            decision = dict(diagnosis.get("evolution_decision", {}) or {})
            if str(decision.get("decision", "") or "") == "public_tree_need":
                public_tree_need_anchor = self._public_tree_need_anchor_from_decision(decision, path, diagnosis)
        if not public_tree_need_anchor:
            comm_diff = dict(diagnosis.get("communication_diff", {}) or {})
            for signal in list(comm_diff.get("tree_need_signals", []) or []):
                if isinstance(signal, dict):
                    public_tree_need_anchor = dict(signal.get("public_tree_need_anchor", {}) or {})
                    if not public_tree_need_anchor:
                        public_tree_need_anchor = self._public_tree_need_anchor_from_signal(signal, path, diagnosis)
                    break
        compact_rounds = [
            self._compact_round_context_for_tree_buffer(ctx)
            for ctx in list(trace_context.get("round_trace_contexts", []) or [])[:8]
            if isinstance(ctx, dict)
        ]
        compact_path = self._compact_path_for_tree_buffer(path)
        base.update(
            {
                "user_id": str(trace_context.get("user_id", "") or base.get("user_id", "") or ""),
                "success": bool(diagnosis.get("success", base.get("success", False))),
                "outcome_signal": str((trace_context.get("evaluation_result") or {}).get("outcome_signal", "") or diagnosis.get("outcome_signal", "") or base.get("outcome_signal", "") or ""),
                "failure_attribution": str(diagnosis.get("failure_attribution", "") or base.get("failure_attribution", "") or ""),
                "path": compact_path,
                "fine_path": self._compact_path_for_tree_buffer(base.get("fine_path", path)),
                "path_key": str(base.get("path_key") or self._path_key(path)),
                "fine_path_key": str(base.get("fine_path_key") or self._path_key_fine(base.get("fine_path", path))),
                "path_prefixes": list(base.get("path_prefixes", []) or self._path_prefixes(path)),
                "fine_path_prefixes": list(base.get("fine_path_prefixes", []) or self._path_prefixes_fine(base.get("fine_path", path))),
                "failed_level": str(base.get("failed_level", "") or diagnosis.get("primary_failure_level", "") or ""),
                "tree_relevance": str(base.get("tree_relevance", "") or ("high" if str((trace_context.get("evaluation_result") or {}).get("outcome_signal", "")) in ["TW", "WW"] else "")),
                "failure_reason": self._short_text(base.get("failure_reason", "") or diagnosis.get("path_effect_explanation", "") or diagnosis.get("communication_reflection_summary", ""), 700),
                "communication_quality_issues": [
                    str(x) for x in list(base.get("communication_quality_issues", []) or self._diagnosis_quality_issues(diagnosis) or [])
                    if str(x).strip()
                ],
                "route_selection_evidence": route_selection,
                "advisor_execution_evidence": advisor_execution,
                "tree_need_signals": list((diagnosis.get("communication_diff", {}) or {}).get("tree_need_signals", []) or []),
                "non_tree_repair_signals": [],
                "round_trace_contexts": compact_rounds,
                "public_tree_need_anchor": public_tree_need_anchor,
            }
        )
        if str(base.get("failure_attribution", "") or "") != "tree_defect" and str(base.get("outcome_signal", "") or "") in ["TW", "WW"]:
            base["tree_relevance"] = "low"
        if public_tree_need_anchor and str(base.get("failure_attribution", "") or "") == "tree_defect":
            layer = str(public_tree_need_anchor.get("level", "") or "")
            base.setdefault("failed_stage", {"what": "task_mapping", "who": "advisor_selection", "how": "advisor_interaction"}.get(layer, "advisor_interaction"))
            base.setdefault("failure_type", str(public_tree_need_anchor.get("suggested_node_hint", "") or public_tree_need_anchor.get("issue_family", "") or "public_tree_need"))
            base.setdefault(
                "needed_tree_change",
                {
                    "layer": layer,
                    "operation": "add_child_node",
                    "reference_node": str(public_tree_need_anchor.get("reference_node", "") or self._reference_node_for_layer(layer, path)),
                },
            )
        return self._json_safe(base)

    def _aggregate_batch(self, rows):
        prefix_rows = defaultdict(list)
        path_rows = defaultdict(list)
        fine_prefix_rows = defaultdict(list)
        fine_path_rows = defaultdict(list)
        node_rows = defaultdict(list)
        for row in rows or []:
            path = dict(row.get("path", {}) or {})
            fine_path = dict(row.get("fine_path", {}) or path)
            path_key = str(row.get("path_key") or self._path_key(path))
            if path_key.strip(" ->"):
                path_rows[path_key].append(row)
            fine_path_key = str(row.get("fine_path_key") or self._path_key_fine(fine_path))
            if fine_path_key.strip(" ->"):
                fine_path_rows[fine_path_key].append(row)
            for prefix in list(row.get("path_prefixes", []) or self._path_prefixes(path)):
                prefix_rows[prefix].append(row)
            for prefix in list(row.get("fine_path_prefixes", []) or self._path_prefixes_fine(fine_path)):
                fine_prefix_rows[prefix].append(row)
            for level in ['why', "what", "who", "how"]:
                node = str(path.get(level, "") or "")
                if node:
                    node_rows[f"{level}/{node}"].append(row)
            who_subbranch = str(row.get("who_subbranch", "") or fine_path.get("who_subbranch", "") or "")
            if who_subbranch:
                node_rows[f"who_subbranch/trusted-advisors/{who_subbranch}"].append(row)
            for trust_row in list(row.get("advisor_trust_breakdown", []) or []):
                relation = str((trust_row or {}).get("trust_relation", "") or "")
                scope = str((trust_row or {}).get("trust_scope", "") or "")
                similarity = str((trust_row or {}).get("history_similarity_bucket", "") or "")
                subbranch = str((trust_row or {}).get("trust_subbranch", "") or "")
                if relation and relation != "none":
                    node_rows[f"trust_relation/{relation}"].append(row)
                if scope and scope != "none":
                    node_rows[f"trust_scope/{scope}"].append(row)
                if similarity and similarity != "none":
                    node_rows[f"history_similarity/{similarity}"].append(row)
                if subbranch and subbranch != "none":
                    node_rows[f"trust_subbranch/{subbranch}"].append(row)
        return prefix_rows, path_rows, node_rows, fine_prefix_rows, fine_path_rows

    def _stats_for_rows(self, rows):
        rows = list(rows or [])
        n = len(rows)
        outcomes = Counter(str(row.get("outcome_signal", "") or "") for row in rows)
        operations = Counter(str(row.get("suggested_operation", "") or "") for row in rows)
        failure_levels = Counter(str(row.get("failed_level", "") or "") for row in rows if str(row.get("failed_level", "") or ""))
        failures = [str(row.get("failure_reason", "") or "") for row in rows if str(row.get("failure_reason", "") or "")]
        tree_relevant_count = sum(1 for row in rows if str(row.get("tree_relevance", "") or "") == "high")
        dominant_failure = Counter(failures).most_common(1)[0][0] if failures else ""
        success = int(outcomes.get("TT", 0) + outcomes.get("WT", 0))
        fail = int(outcomes.get("TW", 0) + outcomes.get("WW", 0))
        success_rate = float(success / max(1, n))
        min_support = 1
        status = "active"
        last_operation = "record_only"
        if n >= min_support and success_rate >= 0.60 and outcomes.get("TW", 0) == 0:
            status = "active"
            last_operation = "reinforce_branch"
        negative_tree_ops = int(operations.get("weaken_branch", 0) + operations.get("split_branch", 0) + operations.get("grow_branch", 0))
        if tree_relevant_count >= min_support and negative_tree_ops >= min_support and (success_rate <= 0.35 or outcomes.get("TW", 0) >= 1):
            status = "risky"
            last_operation = "weaken_branch"
        if operations.get("split_branch", 0) >= min_support:
            last_operation = "split_branch"
        if operations.get("grow_branch", 0) >= min_support:
            last_operation = "grow_branch"
        return {
            "support": int(n),
            "success": success,
            "fail": fail,
            "success_rate": success_rate,
            "outcomes": dict(outcomes),
            "failure_levels": dict(failure_levels),
            "dominant_failure": dominant_failure,
            "suggested_operations": dict(operations),
            "tree_relevant_count": int(tree_relevant_count),
            "status": status,
            "last_operation": last_operation,
            "updated_at": int(time.time()),
        }

    def _merge_stat_maps(self, old_map, new_map):
        merged = {}
        for key in sorted(set((old_map or {}).keys()) | set((new_map or {}).keys())):
            old = dict((old_map or {}).get(key, {}) or {})
            new = dict((new_map or {}).get(key, {}) or {})
            if not old:
                merged[key] = new
                continue
            if not new:
                merged[key] = old
                continue

            outcomes = Counter(dict(old.get("outcomes", {}) or {}))
            outcomes.update(dict(new.get("outcomes", {}) or {}))
            failure_levels = Counter(dict(old.get("failure_levels", {}) or {}))
            failure_levels.update(dict(new.get("failure_levels", {}) or {}))
            operations = Counter(dict(old.get("suggested_operations", {}) or {}))
            operations.update(dict(new.get("suggested_operations", {}) or {}))
            support = int(old.get("support", 0) or 0) + int(new.get("support", 0) or 0)
            tree_relevant_count = int(old.get("tree_relevant_count", 0) or 0) + int(new.get("tree_relevant_count", 0) or 0)
            success = int(outcomes.get("TT", 0) + outcomes.get("WT", 0))
            fail = int(outcomes.get("TW", 0) + outcomes.get("WW", 0))
            success_rate = float(success / max(1, support))
            status = str(old.get("status", "active") or "active")
            last_operation = str(old.get("last_operation", "record_only") or "record_only")
            if support >= 1 and success_rate >= 0.60 and outcomes.get("TW", 0) == 0:
                status = "active"
                last_operation = "reinforce_branch"
            negative_tree_ops = int(operations.get("weaken_branch", 0) + operations.get("split_branch", 0) + operations.get("grow_branch", 0))
            if tree_relevant_count >= 1 and negative_tree_ops >= 1 and (success_rate <= 0.35 or outcomes.get("TW", 0) >= 1):
                status = "risky"
                last_operation = "weaken_branch"
            if operations.get("split_branch", 0) >= 1:
                last_operation = "split_branch"
            if operations.get("grow_branch", 0) >= 1:
                last_operation = "grow_branch"

            merged[key] = {
                "support": support,
                "success": success,
                "fail": fail,
                "success_rate": success_rate,
                "outcomes": dict(outcomes),
                "failure_levels": dict(failure_levels),
                "dominant_failure": str(new.get("dominant_failure") or old.get("dominant_failure") or ""),
                "suggested_operations": dict(operations),
                "tree_relevant_count": int(tree_relevant_count),
                "status": status,
                "last_operation": last_operation,
                "updated_at": int(time.time()),
            }
        return merged

    @staticmethod
    def _output_field_names_from_lines(lines):
        fields = []
        seen = set()
        for raw in list(lines or []):
            text = str(raw or "").strip()
            if not text:
                continue
            text = re.sub(r"^\s*[-*]\s+", "", text).strip()
            if not text or text.startswith("<"):
                continue
            field = text.split(":", 1)[0].strip()
            if "|" in field:
                field = field.split("|", 1)[0].strip()
            field = re.sub(r"\s+", "", field)
            if not re.match(r"^[A-Za-z][A-Za-z0-9_]{1,48}$", field):
                continue
            key = field.lower()
            if key in seen:
                continue
            seen.add(key)
            fields.append(field)
        return fields

    @staticmethod
    def _what_output_fields_from_skill_md(skill_md):
        text = str(skill_md or "")
        match = re.search(
            r"###\s*Advisor Output Format For This Task\s*(.*?)(?:\n##|\Z)",
            text,
            flags=re.S | re.I,
        )
        if not match:
            return []
        return TrainOnlyEvolver._output_field_names_from_lines(match.group(1).splitlines())

    def _normalized_summary_hints_for_patch(self, layer, skill_json=None, skill_md=""):
        layer = str(layer or "")
        skill_json = dict(skill_json or {})
        hints = dict(skill_json.get("summary_hints", {}) or {})
        if layer == "what":
            fields = []
            fields.extend(self._output_field_names_from_lines(skill_json.get("task_output_format", []) or []))
            fields.extend(self._output_field_names_from_lines(hints.get("important_output_fields", []) or []))
            fields.extend(self._what_output_fields_from_skill_md(skill_md))
            if not fields:
                fields = ["CandidateView", "TaskAnswer", "AskUser"]
            merged_fields = []
            seen = set()
            for field in fields:
                key = str(field or "").lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged_fields.append(str(field))
            preserve = list(hints.get("preserve_interaction_fields", []) or [])
            if not preserve:
                preserve = ["ChallengeOrSupportPrevious", "ResponseToPrevious", "Correction"]
            return {
                "task_focus": self._short_text(
                    hints.get("task_focus", "")
                    or skill_json.get("description", "")
                    or skill_json.get("use_why", "")
                    or "preserve task-specific advisor evidence for this what node",
                    220,
                ),
                "important_output_fields": merged_fields[:8],
                "preserve_interaction_fields": [str(x) for x in preserve[:6] if str(x or "").strip()],
            }
        if layer == "how":
            fields = list(hints.get("important_output_fields", []) or [])
            fields.extend(self._output_field_names_from_lines(skill_json.get("advisor_output_format", []) or []))
            merged_fields = []
            seen = set()
            for field in fields:
                key = str(field or "").lower()
                if key and key not in seen:
                    seen.add(key)
                    merged_fields.append(str(field))
            preserve = list(hints.get("preserve_interaction_fields", []) or [])
            if not preserve:
                preserve = list(skill_json.get("advisor_output_format", []) or [])
            return {
                "task_focus": self._short_text(
                    hints.get("task_focus", "") or "preserve how-specific advisor interaction signals",
                    220,
                ),
                "important_output_fields": merged_fields[:8],
                "preserve_interaction_fields": [str(x) for x in preserve[:6] if str(x or "").strip()],
            }
        return hints

    @staticmethod
    def _tree_patch_template(layer):
        templates = {
            'why': (
                "---\nname: <node_id>\ndescription: <when communication should trigger>\nlevel: when\nstatus: sprout\n---\n\n"
                "# <node_id>\n\n## Use When\n<condition>\n\n## If Selected\nTrigger communication because <reason>.\n\n"
                "## Communication Trigger\nThis node only decides whether communication is needed. It does not select what, who, or how.\n\n"
                "### Goal\n<what uncertainty this trigger protects>\n\n### Required Actions\n- Detect the trigger from current planning state.\n"
                "- Do not force communication if the trigger is absent.\n- Pass the trigger reason to task planning.\n\n"
                "## Runtime Status\nThis node is sprout and may only be trialed during train."
            ),
            "what": (
                "---\nname: <node_id>\ndescription: <user task type>\nlevel: what\nstatus: sprout\n---\n\n"
                "# <node_id>\n\n## Use When\n<what kind of user task maps here>\n\n## If Selected\n"
                "Use this task type to decide what advisor evidence should answer.\n\n## User Task Type\n"
                "Interpret the user's natural-language communication task.\n\n### Goal\n<advisor content goal>\n\n"
                "### Required Actions\n- Use UserTask as the content goal.\n- Keep the answer inside the HesitationSet.\n"
                "- Do not select who or how.\n- Preserve unmapped details as AskUser or StillMissing.\n\n"
                "### Advisor Output Format For This Task\n- CandidateView:\n- <exact candidate> | <task-specific label> | <short reason>\n"
                "- <new task-specific field>: <meaning, if this node needs one>\n"
                "- TaskAnswer: <direct answer to Task>\n- AskUser: <specific question for the user, or none>\n\n"
                "## Skill JSON Requirements\nskill_json.task_output_format must list every advisor output field above. "
                "skill_json.summary_hints.important_output_fields must include CandidateView, TaskAnswer, AskUser, and every new task-specific field so the summary agent preserves them.\n\n"
                "## Runtime Status\nThis node is sprout and may only be trialed during train."
            ),
            "who": (
                "---\nname: <node_id>\ndescription: <advisor source or subbranch>\nlevel: who\nstatus: sprout\n---\n\n"
                "# <node_id>\n\n## Use When\n<when this advisor subgroup is useful>\n\n## If Selected\n"
                "Select advisors matching this concrete subgroup.\n\n## Advisor Role Label\n<role label>\n\n"
                "## Retrieval Policy\nDescribe the concrete advisor subgroup, not an abstract policy. Examples: mutual trusted advisors, one-way trusted advisors, history-dissimilar trusted advisors, item-experienced users, or two-hop social advisors.\n\n## Do Not Use When\n<when this subgroup is weak>\n\n"
                "## Runtime Status\nThis node is sprout and may only be trialed during train."
            ),
            "how": (
                "---\nname: <node_id>\ndescription: <advisor organization mode>\nlevel: how\nstatus: sprout\n---\n\n"
                "# <node_id>\n\n## Use When\n<when this organization helps>\n\n## If Selected\n"
                "Use this communication organization mode.\n\n## Advisor Communication Skill\n<how advisors interact>\n\n"
                "### Goal\n<organization goal>\n\n### Required Actions\n- UserTask defines what content to answer.\n"
                "- This node only defines how advisors organize.\n- Keep continuity with previous discussion memory when available.\n\n"
                "### Output Contract\n- Return task-specific fields from the selected what node.\n"
                "- Add only the how-specific interaction field below.\n\n### Advisor Output Format\n- <how-specific field>: <meaning>\n\n"
                "## Runtime Status\nThis node is sprout and may only be trialed during train."
            ),
        }
        return templates.get(str(layer), "")

    def _tree_patch_forbidden_terms(self, rows):
        terms = []
        for row in list(rows or []):
            trace = dict((row or {}).get("interaction_trace", {}) or {})
            terms.extend([str(x) for x in list(trace.get("candidate_items", []) or []) if str(x).strip()])
            terms.extend([str(x) for x in list(trace.get("target_item", []) or []) if str(x).strip()])
            for key in ["selected_item", "initial_selected_item", "final_selected_item", "user_id"]:
                value = str(trace.get(key, "") or "").strip()
                if value:
                    terms.append(value)
            user_id = str((row or {}).get("user_id", "") or "").strip()
            if user_id:
                terms.append(user_id)
        return sorted({x for x in terms if len(str(x)) >= 3}, key=len, reverse=True)

    @staticmethod
    def _is_effective_tree_diagnosis(row):
        row = dict(row or {})
        outcome = str(row.get("outcome_signal", "") or "")
        if outcome not in ["TW", "WW"]:
            return False
        if str(row.get("failure_attribution", "") or "") != "tree_defect":
            return False
        path = dict(row.get("path", {}) or {})
        if str(path.get('why', "") or "") in ["", "skip", "none"]:
            return False
        round_contexts = list(row.get("round_trace_contexts", []) or [])
        if round_contexts:
            for ctx in round_contexts:
                ctx_path = dict((ctx or {}).get("path", {}) or {})
                if str(ctx_path.get('why', "") or "") not in ["", "skip", "none"]:
                    return True
            return False
        return True

    def _pending_effective_tree_diagnosis_count(self, engine):
        rows = self._load_dataset_tree_evolution_buffer(engine)
        return sum(1 for row in rows if self._is_current_dataset_tree_signal(engine, row) and self._is_effective_tree_diagnosis(row))

    @staticmethod
    def _dataset_slug(engine):
        dataset = str(getattr(engine, "dataset", "") or getattr(getattr(engine, "args", None), "dataset", "") or "default").strip()
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset)
        return safe or "default"

    def _tree_evolution_buffer_path(self, engine):
        return engine.public_tree_store.index_dir / "tree_evolution_buffer.jsonl"

    def _load_dataset_tree_evolution_buffer(self, engine):
        rows = load_jsonl(self._tree_evolution_buffer_path(engine), default=[])
        dataset = self._dataset_slug(engine)
        return [row for row in rows if self._is_current_dataset_tree_signal(engine, row, default_dataset=dataset)]

    def _is_current_dataset_tree_signal(self, engine, row, default_dataset=""):
        row = dict(row or {})
        current = self._dataset_slug(engine)
        signal_dataset = str(row.get("dataset", "") or row.get("source_dataset", "") or default_dataset or "").strip()
        return not signal_dataset or signal_dataset == current

    def _with_dataset_evolution_metadata(self, engine, payload):
        payload = dict(payload or {})
        dataset = self._dataset_slug(engine)
        payload["dataset"] = dataset
        payload["source_dataset"] = dataset
        payload.setdefault("evolution_source", "communication_tree_batch")
        return payload

    def maybe_evolve_tree_batch_by_threshold(self, engine, stage="train", force=False):
        if str(stage or "") != "train":
            return {"skipped": True, "reason": "not_train"}
        if bool(getattr(engine.args, "com_stage1_only", False)):
            return {"skipped": True, "reason": "com_stage1_only"}
        pending = self._pending_effective_tree_diagnosis_count(engine)
        batch_size = int(getattr(engine.args, "com_tree_evolve_batch_size", 50) or 0)
        if not force and (batch_size <= 0 or pending < batch_size):
            return {
                "skipped": True,
                "reason": "below_threshold" if batch_size > 0 else "threshold_disabled",
                "pending_effective": int(pending),
                "threshold": int(batch_size),
            }
        if pending <= 0:
            repair = self.repair_active_tree_route_injections(engine) if force else {}
            return {
                "skipped": True,
                "reason": "no_effective_tree_evidence",
                "pending_effective": 0,
                "route_injection_repair": dict(repair or {}),
            }
        update = self.evolve_tree_batch(engine, stage=stage)
        update["trigger"] = "final_flush" if force else "threshold"
        update["pending_effective_before"] = int(pending)
        update["threshold"] = int(batch_size)
        return update

    def _failure_type_from_round(self, row, ctx, quality_issues, failed_level, session_outcome):
        issues = {str(x or "") for x in list(quality_issues or [])}
        issue_text = " ".join(sorted(issues)).lower()
        failed_level = str(failed_level or "")
        if any(x in issue_text for x in ["feedback_not", "feedback_ignored", "unanswered_feedback"]):
            return "feedback_not_inherited"
        if any(x in issue_text for x in ["evidence_lost", "missing_advisor_evidence", "silent_focus"]):
            return "evidence_lost"
        if any(x in issue_text for x in ["multi_candidate_protocol", "cooperative_monologue", "protocol", "no_candidate_set_signal"]):
            return "protocol_not_enforced"
        if any(x in issue_text for x in ["reason_item_mismatch", "wrong_path", "mismatch"]):
            return "wrong_path"
        if any(x in issue_text for x in ["advisor_pool_empty", "advisor_not", "unresolved_advisor"]):
            return "advisor_not_answering"
        if failed_level == "feedback_absorption":
            return "evidence_lost"
        if failed_level == "advisor_feedback":
            return "advisor_not_answering"
        if failed_level == "communication_protocol":
            return "protocol_not_enforced"
        if failed_level == "path_selection":
            return "wrong_path"
        if str(session_outcome or "") == "TW":
            return "misled_user"
        return "failed_to_recover"

    @staticmethod
    def _failed_stage_for_type(failure_type):
        return {
            "misled_user": "user_redecision",
            "failed_to_recover": "advisor_speaking",
            "advisor_not_answering": "advisor_speaking",
            "feedback_not_inherited": "path_selection",
            "evidence_lost": "aggregation",
            "wrong_path": "path_selection",
            "protocol_not_enforced": "advisor_interaction",
        }.get(str(failure_type or ""), "advisor_interaction")

    @staticmethod
    def _layer_for_failure_type(failure_type, failed_stage):
        failure_type = str(failure_type or "")
        failed_stage = str(failed_stage or "")
        if failed_stage == "path_selection" and failure_type == "wrong_path":
            return "what"
        if failure_type in ["advisor_not_answering", "feedback_not_inherited", "evidence_lost", "failed_to_recover"]:
            return "what"
        if failure_type in ["protocol_not_enforced", "misled_user"]:
            return "how"
        return "how"

    @staticmethod
    def _reference_node_for_layer(layer, path):
        path = dict(path or {})
        layer = str(layer or "")
        if layer == 'why':
            for candidate in list(path.get("matched_why", []) or []):
                candidate = str(candidate or "")
                if candidate and candidate not in ["skip", "none"]:
                    return candidate
        value = str(path.get(layer, "") or "")
        if value and value not in ["skip", "none"]:
            return value
        fallback = {
            'why': "candidate-conflict",
            "what": "compare_remaining_candidates",
            "who": "trusted-advisors",
            "how": "multi-cooperative",
        }
        return fallback.get(layer, "")

    def _round_effect(self, session_outcome, ctx, is_last_round):
        session_outcome = str(session_outcome or "")
        round_outcome = str(((ctx or {}).get("evaluation_result") or {}).get("outcome_signal", "") or "")
        if session_outcome == "TW":
            if round_outcome == "TW" or is_last_round:
                return "harmful"
            return "uncertain_failure"
        if session_outcome == "WW":
            return "ineffective" if round_outcome in ["WW", ""] or is_last_round else "uncertain_failure"
        return "ignore_success"

    def _round_fact_trace_contexts(self, row):
        row = dict(row or {})
        contexts = list(row.get("round_trace_contexts", []) or [])
        if contexts:
            return [dict(ctx or {}) for ctx in contexts if isinstance(ctx, dict)]
        return [
            {
                "round_id": 1,
                "path": dict(row.get("path", {}) or {}),
                "evaluation_result": {
                    "outcome_signal": str(row.get("outcome_signal", "") or ""),
                    "initial_hit": bool(row.get("initial_hit", False)),
                    "final_hit": bool(row.get("final_hit", False)),
                    "focus_target_overlap": bool(row.get("focus_target_overlap", False)),
                    "candidate_target_overlap": bool(row.get("candidate_target_overlap", False)),
                },
                "execution_packet": {},
                "decision_state": {},
                "redecision_packet": {},
            }
        ]

    @staticmethod
    def _text_contains_any(text, tokens):
        text = str(text or "").lower()
        return any(str(token or "").lower() in text for token in list(tokens or []))

    def _feedback_text(self, feedback):
        feedback = dict(feedback or {})
        return " ".join(
            str(feedback.get(key, "") or "")
            for key in [
                "task_answer",
                "ask_user",
                "response_to_previous",
                "challenge_or_support_previous",
                "correction",
                "raw_text",
            ]
        )

    def _candidate_view_rows(self, feedback):
        feedback = dict(feedback or {})
        raw = feedback.get("candidate_views", None)
        if raw in [None, ""]:
            raw = feedback.get("candidate_view", [])
        if isinstance(raw, dict):
            raw = [raw]
        elif isinstance(raw, str):
            raw = [{"view": raw}] if raw.strip() else []
        elif not isinstance(raw, list):
            raw = []
        rows = []
        for item in raw:
            if isinstance(item, dict):
                rows.append(item)
            elif isinstance(item, str) and item.strip():
                rows.append({"view": item})
        return rows

    def _advisor_behavior_counts(self, advisor_feedbacks):
        rows = [dict(row or {}) for row in list(advisor_feedbacks or []) if isinstance(row, dict)]
        counts = {
            "advisor_count": int(len(rows)),
            "later_advisor_count": max(0, int(len(rows)) - 1),
            "answered_task_count": 0,
            "generic_answer_count": 0,
            "candidate_view_count": 0,
            "support_previous_count": 0,
            "challenge_previous_count": 0,
            "correction_count": 0,
            "candidate_contrast_count": 0,
            "later_advisor_repeat_count": 0,
        }
        seen_signatures = set()
        for idx, row in enumerate(rows):
            text = self._feedback_text(row)
            text_low = text.lower()
            task_answer = str(row.get("task_answer", "") or row.get("raw_text", "") or "")
            if task_answer.strip():
                counts["answered_task_count"] += 1
            if not task_answer.strip() or self._text_contains_any(task_answer, ["generally", "could be", "might be", "depends", "similar"]):
                counts["generic_answer_count"] += 1
            candidate_views = self._candidate_view_rows(row)
            if candidate_views:
                counts["candidate_view_count"] += 1
                labels = {
                    str(x.get("view", "") or x.get("label", "") or "").strip().lower()
                    for x in candidate_views
                    if str(x.get("view", "") or x.get("label", "") or "").strip()
                }
                if len(labels) >= 2:
                    counts["candidate_contrast_count"] += 1
            if self._text_contains_any(text_low, ["correction", "correct", "revise", "instead", "actually"]):
                counts["correction_count"] += 1
            interaction_text = str(row.get("challenge_or_support_previous", "") or row.get("response_to_previous", "") or "")
            interaction_low = interaction_text.lower()
            if idx > 0:
                if self._text_contains_any(interaction_low, ["support", "agree", "same", "reinforce", "align"]):
                    counts["support_previous_count"] += 1
                if self._text_contains_any(interaction_low, ["challenge", "question", "disagree", "however", "but", "counter", "oppose", "rebut"]):
                    counts["challenge_previous_count"] += 1
                signature = " ".join(re.findall(r"[a-z0-9]+", text_low)[:24])
                if signature and signature in seen_signatures:
                    counts["later_advisor_repeat_count"] += 1
            signature = " ".join(re.findall(r"[a-z0-9]+", text_low)[:24])
            if signature:
                seen_signatures.add(signature)
        return counts

    def _aggregation_fact_context(self, execution, redecision, advisor_behavior):
        execution = dict(execution or {})
        redecision = dict(redecision or {})
        committee = dict(execution.get("committee_packet", {}) or {})
        evidence_summary = dict(committee.get("evidence_summary", {}) or {})
        summary_text = " ".join(
            [
                str(evidence_summary.get("discussion_summary", "") or ""),
                str((redecision.get("discussion_result", {}) or {}).get("discussion_summary", "") if isinstance(redecision.get("discussion_result"), dict) else ""),
                str(redecision.get("prompt_communication_summary", "") or ""),
            ]
        )
        summary_low = summary_text.lower()
        advisor_had_challenges = int((advisor_behavior or {}).get("challenge_previous_count", 0) or 0) > 0
        advisor_had_corrections = int((advisor_behavior or {}).get("correction_count", 0) or 0) > 0
        return {
            "summary_present": bool(summary_text.strip() or evidence_summary),
            "advisor_had_challenges": bool(advisor_had_challenges),
            "summary_kept_challenges": bool(advisor_had_challenges and self._text_contains_any(summary_low, ["challenge", "disagree", "counter", "question", "however", "but"])),
            "advisor_had_corrections": bool(advisor_had_corrections),
            "summary_kept_corrections": bool(advisor_had_corrections and self._text_contains_any(summary_low, ["correction", "correct", "revise", "actually"])),
            "summary_overstated_consensus": bool(
                self._text_contains_any(summary_low, ["consensus", "all advisors", "unanimous", "strong agreement"])
                and (advisor_had_challenges or advisor_had_corrections)
            ),
        }

    def _build_round_fact_trace(self, row, ctx):
        row = dict(row or {})
        ctx = dict(ctx or {})
        path = dict(ctx.get("path", {}) or row.get("path", {}) or {})
        execution = dict(ctx.get("execution_packet", {}) or {})
        redecision = dict(ctx.get("redecision_packet", {}) or {})
        evaluation = dict(ctx.get("evaluation_result", {}) or {})
        decision_state = dict(ctx.get("decision_state", {}) or {})
        advisor_feedbacks = [dict(x or {}) for x in list(execution.get("advisor_feedbacks", []) or []) if isinstance(x, dict)]
        advisor_execution = dict(row.get("advisor_execution_evidence", {}) or {})
        committee = dict(execution.get("committee_packet", {}) or {})
        focus_candidates = list(execution.get("focus_candidates", []) or decision_state.get("candidate_shortlist", []) or decision_state.get("shortlist", []) or [])
        target_in_focus = bool(
            evaluation.get("focus_target_overlap", False)
            or evaluation.get("candidate_target_overlap", False)
            or row.get("focus_target_overlap", False)
            or row.get("candidate_target_overlap", False)
        )
        advisor_behavior = self._advisor_behavior_counts(advisor_feedbacks)
        aggregation_context = self._aggregation_fact_context(execution, redecision, advisor_behavior)
        try:
            quality_issues = list(self.analyzer._communication_quality_issues(ctx) or [])
        except Exception:
            quality_issues = list(row.get("communication_quality_issues", []) or [])
        task_text = str(execution.get("task") or execution.get("user_task") or path.get("user_task", "") or "")
        selected_what = str(path.get("what", "") or "")
        requires_contrast = bool(
            self._text_contains_any(task_text, ["compare", "contrast", "candidate", "which", "stronger", "weaker"])
            or selected_what in {"compare_remaining_candidates", "reduce_hesitation_set", "reasoning_check"}
        )
        previous_feedback = execution.get("previous_user_feedback", {}) or {}
        if isinstance(previous_feedback, dict):
            previous_feedback_text = str(previous_feedback.get("feedback_to_advisors", "") or previous_feedback.get("raw_feedback", "") or "")
        else:
            previous_feedback_text = str(previous_feedback or "")
        fact = {
            "case_id": f"{row.get('diagnosis_id', row.get('user_id', ''))}#r{ctx.get('round_id', 1)}",
            "user_id": str(row.get("user_id", "") or ""),
            "session_outcome": str(row.get("outcome_signal", "") or ""),
            "round_effect": "",
            "round_index": int(ctx.get("round_id", 1) or 1),
            "path": {
                'why': str(path.get('why', "") or ""),
                "matched_why": list(path.get("matched_why", []) or []),
                "what": selected_what,
                "who": str(path.get("who", "") or ""),
                "how": str(path.get("how", "") or ""),
            },
            "candidate_context": {
                "target_in_focus": bool(target_in_focus),
                "focus_candidate_count": int(len(focus_candidates)),
                "candidate_coverage": str(advisor_execution.get("candidate_coverage", "") or ("full" if focus_candidates else "none")),
                "has_comparable_candidates": bool(len(focus_candidates) >= 2),
            },
            "task_context": {
                "selected_task": self._short_text(task_text, 360),
                "requires_candidate_contrast": bool(requires_contrast),
                "requires_followup_answer": bool(previous_feedback_text.strip() or path.get("unmapped_task", False)),
                "unmapped_task": bool(path.get("unmapped_task", False)),
            },
            "advisor_context": {
                "advisor_count": int(advisor_behavior.get("advisor_count", 0) or 0),
                "advisor_pool_empty": bool(committee.get("advisor_pool_empty", False) or not advisor_feedbacks),
                "advisor_diversity": str(advisor_execution.get("advisor_diversity", "") or "unknown"),
            },
            "advisor_behavior": advisor_behavior,
            "aggregation_context": aggregation_context,
            "user_redecision_context": {
                "initial_hit": bool(evaluation.get("initial_hit", row.get("initial_hit", False))),
                "final_hit": bool(evaluation.get("final_hit", row.get("final_hit", False))),
                "changed_after_communication": bool(evaluation.get("final_equals_prior", None) is False or evaluation.get("proposal_equals_prior", None) is False),
            },
            "quality_issues": [str(x) for x in list(quality_issues or []) if str(x).strip()],
            "raw_evidence_snippets": [
                self._short_text(self._feedback_text(row), 240)
                for row in advisor_feedbacks[:4]
                if self._feedback_text(row).strip()
            ],
            "advisor_speech_summaries": [
                self._advisor_speech_summary_for_batch(row)
                for row in advisor_feedbacks[:4]
            ],
        }
        return self._json_safe(fact)

    def _public_tree_need_case_material(self, row, anchor):
        row = dict(row or {})
        anchor = dict(anchor or {})
        contexts = []
        for ctx in self._round_fact_trace_contexts(row):
            ctx = dict(ctx or {})
            path = dict(ctx.get("path", {}) or row.get("path", {}) or {})
            if str(path.get('why', "") or "") not in ["", "skip", "none"]:
                contexts.append(ctx)
        ctx = contexts[-1] if contexts else {"round_id": 1, "path": dict(row.get("path", {}) or {}), "execution_packet": {}}
        fact = self._build_round_fact_trace(row, ctx)
        missing = str(anchor.get("missing_capability", "") or anchor.get("suggested_node_hint", "") or "public communication capability")
        issues = [
            str(x).strip()
            for x in list(row.get("communication_quality_issues", []) or fact.get("quality_issues", []) or [])
            if str(x).strip()
        ]
        actual = (
            row.get("failure_reason")
            or row.get("path_effect_explanation")
            or row.get("communication_reflection_summary")
            or "; ".join(issues[:4])
            or missing
        )
        return self._json_safe(
            {
                "case_id": f"{row.get('diagnosis_id', row.get('user_id', ''))}#public_tree_need",
                "user_id": str(row.get("user_id", "") or ""),
                "session_outcome": str(row.get("outcome_signal", "") or ""),
                "round_effect": "harmful" if str(row.get("outcome_signal", "") or "") == "TW" else "ineffective",
                "problem_area": str(anchor.get("issue_family", "") or "public_tree_need"),
                "specific_problem": self._short_text(missing, 260),
                "expected_behavior": self._short_text(
                    f"the selected public route should express and enforce this communication capability: {missing}",
                    260,
                ),
                "actual_behavior": self._short_text(actual, 260),
                "repair_hint": self._short_text(missing, 220),
                "diagnostic_tags": {"quality_issues": issues[:8]},
                "fact_trace": fact,
            }
        )

    def _extract_round_failure_reflections(self, rows):
        reflections = []
        for row in list(rows or []):
            row = dict(row or {})
            if str(row.get("outcome_signal", "") or "") not in ["TW", "WW"]:
                continue
            anchor = dict(row.get("public_tree_need_anchor", {}) or {})
            if not anchor:
                for signal in list(row.get("tree_need_signals", []) or []):
                    if isinstance(signal, dict):
                        anchor = dict(signal.get("public_tree_need_anchor", {}) or {})
                        if not anchor:
                            anchor = self._public_tree_need_anchor_from_signal(signal, row.get("path", {}), row)
                        break
            if not anchor:
                continue
            user_id = str(row.get("user_id", "") or "")
            layer = str(anchor.get("level", "") or "how")
            reference = str(anchor.get("reference_node", "") or self._reference_node_for_layer(layer, row.get("path", {})))
            failure_type = str(anchor.get("suggested_node_hint", "") or anchor.get("issue_family", "") or "public_tree_need")
            failed_stage = {
                "what": "task_mapping",
                "who": "advisor_selection",
                "how": "advisor_interaction",
            }.get(layer, "advisor_interaction")
            reflections.append(
                {
                    "round_id": f"{row.get('diagnosis_id', row.get('user_id', ''))}#anchor",
                    "session_outcome": str(row.get("outcome_signal", "") or ""),
                    "round_effect": "harmful" if str(row.get("outcome_signal", "") or "") == "TW" else "ineffective",
                    "failure_type": failure_type,
                    "failed_stage": failed_stage,
                    "path": dict(row.get("path", {}) or {}),
                    "observed_failure": self._short_text(anchor.get("missing_capability", "") or failure_type, 480),
                    "why_current_tree_insufficient": self._short_text(anchor.get("missing_capability", "") or failure_type, 360),
                    "needed_tree_change": {
                        "layer": layer,
                        "operation": "add_child_node",
                        "reference_node": reference,
                    },
                    "abstract_evidence": self._short_text(anchor.get("missing_capability", "") or failure_type, 420),
                    "quality_issues": list(row.get("communication_quality_issues", []) or []),
                    "route_selection_context": dict(row.get("route_selection_evidence", {}) or {}),
                    "advisor_execution_context": dict(row.get("advisor_execution_evidence", {}) or {}),
                    "trigger_signature": str((row.get("route_selection_evidence", {}) or {}).get("trigger_signature", "") or self._route_trigger_signature_from_path(row.get("path", {}))),
                    "confidence": str(anchor.get("confidence", "") or "medium"),
                    "user_id": user_id,
                    "source": "public_tree_need_anchor",
                    "cluster_anchor": anchor,
                    "failure_material_case": self._public_tree_need_case_material(row, anchor),
                }
            )
        return reflections

    def _cluster_failure_reflections(self, reflections):
        clusters = {}
        for ref in list(reflections or []):
            ref = dict(ref or {})
            if str(ref.get("session_outcome", "") or "") not in ["TW", "WW"]:
                continue
            change = dict(ref.get("needed_tree_change", {}) or {})
            layer = str(change.get("layer", "") or "how")
            reference = str(change.get("reference_node", "") or "")
            material_case = dict(ref.get("failure_material_case", {}) or {})
            anchor = dict(ref.get("cluster_anchor", {}) or {})
            if anchor:
                layer = str(anchor.get("level", "") or layer or "how")
                reference = str(anchor.get("reference_node", "") or reference)
                problem_area = str(anchor.get("issue_family", "") or ref.get("failed_stage", "") or "public_tree_need")
                specific_problem = str(anchor.get("missing_capability", "") or ref.get("observed_failure", "") or "")
                repair_hint = str(anchor.get("missing_capability", "") or specific_problem)
                material_signature = [
                    f"issue_family={str(anchor.get('issue_family', '') or '')}",
                    f"suggested_node_hint={str(anchor.get('suggested_node_hint', '') or '')}",
                    f"missing_capability={self._slug_fragment(anchor.get('missing_capability', ''), 'missing-capability')}",
                    f"rule_source={str(anchor.get('rule_source', '') or '')}",
                ]
                key_parts = [
                    layer or "none",
                    reference or "none",
                    problem_area or "public_tree_need",
                    self._material_signature_slug(material_signature or [specific_problem]),
                ]
                cluster_key_version = "public_tree_need_anchor_v1"
            elif material_case:
                problem_area = str(material_case.get("problem_area", "") or ref.get("problem_area", "") or "unknown_or_mixed")
                material_signature = [
                    str(x)
                    for x in list(material_case.get("material_signature", []) or ref.get("material_signature", []) or [])
                    if str(x).strip()
                ]
                specific_problem = str(material_case.get("specific_problem", "") or ref.get("specific_problem", "") or ref.get("observed_failure", "") or "")
                repair_hint = str(material_case.get("repair_hint", "") or ref.get("repair_hint", "") or "")
                key_parts = [
                    layer or "none",
                    reference or "none",
                    problem_area or "unknown_or_mixed",
                    self._material_signature_slug(material_signature or [specific_problem]),
                ]
                cluster_key_version = "material_signature_v1"
            else:
                mechanism_case = dict(ref.get("failure_mechanism", {}) or {})
                primary = dict(mechanism_case.get("primary_failure", {}) or {})
                repair_target = dict(ref.get("repair_target", {}) or {})
                repair_need = str(repair_target.get("repair_need", "") or "")
                problem_area = str(primary.get("stage", "") or ref.get("failed_stage", "") or "unknown_or_mixed")
                specific_problem = str(primary.get("specific_problem", "") or ref.get("observed_failure", "") or ref.get("failure_type", "") or "")
                repair_hint = repair_need or str(ref.get("why_current_tree_insufficient", "") or "")
                material_signature = [
                    f"failure_type={str(ref.get('failure_type', '') or '')}",
                    f"failed_stage={problem_area}",
                ]
                if primary:
                    material_signature.append(f"mechanism={str(primary.get('mechanism', '') or '')}")
                if repair_need:
                    material_signature.append(f"repair_need={self._slug_fragment(repair_need, 'repair-need')}")
                if (
                    str(ref.get("failure_type", "") or "") == "unmapped_followup_task"
                    and str(ref.get("failed_stage", "") or "") == "task_mapping"
                    and layer == "what"
                ):
                    material_signature.append(f"trigger_signature={str(ref.get('trigger_signature', '') or 'unknown_signature')}")
                key_parts = [
                    layer or "none",
                    reference or "none",
                    problem_area or "unknown_or_mixed",
                    self._material_signature_slug(material_signature),
                ]
                cluster_key_version = "legacy_material_fallback_v1"
            key = "/".join(key_parts)
            cluster = clusters.setdefault(
                key,
                {
                    "cluster_id": key,
                    "cluster_key_version": cluster_key_version,
                    "repair_location": {"layer": layer, "reference_node": reference},
                    "problem_area": problem_area,
                    "specific_problem": self._short_text(specific_problem, 420),
                    "material_signature": material_signature,
                    "repair_hint": self._short_text(repair_hint, 260),
                    "session_outcome": str(ref.get("session_outcome", "") or ""),
                    "failure_type": str(ref.get("failure_type", "") or ""),
                    "failed_stage": str(ref.get("failed_stage", "") or ""),
                    "primary_failure_stage": problem_area,
                    "primary_failure_mechanism": "",
                    "layer_hint": layer,
                    "reference_node": reference,
                    "repair_need": self._short_text(repair_hint, 260),
                    "support": 0,
                    "unique_users": set(),
                    "support_user_ids": set(),
                    "support_user_paths": {},
                    "outcome_distribution": Counter(),
                    "round_effects": Counter(),
                    "quality_issues": Counter(),
                    "mechanism_distribution": Counter(),
                    "repair_need_distribution": Counter(),
                    "not_the_problem_distribution": Counter(),
                    "interaction_quality_distribution": Counter(),
                    "ruled_out_distribution": Counter(),
                    "evidence_counter_totals": Counter(),
                    "legacy_failure_type_distribution": Counter(),
                    "diagnostic_tag_distribution": Counter(),
                    "trigger_signature_distribution": Counter(),
                    "route_selection_patterns": Counter(),
                    "advisor_execution_patterns": Counter(),
                    "public_tree_need_anchor": anchor,
                    "material_cases": [],
                    "causal_cases": [],
                    "representative_cases": [],
                    "representative_reflections": [],
                    "ready": False,
                    "severity": 0.0,
                    "evidence_strength": "single_case",
                    "design_constraint": "generate a narrow train-only sprout; do not overgeneralize beyond the observed fact pattern",
                },
            )
            cluster["support"] += 1
            if ref.get("user_id"):
                user_id = str(ref.get("user_id"))
                cluster["unique_users"].add(user_id)
                cluster["support_user_ids"].add(user_id)
                if user_id not in cluster["support_user_paths"]:
                    cluster["support_user_paths"][user_id] = dict(ref.get("path", {}) or {})
            cluster["outcome_distribution"][str(ref.get("session_outcome", "") or "")] += 1
            cluster["round_effects"][str(ref.get("round_effect", "") or "")] += 1
            legacy_failure_type = str(ref.get("failure_type", "") or "")
            if legacy_failure_type:
                cluster["legacy_failure_type_distribution"][legacy_failure_type] += 1
                cluster["diagnostic_tag_distribution"][legacy_failure_type] += 1
            for issue in list(ref.get("quality_issues", []) or []):
                if str(issue or "").strip():
                    cluster["quality_issues"][str(issue).strip()] += 1
                    cluster["diagnostic_tag_distribution"][str(issue).strip()] += 1
            if repair_hint:
                cluster["repair_need_distribution"][repair_hint] += 1
            if anchor and not cluster.get("public_tree_need_anchor"):
                cluster["public_tree_need_anchor"] = anchor
            if material_case:
                evidence = dict(material_case.get("evidence", {}) or {})
                for name, value in evidence.items():
                    try:
                        cluster["evidence_counter_totals"][str(name)] += int(value or 0)
                    except Exception:
                        continue
                for ruled in list(material_case.get("not_the_problem", []) or []):
                    if str(ruled or "").strip():
                        cluster["not_the_problem_distribution"][str(ruled).strip()] += 1
                        cluster["ruled_out_distribution"][str(ruled).strip()] += 1
                if len(cluster["material_cases"]) < 8:
                    cluster["material_cases"].append(material_case)
                interaction_quality = dict(material_case.get("interaction_quality", {}) or {})
                failure_mode = str(interaction_quality.get("failure_mode", "") or "")
                if failure_mode:
                    cluster["interaction_quality_distribution"][f"failure_mode={failure_mode}"] += 1
                for issue in list(interaction_quality.get("quality_issues", []) or [])[:4]:
                    if str(issue or "").strip():
                        cluster["interaction_quality_distribution"][f"quality_issue={str(issue).strip()}"] += 1
                if len(cluster["representative_cases"]) < 4:
                    cluster["representative_cases"].append(
                        {
                            "case_id": str(material_case.get("case_id", "") or ""),
                            "actual_behavior": self._short_text(material_case.get("actual_behavior", ""), 260),
                            "expected_behavior": self._short_text(material_case.get("expected_behavior", ""), 260),
                            "specific_problem": self._short_text(material_case.get("specific_problem", ""), 260),
                            "evidence": [
                                f"{name}={value}"
                                for name, value in evidence.items()
                                if isinstance(value, int) and int(value) != 0
                            ][:8],
                            "interaction_quality": interaction_quality,
                        }
                    )
            else:
                mechanism_case = dict(ref.get("failure_mechanism", {}) or {})
                primary = dict(mechanism_case.get("primary_failure", {}) or {})
                mechanism = str(primary.get("mechanism", "") or "")
                if mechanism:
                    cluster["mechanism_distribution"][mechanism] += 1
                for evidence in list(primary.get("evidence", []) or []):
                    text = str(evidence or "")
                    m = re.match(r"^([A-Za-z0-9_ -]+)=(-?\d+)$", text.strip())
                    if m:
                        cluster["evidence_counter_totals"][m.group(1).strip()] += int(m.group(2))
                for ruled in list(mechanism_case.get("ruled_out", []) or ref.get("ruled_out", []) or []):
                    if str(ruled or "").strip():
                        cluster["ruled_out_distribution"][str(ruled).strip()] += 1
                failure_case = dict(ref.get("failure_case", {}) or {})
                if failure_case and len(cluster["causal_cases"]) < 8:
                    cluster["causal_cases"].append(failure_case)
            trigger = str(ref.get("trigger_signature", "") or "")
            if trigger:
                cluster["trigger_signature_distribution"][trigger] += 1
            route_context = dict(ref.get("route_selection_context", {}) or {})
            selected_nodes = dict(route_context.get("selected_nodes", {}) or {})
            route_pattern = "|".join(
                [
                    str(route_context.get("trigger_signature", "") or trigger or ""),
                    str(selected_nodes.get("what", "") or ""),
                    str(selected_nodes.get("how", "") or ""),
                    str(selected_nodes.get("who", "") or ""),
                ]
            ).strip("|")
            if route_pattern:
                cluster["route_selection_patterns"][route_pattern] += 1
            advisor_context = dict(ref.get("advisor_execution_context", {}) or {})
            advisor_pattern = "|".join(
                [
                    str(advisor_context.get("candidate_coverage", "") or ""),
                    str(advisor_context.get("advisor_diversity", "") or ""),
                    str(advisor_context.get("evidence_specificity", "") or ""),
                    "protocol_ok" if bool(advisor_context.get("protocol_enforced", False)) else "protocol_gap",
                ]
            ).strip("|")
            if advisor_pattern:
                cluster["advisor_execution_patterns"][advisor_pattern] += 1
            if len(cluster["representative_reflections"]) < 4:
                cluster["representative_reflections"].append(ref)
        out = []
        for cluster in clusters.values():
            unique_users = len(cluster.get("unique_users", set()) or set())
            support = int(cluster.get("support", 0) or 0)
            effects = Counter(cluster.get("round_effects", {}) or {})
            harmful = int(effects.get("harmful", 0) or 0)
            ineffective = int(effects.get("ineffective", 0) or 0)
            uncertain = int(effects.get("uncertain_failure", 0) or 0)
            severity = harmful * 4.0 + ineffective * 2.5 + uncertain * 1.0 + support * 0.2
            cluster["unique_users"] = unique_users
            cluster["support_user_ids"] = sorted(str(x) for x in list(cluster.get("support_user_ids", set()) or []) if str(x).strip())
            cluster["support_user_paths"] = {
                str(k): dict(v or {})
                for k, v in dict(cluster.get("support_user_paths", {}) or {}).items()
                if str(k).strip()
            }
            cluster["outcome_distribution"] = dict(cluster.get("outcome_distribution", {}) or {})
            cluster["round_effects"] = dict(effects)
            cluster["quality_issues"] = dict(cluster.get("quality_issues", {}) or {})
            cluster["mechanism_distribution"] = dict(cluster.get("mechanism_distribution", {}) or {})
            cluster["repair_need_distribution"] = dict(cluster.get("repair_need_distribution", {}) or {})
            cluster["not_the_problem_distribution"] = dict(cluster.get("not_the_problem_distribution", {}) or {})
            cluster["interaction_quality_distribution"] = dict(cluster.get("interaction_quality_distribution", {}) or {})
            cluster["ruled_out_distribution"] = dict(cluster.get("ruled_out_distribution", {}) or {})
            cluster["evidence_counter_totals"] = dict(cluster.get("evidence_counter_totals", {}) or {})
            cluster["legacy_failure_type_distribution"] = dict(cluster.get("legacy_failure_type_distribution", {}) or {})
            cluster["diagnostic_tag_distribution"] = dict(cluster.get("diagnostic_tag_distribution", {}) or {})
            cluster["trigger_signature_distribution"] = dict(cluster.get("trigger_signature_distribution", {}) or {})
            cluster["route_selection_patterns"] = dict(cluster.get("route_selection_patterns", {}) or {})
            cluster["advisor_execution_patterns"] = dict(cluster.get("advisor_execution_patterns", {}) or {})
            cluster["ready"] = bool(support >= 1)
            cluster["evidence_strength"] = "multi_case" if support >= 2 else "single_case"
            cluster["design_constraint"] = (
                "generate a reusable active public-tree node for this repeated fact pattern, then inject it only into the supporting users' route skills"
                if support >= 2
                else "generate a narrow active public-tree node for this observed fact pattern, then inject it only into the supporting user's route skill"
            )
            cluster["severity"] = severity
            out.append(cluster)
        out.sort(key=lambda row: (not bool(row.get("ready", False)), -float(row.get("severity", 0.0) or 0.0), str(row.get("cluster_id", ""))))
        return out

    def _node_exists(self, engine, layer, node_id):
        if not layer or not node_id:
            return False
        tree = engine.public_tree_store.load_tree(force_reload=True)
        return str(node_id) in (tree.get(str(layer), {}) or {})

    @staticmethod
    def _slug_fragment(text, fallback="repair"):
        tokens = re.findall(r"[a-z0-9]+", str(text or "").lower().replace("_", "-"))
        slug = "-".join(tokens[:5]).strip("-")
        return slug or fallback

    def _material_signature_slug(self, signature):
        text = " ".join(str(x or "") for x in list(signature or []))
        text = (
            text.lower()
            .replace(">=", " gte ")
            .replace("<=", " lte ")
            .replace(">", " gt ")
            .replace("<", " lt ")
            .replace("=", " eq ")
            .replace("/", " slash ")
            .replace("_", "-")
        )
        tokens = re.findall(r"[a-z0-9]+", text)
        return "-".join(tokens[:32]).strip("-") or "observed-fact-pattern"

    @staticmethod
    def _node_depth(node_id):
        return len([part for part in str(node_id or "").strip("/").split("/") if part])

    @staticmethod
    def _infer_who_retrieval_constraints(node_id):
        parts = [part for part in str(node_id or "").strip("/").split("/") if part]
        tokens = set()
        for part in parts:
            tokens.update(tok for tok in re.split(r"[^a-z0-9]+", part.lower()) if tok)
        constraints = {}
        if "mutual" in tokens:
            constraints["trust_relation"] = "mutual-trust"
        elif "one" in tokens and "way" in tokens:
            constraints["trust_relation"] = "one-way-trust"
        if "multi" in tokens and "trust" in tokens:
            constraints["trust_scope"] = "multi-trust"
        elif "single" in tokens and "trust" in tokens:
            constraints["trust_scope"] = "single-trust"
        if "dissimilar" in tokens or "diverse" in tokens:
            constraints["history_similarity"] = "dissimilar"
        elif "similar" in tokens or "nearest" in tokens or "neighbor" in tokens:
            constraints["history_similarity"] = "similar"
        if "item" in tokens and "experienced" in tokens:
            constraints["requires_item_experience"] = True
        if "two" in tokens and "hop" in tokens:
            constraints["hop"] = 2
        return constraints

    @staticmethod
    def _infer_trial_anchor_node(layer, node_id, node=None, fallback_reference=""):
        layer = str(layer or "").strip()
        node_id = str(node_id or "").strip().strip("/")
        node = dict(node or {}) if isinstance(node, dict) else {}
        explicit = str(node.get("trial_anchor_node", "") or fallback_reference or "").strip().strip("/")
        if explicit and explicit != node_id and "/" not in explicit:
            return explicit
        text = " ".join(
            [
                node_id,
                str(node.get("description", "") or ""),
                str(node.get("use_why", "") or ""),
                str(node.get("if_selected", "") or ""),
                str(node.get("evidence_pattern", "") or ""),
            ]
        ).lower()
        tokens = set(tok for tok in re.split(r"[^a-z0-9]+", text) if tok)
        if layer == "what":
            if {"candidate", "contrast"} & tokens or "evidence" in tokens or "compare" in tokens:
                return "compare_remaining_candidates"
            if "reduce" in tokens or "hesitation" in tokens:
                return "reduce_hesitation_set"
        if layer == "how":
            if {"opposition", "oppose", "rebuttal", "rebut", "counterargument", "competitive", "challenge"} & tokens:
                return "multi-competitive"
            if {"cooperative", "coverage", "consensus", "support"} & tokens:
                return "multi-cooperative"
        if layer == 'why':
            if "candidate" in tokens and "conflict" in tokens:
                return "candidate-conflict"
            if ("prior" in tokens or "internal" in tokens) and "conflict" in tokens:
                return "internal-prior-conflict"
        return ""

    def _unique_new_node_id(self, engine, layer, parent, base):
        base = self._slug_fragment(base, fallback="failure-repair")
        tree_nodes = engine.public_tree_store.load_tree(force_reload=True).get(str(layer), {}) or {}
        for idx in range(1, 20):
            node_id = base if idx == 1 else f"{base}-{idx}"
            full = f"{parent}/{node_id}" if parent else node_id
            if full not in tree_nodes:
                return node_id
        return f"{base}-{int(time.time())}"

    def _decide_patch_frame_for_cluster(self, engine, cluster):
        layer = str((cluster or {}).get("layer_hint", "") or "how")
        if layer not in {"what", "how", "who"}:
            layer = "what"
        reference = self._infer_patch_parent_node(engine, layer, cluster)
        operation = "add_child_node" if self._node_exists(engine, layer, reference) else "add_sibling_node"
        parent = reference if operation == "add_child_node" else ""
        base = (
            str(cluster.get("repair_need", "") or "")
            or str(cluster.get("primary_failure_mechanism", "") or "")
            or f"{cluster.get('failure_type', 'failure')}-{cluster.get('failed_stage', 'repair')}"
        )
        return {
            "operation": operation,
            "layer": layer,
            "parent_node": parent,
            "new_node_id": self._unique_new_node_id(engine, layer, parent, base),
            "status": "active",
        }

    def _semantic_fallback_node_id(self, cluster, strategy=""):
        failure_type = str((cluster or {}).get("failure_type", "") or "")
        failed_stage = str((cluster or {}).get("failed_stage", "") or "")
        mechanism = str((cluster or {}).get("primary_failure_mechanism", "") or "")
        repair_need = str((cluster or {}).get("repair_need", "") or "")
        strategy = str(strategy or "")
        if repair_need:
            return self._slug_fragment(repair_need, fallback="targeted-repair")
        if mechanism in {"support_only_convergence", "no_counterargument"}:
            return "mandatory-counterargument"
        if mechanism == "advisor_source_overlap":
            return "source-diverse-advisors"
        if mechanism == "task_does_not_require_candidate_contrast":
            return "contrastive-task-grounding"
        if mechanism in {"advisor_not_answering_task", "missing_candidate_view"}:
            return "task-grounded-candidate-view"
        if "replacement" in strategy:
            if failure_type in {"advisor_homogeneity", "protocol_not_enforced", "multi_candidate_protocol_not_enforced"}:
                return "evidence-grounded-opposition"
            if failure_type in {"advisor_not_answering", "feedback_not_inherited", "evidence_lost"}:
                return "task-grounded-recovery"
            return "alternative-repair-protocol"
        if failure_type in {"advisor_homogeneity", "protocol_not_enforced", "multi_candidate_protocol_not_enforced"}:
            return "mandatory-counterargument"
        if failed_stage == "advisor_interaction":
            return "explicit-interaction-check"
        if failed_stage == "aggregation":
            return "summary-evidence-preservation"
        return "targeted-repair-check"

    def _candidate_patch_strategies_for_cluster(self, engine, cluster, frame):
        cluster = dict(cluster or {})
        frame = dict(frame or {})
        layer = str(frame.get("layer", "") or cluster.get("layer_hint", "") or "how")
        reference = str(frame.get("parent_node", "") or cluster.get("reference_node", "") or "")
        tree_nodes = engine.public_tree_store.load_tree(force_reload=True).get(layer, {}) or {}
        if reference not in tree_nodes:
            reference = self._infer_patch_parent_node(engine, layer, cluster)
        strategies = []
        if layer != "who":
            trial_anchor = self._infer_trial_anchor_node(layer, str(frame.get("new_node_id", "") or ""), fallback_reference=reference)
            strategies.append(
                {
                    "strategy": "create_replacement_parent",
                    "operation": "add_sibling_node",
                    "layer": layer,
                    "parent_node": "",
                    "status": "active",
                    "why_to_choose": (
                        "Default choice for public_tree_need: choose this when repeated failures under the reference node suggest "
                        "the tree needs a distinct same-layer route/capability instead of another deeper child."
                    ),
                    "new_node_id_guidance": "Generate a short semantic top-level strategy name that describes the new capability, not the observed failure tag.",
                    "reference_node_being_replaced_or_bypassed": reference,
                    "requires_trial_anchor_node": True,
                    "trial_anchor_node_guidance": (
                        f"Set trial_anchor_node to an existing same-layer active/root node. Usually use {reference or trial_anchor or 'the closest same-layer competitor'}; "
                        "this controls train-time insertion after the anchor and temporary downstream route inheritance."
                    ),
                    "suggested_trial_anchor_node": trial_anchor or reference,
                }
            )
        if reference and reference in tree_nodes and self._node_depth(reference) < 3:
            child_guidance = "Generate a short semantic capability name such as mandatory-counterargument or evidence-grounded-opposition; do not reuse failure labels."
            if layer == "who":
                child_guidance = (
                    "Generate an advisor subgroup name, not a policy name. Good examples: mutual-trust, one-way-trust, "
                    "multi-trust, single-trust, history-similar, history-dissimilar, item-experienced, two-hop-social. "
                    "Do not use abstract policy names such as distinct-perspective, high-reliability, disagreement-seeking, or evidence-rich."
                )
            strategies.append(
                {
                    "strategy": "refine_existing_node",
                    "operation": "add_child_node",
                    "layer": layer,
                    "parent_node": reference,
                    "status": "active",
                    "why_to_choose": (
                        "Exception choice: choose this only when CommonCauseAnalysis explicitly says the reference node is the right abstraction "
                        "and the missing capability is a narrow specialization inside that parent. Do not choose merely because the parent exists."
                    ),
                    "new_node_id_guidance": child_guidance,
                }
            )
        return {
            "layer": layer,
            "reference_node": reference,
            "max_node_depth": 3,
            "strategy_preference": (
                "Prefer create_replacement_parent/add_sibling_node for what/how public_tree_need clusters. "
                "Use refine_existing_node/add_child_node only for a clearly narrow specialization of an otherwise correct parent."
            ),
            "strategies": strategies,
        }

    def _infer_patch_parent_node(self, engine, layer, cluster):
        layer = str(layer or "")
        cluster = dict(cluster or {})
        tree_nodes = (engine.public_tree_store.load_tree(force_reload=True).get(layer, {}) or {}) if layer else {}
        candidates = []
        reference = str(cluster.get("reference_node", "") or "")
        if reference and reference not in ["none", "skip"]:
            candidates.append(reference)
        for ref in list(cluster.get("representative_reflections", []) or []):
            if not isinstance(ref, dict):
                continue
            change = dict(ref.get("needed_tree_change", {}) or {})
            row_ref = str(change.get("reference_node", "") or "")
            if row_ref and row_ref not in ["none", "skip"]:
                candidates.append(row_ref)
            path = dict(ref.get("path", {}) or {})
            path_ref = self._reference_node_for_layer(layer, path)
            if path_ref and path_ref not in ["none", "skip"]:
                candidates.append(path_ref)
            route_ctx = dict(ref.get("route_selection_context", {}) or {})
            selected_nodes = dict(route_ctx.get("selected_nodes", {}) or route_ctx.get("selected_action", {}) or {})
            ctx_ref = str(selected_nodes.get(layer, "") or "")
            if ctx_ref and ctx_ref not in ["none", "skip"]:
                candidates.append(ctx_ref)
        if layer == "how" and str(cluster.get("failure_type", "") or "") in [
            "advisor_homogeneity",
            "protocol_not_enforced",
            "multi_candidate_protocol_not_enforced",
            "misled_user",
        ]:
            candidates.append("multi-competitive")
        if layer == "what" and str(cluster.get("failure_type", "") or "") in [
            "advisor_not_answering",
            "feedback_not_inherited",
            "evidence_lost",
            "failed_to_recover",
        ]:
            candidates.append("compare_remaining_candidates")
        for candidate in candidates:
            candidate = str(candidate or "").strip()
            if candidate in tree_nodes:
                return candidate
            while "/" in candidate:
                candidate = self._parent_node_id(candidate)
                if candidate in tree_nodes:
                    return candidate
        return reference if reference in tree_nodes else ""

    @staticmethod
    def _cluster_failure_signature(cluster):
        cluster = dict(cluster or {})
        issues = cluster.get("quality_issues", {}) or {}
        if isinstance(issues, dict):
            issue_text = " ".join(sorted(str(k) for k, v in issues.items() if int(v or 0) > 0))
        else:
            issue_text = " ".join(sorted(str(x) for x in list(issues or [])))
        evidence = " ".join(
            [
                str(cluster.get("failure_type", "") or ""),
                str(cluster.get("failed_stage", "") or ""),
                issue_text,
            ]
        ).lower()
        tokens = re.findall(r"[a-z0-9]+", evidence)
        return " ".join(tokens[:10])

    def _find_existing_sprout_for_cluster(self, engine, cluster, frame):
        layer = str(frame.get("layer", "") or cluster.get("layer_hint", "") or "")
        parent = str(frame.get("parent_node", "") or cluster.get("reference_node", "") or "")
        signature = self._cluster_failure_signature(cluster)
        failure_type = str(cluster.get("failure_type", "") or "").lower()
        failed_stage = str(cluster.get("failed_stage", "") or "").lower()
        tree = engine.public_tree_store.load_tree(force_reload=True)
        candidates = []
        for node_id, node in sorted(dict((tree.get(layer, {}) or {}) if layer else {}).items()):
            node = dict(node or {})
            status = str(node.get("status", "") or "")
            if status != "sprout":
                continue
            node_parent = self._parent_node_id(node_id)
            if parent and not (node_id.startswith(parent + "/") or node_parent == parent):
                continue
            node_text = " ".join(
                [
                    node_id,
                    str(node.get("description", "") or ""),
                    str(node.get("evidence_pattern", "") or ""),
                    str(node.get("use_why", "") or ""),
                    self._short_text(node.get("skill_body", ""), 800),
                    json.dumps(node.get("applicability_condition", {}) or {}, ensure_ascii=False, default=str),
                    json.dumps(node.get("execution_hint", {}) or {}, ensure_ascii=False, default=str),
                ]
            ).lower()
            score = 0
            if failure_type and failure_type.replace("_", "-") in node_text.replace("_", "-"):
                score += 2
            if failed_stage and failed_stage.replace("_", "-") in node_text.replace("_", "-"):
                score += 2
            for token in signature.split():
                if len(token) >= 5 and token in node_text:
                    score += 1
            if score >= 3:
                candidates.append((score, node_id, node))
        if not candidates:
            return None
        candidates.sort(key=lambda row: (-row[0], row[1]))
        return {"node_id": candidates[0][1], "node": candidates[0][2], "match_score": candidates[0][0]}

    def _cluster_reuse_fingerprint(self, cluster):
        cluster = dict(cluster or {})
        anchor = dict(cluster.get("public_tree_need_anchor", {}) or cluster.get("cluster_anchor", {}) or {})
        if anchor:
            level = str(anchor.get("level", "") or cluster.get("layer_hint", "") or "")
            reference = str(anchor.get("reference_node", "") or cluster.get("reference_node", "") or "")
            issue_family = str(anchor.get("issue_family", "") or cluster.get("problem_area", "") or "")
            missing = str(anchor.get("missing_capability", "") or cluster.get("repair_need", "") or cluster.get("repair_hint", "") or "")
            return {
                "cluster_key_version": "public_tree_need_anchor_v1",
                "layer": level,
                "reference_node": reference,
                "problem_area": issue_family,
                "material_signature_slug": self._material_signature_slug([
                    f"issue_family={issue_family}",
                    f"suggested_node_hint={str(anchor.get('suggested_node_hint', '') or '')}",
                    f"missing_capability={self._slug_fragment(missing, 'missing-capability')}",
                    f"rule_source={str(anchor.get('rule_source', '') or '')}",
                ]),
                "repair_need_slug": self._slug_fragment(missing, "repair-need"),
                "failure_type": str(anchor.get("suggested_node_hint", "") or cluster.get("failure_type", "") or ""),
                "failed_stage": {
                    "what": "task_mapping",
                    "who": "advisor_selection",
                    "how": "advisor_interaction",
                }.get(level, str(cluster.get("failed_stage", "") or "")),
            }
        material_signature = [
            str(x)
            for x in list(cluster.get("material_signature", []) or [])
            if str(x).strip()
        ]
        repair_need = str(cluster.get("repair_need", "") or cluster.get("repair_hint", "") or "")
        return {
            "cluster_key_version": str(cluster.get("cluster_key_version", "") or ""),
            "layer": str(cluster.get("layer_hint", "") or ""),
            "reference_node": str(cluster.get("reference_node", "") or ""),
            "problem_area": str(cluster.get("problem_area", "") or cluster.get("primary_failure_stage", "") or ""),
            "material_signature_slug": self._material_signature_slug(material_signature),
            "repair_need_slug": self._slug_fragment(repair_need, "repair-need"),
            "failure_type": str(cluster.get("failure_type", "") or ""),
            "failed_stage": str(cluster.get("failed_stage", "") or ""),
        }

    @staticmethod
    def _fingerprint_match_score(left, right):
        left = dict(left or {})
        right = dict(right or {})
        score = 0
        for key, weight in [
            ("layer", 3),
            ("reference_node", 3),
            ("problem_area", 2),
            ("material_signature_slug", 4),
            ("repair_need_slug", 3),
            ("failure_type", 1),
            ("failed_stage", 1),
        ]:
            lv = str(left.get(key, "") or "")
            rv = str(right.get(key, "") or "")
            if lv and rv and lv == rv:
                score += weight
        return score

    def _find_reusable_active_node_for_cluster(self, engine, cluster):
        cluster = dict(cluster or {})
        layer = str(cluster.get("layer_hint", "") or "")
        reference = str(cluster.get("reference_node", "") or "")
        if layer not in {"what", "who", "how"}:
            return None
        fingerprint = self._cluster_reuse_fingerprint(cluster)
        signature = self._cluster_failure_signature(cluster)
        tree = engine.public_tree_store.load_tree(force_reload=True)
        candidates = []
        for node_id, node in sorted(dict((tree.get(layer, {}) or {})).items()):
            node = dict(node or {})
            if str(node.get("status", "") or "") != "active":
                continue
            if not bool(node.get("route_injection_only", False)):
                continue
            node_parent = self._parent_node_id(node_id)
            if reference and not (node_id.startswith(reference + "/") or node_parent == reference or str(node.get("trial_anchor_node", "") or "") == reference):
                continue
            reuse = dict(node.get("evolution_reuse", {}) or {})
            score = self._fingerprint_match_score(fingerprint, reuse)
            node_text = " ".join(
                [
                    node_id,
                    str(node.get("description", "") or ""),
                    str(node.get("use_why", "") or ""),
                    str(node.get("if_selected", "") or ""),
                    str(node.get("evidence_pattern", "") or ""),
                    str(node.get("repair_need", "") or ""),
                    self._short_text(node.get("skill_body", ""), 800),
                    json.dumps(node.get("applicability_condition", {}) or {}, ensure_ascii=False, default=str),
                    json.dumps(node.get("execution_hint", {}) or {}, ensure_ascii=False, default=str),
                ]
            ).lower()
            if not score:
                for token in signature.split():
                    if len(token) >= 5 and token in node_text:
                        score += 1
            if score >= 7:
                candidates.append((score, node_id, node, reuse))
        if not candidates:
            return None
        candidates.sort(key=lambda row: (-row[0], row[1]))
        score, node_id, node, reuse = candidates[0]
        return {
            "node_id": node_id,
            "node": node,
            "match_score": int(score),
            "fingerprint": fingerprint,
            "node_reuse": reuse,
        }

    def _append_sprout_revision_memory(self, engine, layer, node_id, cluster, reason):
        node_dir = engine.public_tree_store._node_dir_for_id(layer, node_id)
        lifecycle_path = node_dir / "references" / "lifecycle.json"
        lifecycle = load_json(lifecycle_path, default={}) or {}
        lifecycle.setdefault("status", "sprout")
        lifecycle.setdefault("parent_node", self._parent_node_id(node_id))
        memory = list(lifecycle.get("revision_memory", []) or [])
        row = {
            "ts": int(time.time()),
            "reason": str(reason or ""),
            "cluster_id": str(cluster.get("cluster_id", "") or ""),
            "failure_type": str(cluster.get("failure_type", "") or ""),
            "failed_stage": str(cluster.get("failed_stage", "") or ""),
            "quality_issues": dict(cluster.get("quality_issues", {}) or {}),
            "support": int(cluster.get("support", 0) or 0),
        }
        memory.append(row)
        lifecycle["revision_memory"] = memory[-30:]
        dump_json(lifecycle_path, lifecycle)
        append_jsonl(
            engine.public_tree_store.index_dir / "tree_patch_revision_memory.jsonl",
            {
                "event": "duplicate_cluster_reused_existing_sprout",
                "ts": int(time.time()),
                "layer": str(layer or ""),
                "node_id": str(node_id or ""),
                "cluster_id": str(cluster.get("cluster_id", "") or ""),
                "reason": str(reason or ""),
            },
        )

    def _skill_md_for_failure_patch(self, frame, cluster, variant=1):
        layer = str(frame.get("layer", "") or "how")
        node_id = str(frame.get("new_node_id", "") or "failure-repair")
        description = f"Repair repeated {cluster.get('failure_type', 'communication failure')} failures in {cluster.get('failed_stage', 'communication')}."
        use_why = (
            f"Use when train failures show {cluster.get('failure_type', 'communication failure')} "
            f"under {cluster.get('reference_node', 'the current node')} and the current node only partially controls the behavior."
        )
        runtime = "This node is active in the public tree but should be prioritized only for users whose communication_route_skill explicitly includes it, unless global_default is true."
        if layer == "what":
            return (
                f"---\nname: {node_id}\ndescription: {description}\nlevel: what\nstatus: active\n---\n\n"
                f"# {node_id}\n\n## Use When\n{use_why}\n\n## If Selected\nUse this task type to repair the repeated failure without changing who or how.\n\n"
                "## User Task Type\nInterpret the user's natural-language communication task as a failure-repair task over the HesitationSet.\n\n"
                "### Goal\nMake advisors answer the missing or harmful evidence pattern that caused previous failed sessions.\n\n"
                "### Required Actions\n- Use UserTask as the content goal.\n- Keep the answer inside the HesitationSet.\n- Do not select who or how.\n- Preserve unresolved details as AskUser or StillMissing.\n\n"
                "### Advisor Output Format For This Task\n- CandidateView:\n- <exact candidate> | <repair-support/repair-risk/unclear> | <short reason>\n- TaskAnswer: <direct answer to Task>\n- AskUser: <specific question for the user, or none>\n\n"
                "## Skill JSON Requirements\nskill_json.task_output_format must list every advisor output field above. "
                "skill_json.summary_hints.important_output_fields must include CandidateView, TaskAnswer, AskUser, and any repair-specific fields so the summary agent preserves them.\n\n"
                f"## Runtime Status\n{runtime}"
            )
        if layer == "how":
            return (
                f"---\nname: {node_id}\ndescription: {description}\nlevel: how\nstatus: active\n---\n\n"
                f"# {node_id}\n\n## Use When\n{use_why}\n\n## If Selected\nUse this communication organization mode to prevent the repeated failure pattern.\n\n"
                "## Advisor Communication Skill\nAdvisors must first answer the selected UserTask, then explicitly check whether their statement avoids the observed failure pattern.\n\n"
                "### Goal\nOrganize advisors so the group does not repeat the failed behavior seen in TW/WW sessions.\n\n"
                "### Required Actions\n- UserTask defines what content to answer.\n- This node only defines how advisors organize.\n- Keep continuity with previous discussion memory when available.\n- If previous advisors missed the task, directly repair that gap instead of repeating their claim.\n\n"
                "### Output Contract\n- Return task-specific fields from the selected what node.\n- Add only the how-specific interaction field below.\n\n"
                "### Advisor Output Format\n- FailureRepairCheck: <how this response avoids the repeated failure pattern>\n\n"
                f"## Runtime Status\n{runtime}"
            )
        if layer == "who":
            return (
                f"---\nname: {node_id}\ndescription: {description}\nlevel: who\nstatus: active\n---\n\n"
                f"# {node_id}\n\n## Use When\n{use_why}\n\n## If Selected\nSelect advisors matching this concrete subgroup within the parent advisor source.\n\n"
                "## Advisor Role Label\nadvisor subgroup\n\n## Retrieval Policy\nUse the node id and skill_json retrieval_constraints to filter or rank the parent advisor source. This node must describe an advisor subgroup, not an abstract selection policy.\n\n"
                "## Do Not Use When\nDo not use when the failure is caused by task wording, advisor interaction protocol, or aggregation rather than advisor subgroup choice.\n\n"
                f"## Runtime Status\n{runtime}"
            )
        return (
            f"---\nname: {node_id}\ndescription: {description}\nlevel: when\nstatus: active\n---\n\n"
            f"# {node_id}\n\n## Use When\n{use_why}\n\n## If Selected\nTrigger communication because the current state matches a repeated failed-session trigger.\n\n"
            "## Communication Trigger\nThis node only decides whether communication is needed. It does not select what, who, or how.\n\n"
            "### Goal\nProtect against a repeated failure condition before communication starts.\n\n"
            "### Required Actions\n- Detect the trigger from current planning state.\n- Do not force communication if the trigger is absent.\n- Pass the trigger reason to task planning.\n\n"
            f"## Runtime Status\n{runtime}"
        )

    def _deterministic_failure_patch(self, engine, cluster, frame=None, variant=1):
        frame = dict(frame or self._decide_patch_frame_for_cluster(engine, cluster))
        layer = str(frame.get("layer", "") or "how")
        node_id = str(frame.get("new_node_id", "") or "failure-repair")
        description = f"Repair repeated {cluster.get('failure_type', 'communication failure')} failures."
        patch = {
            "operation": str(frame.get("operation", "add_child_node") or "add_child_node"),
            "layer": layer,
            "parent_node": str(frame.get("parent_node", "") or ""),
            "new_node_id": node_id,
            "status": "active",
            "why_needed": self._short_text(cluster.get("cluster_id", ""), 320),
            "evidence_pattern": self._short_text(
                f"{cluster.get('session_outcome')} {cluster.get('failure_type')} at {cluster.get('failed_stage')}; support={cluster.get('support')}",
                360,
            ),
            "skill_md": self._skill_md_for_failure_patch(frame, cluster, variant=variant),
            "skill_json": {
                "node_id": node_id,
                "node_level": layer,
                "status": "active",
                "dataset": self._dataset_slug(engine),
                "source_dataset": self._dataset_slug(engine),
                "dataset_scope": [self._dataset_slug(engine)],
                "global_default": False,
                "route_injection_only": True,
                "description": description,
                "use_why": f"Repeated failed TW/WW rounds show {cluster.get('failure_type')} under {cluster.get('reference_node')}.",
                "if_selected": "Apply this failure-repair behavior when this node is explicitly present in the user's communication_route_skill.",
                "advisor_role": "failure-aware advisor" if layer in ["who", "how"] else "",
                "advisor_output_format": ["FailureRepairCheck: <how the response avoids the repeated failure pattern>"] if layer == "how" else [],
                "task_output_format": ["CandidateView:", "TaskAnswer:", "AskUser:"] if layer == "what" else [],
                "selection_prior": 0.7,
                "applicability_condition": {
                    "failure_type": str(cluster.get("failure_type", "") or ""),
                    "failed_stage": str(cluster.get("failed_stage", "") or ""),
                },
                "execution_hint": {"source": "failure_cluster_v2"},
                "evolution_reuse": {
                    **self._cluster_reuse_fingerprint(cluster),
                    "source_cluster_id": str(cluster.get("cluster_id", "") or ""),
                },
                "evolution_state": {"tt": 0, "wt": 0, "tw": 0, "ww": 0},
            },
            "do_not_use_why": "Do not prioritize for users whose communication_route_skill does not explicitly include this node.",
            "expected_benefit": "Reduce repeated TW/WW communication failures of the same abstract type.",
            "risk": "May overfit if it is injected beyond the failed-user cluster.",
        }
        if layer in {"what", "how"}:
            skill_json = dict(patch.get("skill_json", {}) or {})
            skill_json["summary_hints"] = self._normalized_summary_hints_for_patch(
                layer,
                skill_json,
                patch.get("skill_md", ""),
            )
            patch["skill_json"] = skill_json
        if layer == "who":
            parent = str(frame.get("parent_node", "") or "")
            full_for_inference = f"{parent}/{node_id}" if parent else node_id
            skill_json = dict(patch.get("skill_json", {}) or {})
            skill_json["who_node_kind"] = "source_subgroup" if parent else "source"
            skill_json["advisor_source"] = parent.split("/")[0] if parent else node_id
            skill_json["retrieval_constraints"] = self._infer_who_retrieval_constraints(full_for_inference)
            patch["skill_json"] = skill_json
        if str(patch.get("operation", "") or "") == "add_sibling_node" and layer in {'why', "what", "how"}:
            skill_json = dict(patch.get("skill_json", {}) or {})
            anchor = self._infer_trial_anchor_node(
                layer,
                node_id,
                skill_json,
                fallback_reference=str(cluster.get("reference_node", "") or frame.get("reference_node_being_replaced_or_bypassed", "") or ""),
            )
            if anchor:
                patch["trial_anchor_node"] = anchor
                patch["why_anchor_is_the_right_competitor"] = f"{anchor} is the nearest existing same-layer node for train-time sibling comparison."
                skill_json["trial_anchor_node"] = anchor
                skill_json["trial_anchor_source"] = "fallback_inferred"
                patch["skill_json"] = skill_json
        return patch

    def _withered_revision_context(self, engine, cluster, limit=3):
        cluster = dict(cluster or {})
        layer = str(cluster.get("layer_hint", "") or "")
        reference = str(cluster.get("reference_node", "") or "")
        tree = engine.public_tree_store.load_tree(force_reload=True)
        rows = []
        for node_id, node in sorted(dict((tree.get(layer, {}) or {}) if layer else {}).items()):
            if str((node or {}).get("status", "") or "") != "withered":
                continue
            lifecycle = dict((node or {}).get("lifecycle", {}) or {})
            parent = str(lifecycle.get("parent_node", "") or self._parent_node_id(node_id))
            if reference and not (node_id == reference or node_id.startswith(reference + "/") or parent == reference):
                continue
            rows.append(
                {
                    "node_id": node_id,
                    "parent_node": parent,
                    "withered_reason": self._short_text(lifecycle.get("withered_reason", ""), 320),
                    "failure_patterns": list(lifecycle.get("failure_patterns", []) or [])[:4],
                    "repair_hints": list(lifecycle.get("repair_hints", []) or [])[:4],
                    "revision_memory": list(lifecycle.get("revision_memory", []) or [])[-4:],
                    "skill_md": self._short_text(node.get("skill_body", ""), 1200),
                }
            )
            if len(rows) >= int(limit or 3):
                break
        return rows

    def _common_cause_case_card(self, material_case):
        case = dict(material_case or {})
        fact = dict(case.get("fact_trace", {}) or {})
        task = dict(fact.get("task_context", {}) or {})
        protocol_issues = [
            str(x)
            for x in list(fact.get("quality_issues", []) or (case.get("diagnostic_tags", {}) or {}).get("quality_issues", []) or [])
            if str(x).strip()
        ][:5]
        advisor_summaries = []
        for row in list(fact.get("advisor_speech_summaries", []) or [])[:4]:
            if not isinstance(row, dict):
                continue
            meaningful = (
                str(row.get("answer_summary", "") or "").strip()
                or str(row.get("response_or_challenge", "") or "").strip()
                or list(row.get("candidate_evidence", []) or [])
                or list(row.get("protocol_issues", []) or [])
            )
            if not meaningful:
                continue
            advisor_summaries.append(
                {
                    "advisor_type": str(row.get("advisor_type", "") or ""),
                    "answer_summary": self._short_text(row.get("answer_summary", ""), 220),
                    "response_or_challenge": self._short_text(row.get("response_or_challenge", ""), 180),
                    "candidate_evidence": [
                        self._short_text(x, 180)
                        for x in list(row.get("candidate_evidence", []) or [])[:3]
                        if str(x).strip()
                    ],
                    "protocol_issues": [
                        self._short_text(x, 140)
                        for x in list(row.get("protocol_issues", []) or [])[:3]
                        if str(x).strip()
                    ],
                }
            )
        return self._json_safe(
            {
                "path": self._route_path_summary_for_prompt(fact.get("path", {}) or {}),
                "task": self._short_text(task.get("selected_task", ""), 260),
                "requires_candidate_contrast": bool(task.get("requires_candidate_contrast", False)),
                "expected_behavior": self._short_text(case.get("expected_behavior", ""), 260),
                "actual_behavior": self._short_text(case.get("actual_behavior", ""), 260),
                "specific_problem": self._short_text(case.get("specific_problem", ""), 260),
                "advisor_summaries": advisor_summaries,
                "protocol_issues": protocol_issues,
                "repair_hint": self._short_text(case.get("repair_hint", ""), 220),
            }
        )

    def _build_cluster_material_pack(self, cluster):
        cluster = dict(cluster or {})
        material_cases = [dict(x or {}) for x in list(cluster.get("material_cases", []) or []) if isinstance(x, dict)]
        representatives = list(cluster.get("representative_cases", []) or [])
        case_cards = [
            self._common_cause_case_card(case)
            for case in material_cases[:4]
            if isinstance(case, dict)
        ]
        if not representatives:
            for row in list(cluster.get("representative_reflections", []) or [])[:4]:
                row = dict(row or {})
                representatives.append(
                    {
                        "case_id": str(row.get("round_id", "") or ""),
                        "actual_behavior": self._short_text(row.get("observed_failure", "") or row.get("abstract_evidence", ""), 260),
                        "expected_behavior": self._short_text(row.get("why_current_tree_insufficient", ""), 260),
                        "specific_problem": self._short_text(row.get("specific_problem", "") or row.get("observed_failure", "") or row.get("failure_type", ""), 260),
                        "evidence": [self._short_text(row.get("abstract_evidence", ""), 220)] if row.get("abstract_evidence") else [],
                    }
                )
        repair_location = dict(cluster.get("repair_location", {}) or {})
        layer = str(repair_location.get("layer", "") or cluster.get("layer_hint", "") or "")
        reference = str(repair_location.get("reference_node", "") or cluster.get("reference_node", "") or "")
        problem_area = str(cluster.get("problem_area", "") or cluster.get("primary_failure_stage", "") or cluster.get("failed_stage", "") or "")
        material_signature = [str(x) for x in list(cluster.get("material_signature", []) or []) if str(x).strip()]
        what_failed = str(cluster.get("specific_problem", "") or "")
        if not what_failed and representatives:
            what_failed = str((representatives[0] or {}).get("specific_problem", "") or (representatives[0] or {}).get("actual_behavior", "") or "")
        repair_hint = str(cluster.get("repair_hint", "") or cluster.get("repair_need", "") or "")
        why_current = repair_hint
        if layer and reference:
            why_current = (
                f"the current {reference} contract does not yet force the behavior needed by this fact pattern: "
                f"{repair_hint or what_failed}"
            )
        counter_totals = dict(cluster.get("evidence_counter_totals", {}) or {})
        not_the_problem = [
            item for item, _ in Counter(dict(cluster.get("not_the_problem_distribution", {}) or cluster.get("ruled_out_distribution", {}) or {})).most_common(6)
            if str(item).strip()
        ]
        return self._json_safe(
            {
                "repair_location": {
                    "layer": layer,
                    "reference_node": reference,
                },
                "failure_cluster_packet": {
                    "cluster_id": str(cluster.get("cluster_id", "") or ""),
                    "support_user_ids": list(cluster.get("support_user_ids", []) or []),
                    "failure_mechanism": str(cluster.get("primary_failure_mechanism", "") or cluster.get("failure_type", "") or ""),
                    "failed_level": layer,
                    "reference_path": self._route_path_summary_for_prompt(
                        (list(dict(cluster.get("support_user_paths", {}) or {}).values()) or [{}])[0] or {}
                    ),
                    "repair_target": {
                        "layer": layer,
                        "operation": "add_child_or_sibling_node",
                        "reference_node": reference,
                        "desired_behavior": self._short_text(cluster.get("repair_hint", "") or cluster.get("repair_need", ""), 360),
                    },
                },
                "problem_area": problem_area,
                "material_signature": material_signature,
                "current_failure_analysis": {
                    "what_failed": self._short_text(what_failed or "communication failed for this material pattern", 420),
                    "why_current_node_failed": self._short_text(why_current, 520),
                },
                "evidence_summary": {
                    "case_count": int(cluster.get("support", 0) or 0),
                    "unique_users": int(cluster.get("unique_users", 0) or 0),
                    "outcome_distribution": dict(cluster.get("outcome_distribution", {}) or {}),
                    "round_effects": dict(cluster.get("round_effects", {}) or {}),
                    "counter_totals": counter_totals,
                },
                "case_cards": case_cards,
                "representative_cases": representatives[:4],
                "interaction_quality_summary": dict(cluster.get("interaction_quality_distribution", {}) or {}),
                "not_the_problem": not_the_problem,
                "repair_hint": self._short_text(repair_hint, 260),
                "evidence_strength": str(cluster.get("evidence_strength", "") or ("multi_case" if int(cluster.get("support", 0) or 0) >= 2 else "single_case")),
                "design_constraint": str(
                    cluster.get("design_constraint", "")
                    or (
                        "generate a reusable sprout for the repeated fact pattern"
                        if int(cluster.get("support", 0) or 0) >= 2
                        else "generate a narrow train-only sprout; do not overgeneralize beyond the observed fact pattern"
                    )
                ),
                "diagnostic_tags": {
                    "legacy_failure_type_distribution": dict(cluster.get("legacy_failure_type_distribution", {}) or {}),
                    "quality_issues": dict(cluster.get("quality_issues", {}) or {}),
                    "diagnostic_tag_distribution": dict(cluster.get("diagnostic_tag_distribution", {}) or {}),
                },
                "material_case_count": len(material_cases),
            }
        )

    def _build_current_failure_analysis(self, cluster):
        cluster = dict(cluster or {})
        if str(cluster.get("cluster_key_version", "") or "") == "material_signature_v1":
            pack = self._build_cluster_material_pack(cluster)
            analysis = dict(pack.get("current_failure_analysis", {}) or {})
            return self._json_safe(
                {
                    "where_problem_happened": str(pack.get("problem_area", "") or ""),
                    "what_failed": self._short_text(analysis.get("what_failed", ""), 420),
                    "why_current_behavior_failed": self._short_text(analysis.get("why_current_node_failed", ""), 520),
                    "repair_need": self._short_text(pack.get("repair_hint", ""), 260),
                    "evidence_across_users": dict(pack.get("evidence_summary", {}) or {}),
                    "representative_failure_evidence": list(pack.get("representative_cases", []) or [])[:4],
                    "not_the_problem": list(pack.get("not_the_problem", []) or [])[:6],
                    "diagnostic_tags": dict(pack.get("diagnostic_tags", {}) or {}),
                }
            )
        causal_cases = [dict(x or {}) for x in list(cluster.get("causal_cases", []) or []) if isinstance(x, dict)]
        representatives = list(cluster.get("representative_reflections", []) or [])
        primary_stage = str(cluster.get("primary_failure_stage", "") or cluster.get("failed_stage", "") or "")
        primary_mechanism = str(cluster.get("primary_failure_mechanism", "") or cluster.get("failure_type", "") or "")
        repair_need = str(cluster.get("repair_need", "") or "")
        representative_evidence = []
        for case in causal_cases[:4]:
            mechanism = dict((case.get("failure_mechanism", {}) or {}).get("primary_failure", {}) or {})
            if not mechanism:
                mechanism = dict((case.get("failure_mechanism", {}) or {}).get("primary_failure", {}) or {})
            representative_evidence.append(
                {
                    "round_effect": str(case.get("round_effect", "") or ""),
                    "actual_behavior": self._short_text(mechanism.get("actual_behavior", ""), 260),
                    "expected_behavior": self._short_text(mechanism.get("expected_behavior", ""), 260),
                    "specific_problem": self._short_text(mechanism.get("specific_problem", ""), 260),
                    "evidence": list(mechanism.get("evidence", []) or [])[:6],
                }
            )
        if not representative_evidence:
            for row in representatives[:4]:
                row = dict(row or {})
                representative_evidence.append(
                    {
                        "round_effect": str(row.get("round_effect", "") or ""),
                        "actual_behavior": self._short_text(row.get("observed_failure", "") or row.get("abstract_evidence", ""), 260),
                        "expected_behavior": self._short_text(row.get("why_current_tree_insufficient", ""), 260),
                        "specific_problem": self._short_text(row.get("observed_failure", "") or row.get("failure_type", ""), 260),
                        "evidence": [self._short_text(row.get("abstract_evidence", ""), 220)] if row.get("abstract_evidence") else [],
                    }
                )
        top_ruled_out = [
            item for item, _ in Counter(dict(cluster.get("ruled_out_distribution", {}) or {})).most_common(6)
            if str(item).strip()
        ]
        if not top_ruled_out:
            top_ruled_out = ["not aggregation loss", "not advisor pool empty"] if primary_stage == "advisor_interaction" else []
        what_failed = ""
        why_failed = ""
        if representative_evidence:
            first = representative_evidence[0]
            what_failed = str(first.get("specific_problem", "") or first.get("actual_behavior", "") or "")
            why_failed = str(first.get("expected_behavior", "") or repair_need or "")
        if not what_failed:
            what_failed = str(cluster.get("failure_type", "") or primary_mechanism or "communication failure")
        if not why_failed:
            why_failed = repair_need or str(cluster.get("dominant_failure", "") or "")
        return self._json_safe(
            {
                "where_problem_happened": primary_stage,
                "what_failed": self._short_text(what_failed, 420),
                "why_current_behavior_failed": self._short_text(why_failed, 520),
                "repair_need": self._short_text(repair_need, 260),
                "evidence_across_users": {
                    "support": int(cluster.get("support", 0) or 0),
                    "unique_users": int(cluster.get("unique_users", 0) or 0),
                    "round_effects": dict(cluster.get("round_effects", {}) or {}),
                    "common_mechanisms": dict(cluster.get("mechanism_distribution", {}) or {}),
                    "evidence_counter_totals": dict(cluster.get("evidence_counter_totals", {}) or {}),
                    "common_route_patterns": dict(cluster.get("route_selection_patterns", {}) or {}),
                    "common_advisor_execution_patterns": dict(cluster.get("advisor_execution_patterns", {}) or {}),
                },
                "representative_failure_evidence": representative_evidence[:4],
                "not_the_problem": top_ruled_out,
                "diagnostic_tags": {
                    "legacy_failure_type_distribution": dict(cluster.get("legacy_failure_type_distribution", {}) or {}),
                    "quality_issues": dict(cluster.get("quality_issues", {}) or {}),
                    "fallback_failure_type": str(cluster.get("failure_type", "") or ""),
                    "fallback_failed_stage": str(cluster.get("failed_stage", "") or ""),
                },
            }
        )

    def _node_context_summary(self, node_id, node):
        node = dict(node or {})
        body = str(node.get("skill_body", "") or "")
        return {
            "node_id": str(node_id or ""),
            "status": str(node.get("status", "") or ""),
            "description": self._short_text(node.get("description", "") or node.get("use_why", ""), 260),
            "use_why": self._short_text(node.get("use_why", ""), 320),
            "current_contract": self._short_text(body or node.get("if_selected", "") or node.get("action", ""), 1200),
            "skill_json": {
                "advisor_output_format": list(node.get("advisor_output_format", []) or [])[:8],
                "task_output_format": list(node.get("task_output_format", []) or [])[:8],
                "summary_hints": dict(node.get("summary_hints", {}) or {}),
                "execution_hint": dict(node.get("execution_hint", {}) or {}),
            },
        }

    def _build_node_design_context(self, engine, cluster, strategy_menu, current_failure_analysis=None):
        cluster = dict(cluster or {})
        strategy_menu = dict(strategy_menu or {})
        layer = str(strategy_menu.get("layer", "") or cluster.get("layer_hint", "") or "")
        reference = str(strategy_menu.get("reference_node", "") or cluster.get("reference_node", "") or "")
        tree = engine.public_tree_store.load_tree(force_reload=True)
        nodes = dict((tree.get(layer, {}) or {}) if layer else {})
        reference_node = dict(nodes.get(reference, {}) or {})
        same_layer_examples = []
        for node_id, node in sorted(nodes.items()):
            node_id = str(node_id or "")
            if not node_id or node_id == reference:
                continue
            if "/" in node_id:
                continue
            same_layer_examples.append(self._node_context_summary(node_id, node))
            if len(same_layer_examples) >= 4:
                break
        gap = self._short_text(
            (current_failure_analysis or {}).get("repair_need", "")
            or (current_failure_analysis or {}).get("why_current_behavior_failed", "")
            or cluster.get("repair_need", ""),
            420,
        )
        return self._json_safe(
            {
                "layer": layer,
                "reference_node": reference,
                "reference_mode_options": [
                    str(row.get("strategy", "") or "")
                    for row in list(strategy_menu.get("strategies", []) or [])
                    if isinstance(row, dict)
                ],
                "refine_existing_node_context": {
                    "parent_node": self._node_context_summary(reference, reference_node) if reference_node else {},
                    "gap_between_contract_and_failure": gap,
                    "design_instruction": (
                        "Choose refine_existing_node only if this parent is clearly the right abstraction and the repair is a narrow subcase. "
                        "Do not keep growing this parent just because it was the failing route."
                    ),
                },
                "create_replacement_parent_context": {
                    "reference_node_being_bypassed": self._node_context_summary(reference, reference_node) if reference_node else {},
                    "why_not_enough": gap,
                    "same_layer_examples": same_layer_examples,
                    "design_instruction": (
                        "Preferred for repeated public_tree_need clusters: write a distinct same-layer node whose protocol directly repairs "
                        "the failure mechanism and can compete with the reference route."
                    ),
                },
            }
        )

    def _allowed_tree_operations_for_generation(self, strategy_menu):
        strategy_menu = dict(strategy_menu or {})
        reference = str(strategy_menu.get("reference_node", "") or "")
        out = {
            "layer": str(strategy_menu.get("layer", "") or ""),
            "reference_node": reference,
            "allowed_operations": [],
        }
        for row in list(strategy_menu.get("strategies", []) or []):
            if not isinstance(row, dict):
                continue
            strategy = str(row.get("strategy", "") or "")
            operation = str(row.get("operation", "") or "")
            if strategy == "refine_existing_node" or operation == "add_child_node":
                meaning = "Exception path: the reference node is broadly correct; create a narrow stricter child node under it."
                decision_rule = (
                    "Use only when CommonCauseAnalysis explicitly says the current reference behavior needs specialization "
                    "inside the same abstraction. If uncertain, do not use this."
                )
            else:
                meaning = "Preferred path: the reference node is insufficient or overused; create a same-layer alternative route."
                decision_rule = (
                    "Use for most what/how public_tree_need clusters, especially repeated failures under the same parent, "
                    "missing protocol capability, or uncertainty about whether the parent abstraction is right."
                )
            out["allowed_operations"].append(
                {
                    "strategy": strategy,
                    "operation": operation,
                    "layer": str(row.get("layer", "") or strategy_menu.get("layer", "") or ""),
                    "parent_node": str(row.get("parent_node", "") or ""),
                    "reference_node": reference,
                    "meaning": meaning,
                    "decision_rule": decision_rule,
                }
            )
        return self._json_safe(out)

    def _current_tree_context_for_generation(self, node_design_context, current_failure_analysis=None):
        ctx = dict(node_design_context or {})
        refine = dict(ctx.get("refine_existing_node_context", {}) or {})
        replacement = dict(ctx.get("create_replacement_parent_context", {}) or {})
        current_failure_analysis = dict(current_failure_analysis or {})
        return self._json_safe(
            {
                "layer": str(ctx.get("layer", "") or ""),
                "reference_node": str(ctx.get("reference_node", "") or ""),
                "reference_node_contract": dict(refine.get("parent_node", {}) or replacement.get("reference_node_being_bypassed", {}) or {}),
                "same_layer_nodes": list(replacement.get("same_layer_examples", []) or [])[:4],
                "tree_gap_to_repair": self._short_text(
                    current_failure_analysis.get("repair_need", "")
                    or current_failure_analysis.get("why_current_behavior_failed", "")
                    or refine.get("gap_between_contract_and_failure", "")
                    or replacement.get("why_not_enough", ""),
                    520,
                ),
            }
        )

    def _common_cause_prompt_payload(self, cluster):
        cluster = dict(cluster or {})
        pack = self._build_cluster_material_pack(cluster)
        failure_packet = dict(pack.get("failure_cluster_packet", {}) or {})
        support_user_ids = list(failure_packet.pop("support_user_ids", []) or [])
        failure_packet["support_user_count"] = len(support_user_ids)
        pack["failure_cluster_packet"] = failure_packet
        evidence_summary = dict(pack.get("evidence_summary", {}) or {})
        lean_evidence_summary = {
            "case_count": int(evidence_summary.get("case_count", cluster.get("support", 0)) or 0),
            "unique_users": int(evidence_summary.get("unique_users", cluster.get("unique_users", 0)) or 0),
            "outcome_distribution": dict(evidence_summary.get("outcome_distribution", {}) or {}),
            "round_effects": dict(evidence_summary.get("round_effects", {}) or {}),
        }
        return self._json_safe(
            {
                "cluster_id": str(cluster.get("cluster_id", "") or ""),
                "support": int(cluster.get("support", 0) or 0),
                "unique_users": int(cluster.get("unique_users", 0) or 0),
                "repair_location": dict(pack.get("repair_location", {}) or {}),
                "problem_area": str(pack.get("problem_area", "") or ""),
                "evidence_summary": lean_evidence_summary,
                "case_cards": list(pack.get("case_cards", []) or [])[:4],
                "not_the_problem": list(pack.get("not_the_problem", []) or [])[:6],
                "current_failure_analysis_hint": dict(pack.get("current_failure_analysis", {}) or {}),
                "failure_cluster_packet": failure_packet,
            }
        )

    def _normalize_common_cause_analysis(self, payload, cluster, source="llm"):
        payload = dict(payload or {})
        cluster = dict(cluster or {})
        level = str(payload.get("problem_level", "") or payload.get("level", "") or cluster.get("layer_hint", "") or "how").strip()
        if level not in {"what", "who", "how"}:
            level = "how"
        confidence = str(payload.get("confidence", "") or "").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium" if source == "llm" else "low"
        reference = str(payload.get("reference_node", "") or cluster.get("reference_node", "") or "").strip()
        return self._json_safe(
            {
                "cluster_id": str(payload.get("cluster_id", "") or cluster.get("cluster_id", "") or ""),
                "is_public_tree_problem": bool(payload.get("is_public_tree_problem", True)),
                "problem_level": level,
                "common_failure_reason": self._short_text(payload.get("common_failure_reason", "") or cluster.get("specific_problem", ""), 620),
                "why_existing_nodes_insufficient": self._short_text(
                    payload.get("why_existing_nodes_insufficient", "") or cluster.get("repair_hint", "") or cluster.get("repair_need", ""),
                    620,
                ),
                "required_new_capability": self._short_text(payload.get("required_new_capability", "") or cluster.get("repair_need", "") or cluster.get("repair_hint", ""), 420),
                "reference_node": reference,
                "not_public_tree_reason": self._short_text(payload.get("not_public_tree_reason", ""), 360),
                "confidence": confidence,
                "source": source,
            }
        )

    def _llm_analyze_tree_need_common_cause(self, engine, cluster):
        cluster = dict(cluster or {})
        payload = self._common_cause_prompt_payload(cluster)
        system_prompt = (
            "Analyze a cluster of compact public_tree_need signals. Decide whether the shared cause is a public communication-tree problem. "
            "Use case_cards first: task, advisor_summaries, protocol_issues, expected vs actual behavior. "
            "Use evidence_summary only for recurrence strength. Do not generate nodes, patches, SKILL.md, route operations, or item preferences. "
            "If evidence is only user taste, parser/aggregation bug, or user absorption, set is_public_tree_problem=false. "
            "Return strict JSON only."
        )
        user_prompt = (
            "PublicTreeNeedClusterForCommonCause:\n"
            f"{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "cluster_id": "",\n'
            '  "is_public_tree_problem": true,\n'
            '  "problem_level": "what|who|how",\n'
            '  "common_failure_reason": "",\n'
            '  "why_existing_nodes_insufficient": "",\n'
            '  "required_new_capability": "",\n'
            '  "reference_node": "",\n'
            '  "not_public_tree_reason": "",\n'
            '  "confidence": "low|medium|high"\n'
            "}\n"
        )
        raw_response = None
        parsed = None
        llm_usage = {}
        try:
            if not hasattr(engine.args, "max_retry_num"):
                setattr(engine.args, "max_retry_num", 3)
            if not hasattr(engine.args, "temperature"):
                setattr(engine.args, "temperature", 0.2)
            update_llm_prompt_trace_context(
                phase="tree_common_cause_analysis",
                advisor_index="",
                advisor_id="",
                advisor_type="",
                path_why="",
                path_who="",
                path_how="",
            )
            raw_response = llm_request(system_prompt, user_prompt, engine.args)
            llm_usage = get_last_llm_request_usage()
            parsed = self._extract_json_object(raw_response)
        except Exception as exc:
            parsed = {"is_public_tree_problem": False, "not_public_tree_reason": f"llm_failed: {exc}", "confidence": "low"}
        if not isinstance(parsed, dict):
            parsed = {"is_public_tree_problem": False, "not_public_tree_reason": "llm_returned_no_json", "confidence": "low"}
        analysis = self._normalize_common_cause_analysis(parsed, cluster, source="llm")
        append_jsonl(
            engine.public_tree_store.index_dir / "tree_common_cause_prompt_io.jsonl",
            {
                "event": "tree_common_cause_prompt_io",
                "ts": int(time.time()),
                "cluster_id": str(cluster.get("cluster_id", "") or ""),
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "llm_usage": llm_usage,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response_preview": self._short_text(raw_response, 2400),
                "parsed_json": parsed if isinstance(parsed, dict) else {},
                "analysis": analysis,
            },
        )
        return analysis

    def _generate_patch_candidates_for_failure_cluster(self, engine, cluster):
        cluster = dict(cluster or {})
        frame = self._decide_patch_frame_for_cluster(engine, cluster)
        strategy_menu = self._candidate_patch_strategies_for_cluster(engine, cluster, frame)
        existing = self._find_existing_sprout_for_cluster(engine, cluster, frame)
        if existing:
            self._append_sprout_revision_memory(
                engine,
                frame.get("layer"),
                existing.get("node_id"),
                cluster,
                "same failure cluster already has a sprout; expose it as context so the next patch can avoid a near-duplicate",
            )
        max_candidates = 3 if str(cluster.get("session_outcome", "") or "") == "TW" and float(cluster.get("severity", 0.0) or 0.0) >= 8 else 2
        context_nodes = self._retrieve_tree_context_for_signal(
            engine,
            {
                "signal": f"{strategy_menu.get('layer')}/{strategy_menu.get('reference_node') or cluster.get('reference_node')}",
                "rows": [],
            },
        )
        common_cause_analysis = dict(cluster.get("common_cause_analysis", {}) or cluster.get("_common_cause_analysis", {}) or {})
        current_failure_analysis = self._build_current_failure_analysis(cluster)
        if common_cause_analysis:
            current_failure_analysis = self._json_safe(
                {
                    **dict(current_failure_analysis or {}),
                    "where_problem_happened": str(common_cause_analysis.get("problem_level", "") or current_failure_analysis.get("where_problem_happened", "")),
                    "what_failed": self._short_text(
                        common_cause_analysis.get("common_failure_reason", "") or current_failure_analysis.get("what_failed", ""),
                        620,
                    ),
                    "why_current_behavior_failed": self._short_text(
                        common_cause_analysis.get("why_existing_nodes_insufficient", "") or current_failure_analysis.get("why_current_behavior_failed", ""),
                        620,
                    ),
                    "repair_need": self._short_text(
                        common_cause_analysis.get("required_new_capability", "") or current_failure_analysis.get("repair_need", ""),
                        420,
                    ),
                    "common_cause_confidence": str(common_cause_analysis.get("confidence", "") or ""),
                }
            )
        node_design_context = self._build_node_design_context(engine, cluster, strategy_menu, current_failure_analysis=current_failure_analysis)
        allowed_tree_operations = self._allowed_tree_operations_for_generation(strategy_menu)
        current_tree_context = self._current_tree_context_for_generation(node_design_context, current_failure_analysis=current_failure_analysis)
        system_prompt = (
            "You generate candidate public communication-tree nodes. Use CommonCauseAnalysis as the only diagnosis; "
            "CurrentTreeContext is only placement context. Pick one AllowedTreeOperations entry per candidate. "
            "Prefer create_replacement_parent/add_sibling_node for what/how nodes so the tree gains a distinct same-layer route instead of repeatedly deepening one parent. "
            "Use refine_existing_node/add_child_node only when CommonCauseAnalysis explicitly says the reference node is the right abstraction and only a narrow stricter subcase is missing. "
            "If the evidence is ambiguous between child and sibling, choose create_replacement_parent/add_sibling_node. "
            "Do not grow another child under the same parent just because the failed route used that parent. "
                "Directly repair required_new_capability in why_needed/evidence_pattern and use a short semantic new_node_id. "
                "For same-layer when/what/how nodes include trial_anchor_node and why_anchor_is_the_right_competitor. "
                "For when/what include skill_json.selection_profile; for what/how include skill_json.summary_hints. "
                "For every what node, skill_json.task_output_format must list the exact advisor output fields in SKILL.md, "
                "and skill_json.summary_hints.important_output_fields must include CandidateView, TaskAnswer, AskUser, "
                "plus every new task-specific field introduced by this node. "
                "For who create advisor source groups/subgroups only, defaulting to refine under an existing root with "
                "who_node_kind='source_subgroup', advisor_source, and retrieval_constraints. "
            "Do not use item preferences, diagnostic tags, user ids, item/artist names, target labels, or historical failed-node memories. "
            "Return strict JSON only."
        )
        user_prompt = (
            f"CommonCauseAnalysis:\n{json.dumps(common_cause_analysis, ensure_ascii=False, default=str)}\n\n"
            f"CurrentTreeContext:\n{json.dumps(current_tree_context, ensure_ascii=False, default=str)}\n\n"
            f"AllowedTreeOperations:\n{json.dumps(allowed_tree_operations, ensure_ascii=False, default=str)}\n\n"
            f"LayerTemplate:\n{self._tree_patch_template(strategy_menu.get('layer'))}\n\n"
            "Return JSON: {\"candidates\": [{\"chosen_strategy\":\"create_replacement_parent|refine_existing_node\", "
            "\"operation\":\"add_sibling_node|add_child_node\", \"layer\":\"\", \"parent_node\":\"\", \"new_node_id\":\"\", "
            "\"trial_anchor_node\":\"\", \"why_anchor_is_the_right_competitor\":\"\", "
            "\"why_refinement_is_or_is_not_enough\":\"\", \"why_new_parent_is_or_is_not_needed\":\"\", "
            "\"why_needed\":\"\", \"evidence_pattern\":\"\", \"skill_md\":\"\", "
            "\"skill_json\":{\"trial_anchor_node\":\"\", \"trial_anchor_source\":\"llm\", "
            "\"task_output_format\":[\"CandidateView:\", \"<task-specific field>:\", \"TaskAnswer:\", \"AskUser:\"], "
            "\"advisor_output_format\":[], "
            "\"selection_profile\":{\"requires\":[], \"prefers\":[], \"do_not_use_why\":[], \"selection_prior\":0.0}, "
            "\"summary_hints\":{\"task_focus\":\"\", \"important_output_fields\":[], \"preserve_interaction_fields\":[]}, "
            "\"who_node_kind\":\"source_subgroup\", \"advisor_source\":\"\", \"retrieval_constraints\":{}}}]}"
        )
        candidates = []
        raw_response = None
        parsed = None
        llm_usage = {}
        try:
            update_llm_prompt_trace_context(
                phase="tree_node_generation",
                advisor_index="",
                advisor_id="",
                advisor_type="",
                path_why="",
                path_who="",
                path_how="",
            )
            raw_response = llm_request(system_prompt, user_prompt, engine.args)
            llm_usage = get_last_llm_request_usage()
            parsed = self._extract_json_object(raw_response)
        except Exception as exc:
            parsed = {"error": str(exc), "candidates": []}
        if isinstance(parsed, dict):
            strategies = {
                str(row.get("strategy", "") or ""): dict(row)
                for row in list((strategy_menu or {}).get("strategies", []) or [])
                if isinstance(row, dict)
            }
            for raw in list(parsed.get("candidates", []) or [])[:max_candidates]:
                if not isinstance(raw, dict):
                    continue
                patch = dict(raw)
                chosen_strategy = str(patch.get("chosen_strategy", "") or "").strip()
                strategy = dict(strategies.get(chosen_strategy, {}) or {})
                if not strategy:
                    for row in strategies.values():
                        if (
                            str(row.get("operation", "") or "") == str(patch.get("operation", "") or "")
                            and str(row.get("parent_node", "") or "") == str(patch.get("parent_node", "") or "")
                        ):
                            strategy = dict(row)
                            chosen_strategy = str(row.get("strategy", "") or "")
                            break
                patch["chosen_strategy"] = chosen_strategy or str(strategy.get("strategy", "") or "")
                if strategy:
                    patch["operation"] = str(strategy.get("operation", "") or patch.get("operation", "") or "add_child_node")
                    patch["layer"] = str(strategy.get("layer", "") or patch.get("layer", "") or strategy_menu.get("layer") or frame.get("layer") or "how")
                    patch["parent_node"] = str(strategy.get("parent_node", "") or "")
                else:
                    patch["layer"] = str(patch.get("layer", "") or strategy_menu.get("layer") or frame.get("layer") or "how")
                    fallback_operation = "add_child_node" if patch["layer"] == "who" else "add_sibling_node"
                    patch["operation"] = str(patch.get("operation", "") or fallback_operation)
                    patch["parent_node"] = str(patch.get("parent_node", "") or "")
                    if patch["operation"] == "add_sibling_node" and patch["layer"] in {'why', "what", "how"}:
                        patch["parent_node"] = ""
                raw_node_id = patch.get("new_node_id") or self._semantic_fallback_node_id(cluster, patch.get("chosen_strategy"))
                patch["new_node_id"] = self._unique_new_node_id(engine, patch["layer"], patch.get("parent_node", ""), raw_node_id)
                patch["status"] = "active"
                skill_json = dict(patch.get("skill_json", {}) or {})
                skill_json["node_level"] = patch["layer"]
                skill_json["status"] = "active"
                dataset = self._dataset_slug(engine)
                skill_json["dataset"] = dataset
                skill_json["source_dataset"] = dataset
                skill_json["dataset_scope"] = [dataset]
                skill_json.setdefault("global_default", False)
                skill_json.setdefault("route_injection_only", True)
                if patch["layer"] in {'why', "what"}:
                    skill_json.setdefault(
                        "selection_profile",
                        {"requires": [], "prefers": [], "do_not_use_why": [], "selection_prior": skill_json.get("selection_prior", 0.15)},
                    )
                if patch["layer"] == "what":
                    if not list(skill_json.get("task_output_format", []) or []):
                        inferred_fields = self._what_output_fields_from_skill_md(patch.get("skill_md", ""))
                        if inferred_fields:
                            skill_json["task_output_format"] = [f"{field}:" for field in inferred_fields]
                    skill_json["summary_hints"] = self._normalized_summary_hints_for_patch(
                        "what",
                        skill_json,
                        patch.get("skill_md", ""),
                    )
                if patch["layer"] == "how":
                    skill_json["summary_hints"] = self._normalized_summary_hints_for_patch(
                        "how",
                        skill_json,
                        patch.get("skill_md", ""),
                    )
                if patch["layer"] == "who":
                    parent = str(patch.get("parent_node", "") or "")
                    full_for_inference = f"{parent}/{patch['new_node_id']}" if parent else str(patch["new_node_id"])
                    skill_json.setdefault("who_node_kind", "source_subgroup" if parent else "source")
                    skill_json.setdefault("advisor_source", parent.split("/")[0] if parent else str(patch["new_node_id"]))
                    inferred_constraints = self._infer_who_retrieval_constraints(full_for_inference)
                    constraints = dict(inferred_constraints)
                    constraints.update(dict(skill_json.get("retrieval_constraints", {}) or {}))
                    skill_json["retrieval_constraints"] = constraints
                if str(patch.get("operation", "") or "") == "add_sibling_node" and patch["layer"] in {'why', "what", "how"}:
                    raw_anchor = str(patch.get("trial_anchor_node", "") or skill_json.get("trial_anchor_node", "") or "").strip()
                    fallback_reference = str(
                        strategy.get("reference_node_being_replaced_or_bypassed", "")
                        or strategy.get("suggested_trial_anchor_node", "")
                        or strategy_menu.get("reference_node", "")
                        or cluster.get("reference_node", "")
                        or ""
                    )
                    anchor = self._infer_trial_anchor_node(
                        patch["layer"],
                        patch["new_node_id"],
                        {**skill_json, "trial_anchor_node": raw_anchor},
                        fallback_reference=fallback_reference,
                    )
                    if anchor:
                        patch["trial_anchor_node"] = anchor
                        patch.setdefault(
                            "why_anchor_is_the_right_competitor",
                            f"{anchor} is the closest existing same-layer node for train-time sibling trial.",
                        )
                        skill_json["trial_anchor_node"] = anchor
                        skill_json["trial_anchor_source"] = str(skill_json.get("trial_anchor_source", "") or ("llm" if raw_anchor else "fallback_inferred"))
                skill_json["evolution_reuse"] = self._cluster_reuse_fingerprint(cluster)
                skill_json["evolution_reuse"]["source_cluster_id"] = str(cluster.get("cluster_id", "") or "")
                patch["skill_json"] = skill_json
                patch["_source_cluster"] = str(cluster.get("cluster_id", "") or "")
                patch["repair_need"] = str(current_failure_analysis.get("repair_need", "") or cluster.get("repair_need", "") or "")
                candidates.append(patch)
        if not candidates:
            candidates.append(self._deterministic_failure_patch(engine, cluster, frame=frame, variant=1))
        event = {
            "event": "tree_patch_candidates",
            "ts": int(time.time()),
            "cluster_id": str(cluster.get("cluster_id", "") or ""),
            "frame": frame,
            "allowed_tree_operations": allowed_tree_operations,
            "common_cause_analysis": common_cause_analysis,
            "current_failure_analysis": current_failure_analysis,
            "current_tree_context": current_tree_context,
            "relevant_existing_nodes_debug": context_nodes,
            "existing_sprout_context": existing or {},
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
            "llm_usage": llm_usage,
            "raw_response_preview": self._short_text(raw_response, 1600),
            "parsed_json": parsed if isinstance(parsed, dict) else {},
            "candidate_count": len(candidates),
        }
        append_jsonl(engine.public_tree_store.index_dir / "tree_patch_candidates.jsonl", event)
        append_jsonl(
            engine.public_tree_store.index_dir / "tree_patch_prompt_io.jsonl",
            {
                "event": "tree_patch_prompt_io",
                "ts": int(time.time()),
                "cluster_id": str(cluster.get("cluster_id", "") or ""),
                "cluster_key_version": str(cluster.get("cluster_key_version", "") or ""),
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "llm_usage": llm_usage,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "common_cause_analysis": common_cause_analysis,
                "current_tree_context": current_tree_context,
                "allowed_tree_operations": allowed_tree_operations,
                "raw_response_preview": self._short_text(raw_response, 2400),
                "parsed_json": parsed if isinstance(parsed, dict) else {},
                "candidate_count": len(candidates),
            },
        )
        return candidates

    def _critic_and_repair_patch_candidate(self, engine, candidate, cluster, forbidden_terms):
        candidate = dict(candidate or {})
        cluster = dict(cluster or {})
        ok, reason = engine.public_tree_store.validate_tree_patch(candidate, forbidden_terms=forbidden_terms)
        repaired = dict(candidate)
        repaired_from = ""
        if not ok:
            repaired_from = str(reason)
            frame = {
                "operation": str(candidate.get("operation", "") or "add_child_node"),
                "layer": str(candidate.get("layer", "") or cluster.get("layer_hint", "") or "how"),
                "parent_node": str(candidate.get("parent_node", "") or cluster.get("reference_node", "") or ""),
                "new_node_id": str(candidate.get("new_node_id", "") or ""),
                "status": "active",
            }
            if not frame["new_node_id"]:
                frame["new_node_id"] = self._unique_new_node_id(
                    engine,
                    frame["layer"],
                    frame["parent_node"],
                    self._semantic_fallback_node_id(cluster, candidate.get("chosen_strategy", "")),
                )
            can_repair_same_strategy = True
            if frame["operation"] == "add_child_node" and not self._node_exists(engine, frame["layer"], frame["parent_node"]):
                can_repair_same_strategy = False
            if frame["operation"] not in {"add_child_node", "add_sibling_node", "mark_withered"}:
                can_repair_same_strategy = False
            if can_repair_same_strategy:
                repaired = self._deterministic_failure_patch(engine, cluster, frame=frame, variant=2)
                for key in [
                    "chosen_strategy",
                    "why_refinement_is_or_is_not_enough",
                    "why_new_parent_is_or_is_not_needed",
                ]:
                    if key in candidate:
                        repaired[key] = candidate.get(key)
                ok, reason = engine.public_tree_store.validate_tree_patch(repaired, forbidden_terms=forbidden_terms)
            append_jsonl(
                engine.public_tree_store.index_dir / "tree_patch_repair_io.jsonl",
                {
                    "event": "tree_patch_repair",
                    "ts": int(time.time()),
                    "cluster_id": str(cluster.get("cluster_id", "") or ""),
                    "original_reason": repaired_from,
                    "repair_valid": bool(ok),
                    "repair_reason": str(reason),
                    "repaired_patch": repaired,
                },
            )
        score = 0.0
        if ok:
            score += 10.0
            if str(repaired.get("layer", "") or "") == str(cluster.get("layer_hint", "") or ""):
                score += 2.0
            if str(repaired.get("status", "") or "") == "active":
                score += 1.0
            score += min(3.0, float(cluster.get("severity", 0.0) or 0.0) / 4.0)
        result = {
            "event": "tree_patch_critic",
            "ts": int(time.time()),
            "cluster_id": str(cluster.get("cluster_id", "") or ""),
            "valid": bool(ok),
            "reason": str(reason),
            "score": float(score),
            "patch": repaired,
            "repaired_from": repaired_from,
        }
        append_jsonl(engine.public_tree_store.index_dir / "tree_patch_critic.jsonl", result)
        return result

    def _select_best_patch_candidate(self, critic_rows):
        valid = [dict(row or {}) for row in list(critic_rows or []) if bool((row or {}).get("valid", False))]
        if not valid:
            return None
        valid.sort(key=lambda row: (-float(row.get("score", 0.0) or 0.0), str((row.get("patch", {}) or {}).get("new_node_id", ""))))
        return dict(valid[0].get("patch", {}) or {})

    def _layer_examples(self, engine, layer, limit=1):
        tree = engine.public_tree_store.load_tree(force_reload=True)
        nodes = dict((tree.get(layer, {}) or {}))
        examples = []
        preferred = {
            'why': ["candidate-conflict", "internal-prior-conflict"],
            "what": ["reduce_hesitation_set", "compare_remaining_candidates"],
            "who": ["trusted-advisors", "similar-users"],
            "how": ["multi-cooperative", "multi-competitive"],
        }.get(layer, [])
        ordered = preferred + [node_id for node_id in sorted(nodes) if node_id not in preferred]
        for node_id in ordered:
            node = nodes.get(node_id, {}) or {}
            body = str(node.get("skill_body", "") or "")
            if body:
                examples.append({"node_id": node_id, "skill_md": self._short_text(body, 1200)})
            if len(examples) >= int(limit):
                break
        return examples

    @staticmethod
    def _signal_key_parts(signal):
        raw = str(signal or "").strip().replace("\\", "/")
        parts = [part for part in raw.split("/") if part]
        layer = parts[0] if parts and parts[0] in {'why', "what", "who", "how"} else ""
        node_hint = "/".join(parts[1:]) if layer else raw
        return layer, node_hint

    def _retrieve_tree_context_for_signal(self, engine, problem_signal):
        signal = str((problem_signal or {}).get("signal", "") or "")
        signal_low = signal.lower()
        layer, node_hint = self._signal_key_parts(signal)
        wanted = []

        def add(level, node_id):
            if level and node_id:
                wanted.append((level, node_id))

        if layer in {'why', "what", "who", "how"}:
            add(layer, node_hint.split("/")[0] if layer != "who" else node_hint.split("/")[0])
        if signal_low.startswith("what/"):
            add("what", "compare_remaining_candidates")
            add("what", "reasoning_check")
            add("what", "evidence_gap_check")
        if signal_low.startswith("how/"):
            add("how", "multi-cooperative")
            add("how", "multi-competitive")
        if signal_low.startswith("who/") or signal_low.startswith("trusted-advisors"):
            add("who", "trusted-advisors")
            add("who", "similar-users")
            add("who", "experienced-users")
        if any(token in signal_low for token in ["counterfactual", "comparison", "compare", "debate", "tradeoff"]):
            add("what", "compare_remaining_candidates")
            add("what", "reasoning_check")
            add("how", "multi-competitive")
        if any(token in signal_low for token in ["feedback", "unanswered", "repair", "missing", "evidence"]):
            add("what", "evidence_gap_check")
            add("how", "multi-cooperative")
        if any(token in signal_low for token in ["cooperative", "coverage"]):
            add("how", "multi-cooperative")
        if any(token in signal_low for token in ["competitive", "challenge", "structured"]):
            add("how", "multi-competitive")

        if not wanted:
            for level in ["what", "how"]:
                for example in self._layer_examples(engine, level, limit=1):
                    add(level, example.get("node_id", ""))

        tree = engine.public_tree_store.load_tree(force_reload=True)
        context = []
        seen = set()
        for level, node_id in wanted:
            key = (level, node_id)
            if key in seen:
                continue
            seen.add(key)
            node = dict((tree.get(level, {}) or {}).get(node_id, {}) or {})
            if not node:
                continue
            context.append(
                {
                    "layer": level,
                    "node_id": node_id,
                    "status": str(node.get("status", "") or ""),
                    "skill_md": self._short_text(node.get("skill_body", ""), 1600),
                }
            )
            if len(context) >= 4:
                break
        return context

    def _llm_generate_tree_patch_proposals(
        self,
        engine,
        rows,
        branch_stats,
        fine_branch_stats,
        path_stats,
        fine_path_stats,
        risky_paths,
        proposed_new,
        node_stats=None,
    ):
        if not bool(getattr(engine.args, "com_llm_evolve_user_skill", True)):
            return {"summary": "skipped because com_llm_evolve_user_skill is false", "patches": [], "problem_signals": [], "failure_clusters": []}
        reflections = self._extract_round_failure_reflections(rows)
        for ref in reflections:
            append_jsonl(engine.public_tree_store.index_dir / "round_failure_reflections.jsonl", ref)
        clusters = self._cluster_failure_reflections(reflections)
        for cluster in clusters:
            cluster_to_log = dict(cluster or {})
            cluster_to_log["representative_reflections"] = list(cluster_to_log.get("representative_reflections", []) or [])[:3]
            cluster_to_log["common_cause_payload_preview"] = self._common_cause_prompt_payload(cluster_to_log)
            cluster_to_log["current_failure_analysis"] = self._build_current_failure_analysis(cluster_to_log)
            append_jsonl(engine.public_tree_store.index_dir / "round_failure_clusters.jsonl", cluster_to_log)
        ready_clusters = [dict(row or {}) for row in clusters if bool((row or {}).get("ready", False))]
        if not ready_clusters:
            return {
                "summary": "no public-tree-need anchor clusters were ready",
                "patches": [],
                "problem_signals": [],
                "failure_clusters": [
                    {
                        "cluster_id": str(row.get("cluster_id", "") or ""),
                        "support": int(row.get("support", 0) or 0),
                        "unique_users": int(row.get("unique_users", 0) or 0),
                        "ready": bool(row.get("ready", False)),
                    }
                    for row in clusters[:8]
                ],
            }
        patches = []
        summaries = []
        common_cause_by_cluster = {}
        reused_nodes = []
        patch_generation_errors = []
        forbidden_terms = self._tree_patch_forbidden_terms(rows)
        for cluster in ready_clusters:
            reusable = self._find_reusable_active_node_for_cluster(engine, cluster)
            if reusable:
                reused = {
                    "operation": "reuse_existing_node",
                    "layer": str(cluster.get("layer_hint", "") or ""),
                    "node_id": str(reusable.get("node_id", "") or ""),
                    "new_node_id": str(reusable.get("node_id", "") or ""),
                    "status": "active",
                    "reuse_reason": "matched_existing_active_tree_node_for_same_failure_cluster",
                    "match_score": int(reusable.get("match_score", 0) or 0),
                    "reuse_fingerprint": dict(reusable.get("fingerprint", {}) or {}),
                    "_source_cluster": str(cluster.get("cluster_id", "") or ""),
                    "_route_injection_cluster": self._cluster_route_injection_packet(cluster),
                }
                reused_nodes.append(reused)
                summaries.append(
                    f"{cluster.get('cluster_id', '')}: reused existing {reused.get('layer')}/{reused.get('node_id')}"
                )
                append_jsonl(
                    engine.public_tree_store.index_dir / "tree_node_reuse.jsonl",
                    {
                        "event": "reuse_existing_active_tree_node",
                        "ts": int(time.time()),
                        "cluster_id": str(cluster.get("cluster_id", "") or ""),
                        "node_id": str(reused.get("node_id", "") or ""),
                        "layer": str(reused.get("layer", "") or ""),
                        "match_score": int(reused.get("match_score", 0) or 0),
                        "fingerprint": dict(reusable.get("fingerprint", {}) or {}),
                    },
                )
                continue
            common_cause = self._llm_analyze_tree_need_common_cause(engine, cluster)
            cluster["common_cause_analysis"] = common_cause
            common_cause_by_cluster[str(cluster.get("cluster_id", "") or "")] = common_cause
            if not bool(common_cause.get("is_public_tree_problem", False)):
                summaries.append(
                    f"{cluster.get('cluster_id', '')}: common-cause analysis rejected public-tree repair: "
                    f"{common_cause.get('not_public_tree_reason', '')}"
                )
                append_jsonl(
                    engine.public_tree_store.index_dir / "non_tree_repair_signals.jsonl",
                    {
                        "event": "common_cause_rejected_public_tree",
                        "ts": int(time.time()),
                        "cluster_id": str(cluster.get("cluster_id", "") or ""),
                        "analysis": common_cause,
                    },
                )
                continue
            cluster["layer_hint"] = str(common_cause.get("problem_level", "") or cluster.get("layer_hint", "") or "how")
            cluster["reference_node"] = str(common_cause.get("reference_node", "") or cluster.get("reference_node", "") or "")
            cluster["repair_location"] = {
                "layer": str(cluster.get("layer_hint", "") or ""),
                "reference_node": str(cluster.get("reference_node", "") or ""),
            }
            if common_cause.get("common_failure_reason"):
                cluster["specific_problem"] = self._short_text(common_cause.get("common_failure_reason", ""), 420)
            if common_cause.get("required_new_capability"):
                cluster["repair_need"] = self._short_text(common_cause.get("required_new_capability", ""), 360)
                cluster["repair_hint"] = self._short_text(common_cause.get("required_new_capability", ""), 260)
            try:
                candidates = self._generate_patch_candidates_for_failure_cluster(engine, cluster)
                critic_rows = [
                    self._critic_and_repair_patch_candidate(engine, candidate, cluster, forbidden_terms)
                    for candidate in list(candidates or [])
                ]
                selected = self._select_best_patch_candidate(critic_rows)
            except Exception as exc:
                import traceback as _traceback

                error_row = {
                    "event": "tree_patch_generation_failed",
                    "ts": int(time.time()),
                    "cluster_id": str(cluster.get("cluster_id", "") or ""),
                    "support": int(cluster.get("support", 0) or 0),
                    "support_user_ids": list(cluster.get("support_user_ids", []) or [])[:50],
                    "layer_hint": str(cluster.get("layer_hint", "") or ""),
                    "reference_node": str(cluster.get("reference_node", "") or ""),
                    "common_cause_analysis": dict(common_cause or {}),
                    "error": str(exc),
                    "traceback": _traceback.format_exc()[-4000:],
                }
                patch_generation_errors.append(error_row)
                append_jsonl(engine.public_tree_store.index_dir / "tree_batch_errors.jsonl", error_row)
                summaries.append(f"{cluster.get('cluster_id', '')}: patch generation failed: {exc}")
                continue
            if selected:
                selected["_source_cluster"] = str(cluster.get("cluster_id", "") or "")
                selected["_route_injection_cluster"] = self._cluster_route_injection_packet(cluster)
                patches.append(selected)
                summaries.append(f"{cluster.get('cluster_id', '')}: selected {selected.get('layer')}/{selected.get('new_node_id')}")
            else:
                summaries.append(f"{cluster.get('cluster_id', '')}: no valid patch candidate")
        return {
            "summary": " | ".join([x for x in summaries if x])[:1200],
            "patches": patches,
            "reused_nodes": reused_nodes,
            "patch_generation_errors": patch_generation_errors,
            "problem_signals": [
                {
                    "signal": str(row.get("cluster_id", "") or ""),
                    "support": int(row.get("support", 0) or 0),
                    "severity": float(row.get("severity", 0.0) or 0.0),
                    "outcomes": {str(row.get("session_outcome", "") or ""): int(row.get("support", 0) or 0)},
                    "failure_levels": {str(row.get("primary_failure_stage", "") or row.get("failed_stage", "") or ""): int(row.get("support", 0) or 0)},
                    "mechanism": str(row.get("primary_failure_mechanism", "") or ""),
                }
                for row in ready_clusters[:8]
            ],
            "failure_clusters": [
                {
                    "cluster_id": str(row.get("cluster_id", "") or ""),
                    "cluster_key_version": str(row.get("cluster_key_version", "") or ""),
                    "support": int(row.get("support", 0) or 0),
                    "support_user_ids": list(row.get("support_user_ids", []) or [])[:50],
                    "unique_users": int(row.get("unique_users", 0) or 0),
                    "severity": float(row.get("severity", 0.0) or 0.0),
                    "ready": bool(row.get("ready", False)),
                    "problem_area": str(row.get("problem_area", "") or ""),
                    "material_signature": list(row.get("material_signature", []) or []),
                    "evidence_strength": str(row.get("evidence_strength", "") or ""),
                    "primary_failure_stage": str(row.get("primary_failure_stage", "") or ""),
                    "primary_failure_mechanism": str(row.get("primary_failure_mechanism", "") or ""),
                    "repair_need": str(row.get("repair_need", "") or ""),
                    "quality_issues": dict(row.get("quality_issues", {}) or {}),
                    "common_cause_analysis": dict(
                        row.get("common_cause_analysis", {})
                        or common_cause_by_cluster.get(str(row.get("cluster_id", "") or ""), {})
                        or {}
                    ),
                }
                for row in clusters[:8]
            ],
        }

    def _cluster_route_injection_packet(self, cluster):
        cluster = dict(cluster or {})
        return self._json_safe(
            {
                "dataset": str(cluster.get("dataset", "") or cluster.get("source_dataset", "") or ""),
                "cluster_id": str(cluster.get("cluster_id", "") or ""),
                "support_user_ids": list(cluster.get("support_user_ids", []) or []),
                "support_user_paths": dict(cluster.get("support_user_paths", {}) or {}),
                "layer": str(cluster.get("layer_hint", "") or ""),
                "reference_node": str(cluster.get("reference_node", "") or ""),
                "trigger_signature_distribution": dict(cluster.get("trigger_signature_distribution", {}) or {}),
                "route_selection_patterns": dict(cluster.get("route_selection_patterns", {}) or {}),
                "failure_type": str(cluster.get("failure_type", "") or ""),
                "failed_stage": str(cluster.get("failed_stage", "") or ""),
                "primary_failure_mechanism": str(cluster.get("primary_failure_mechanism", "") or ""),
                "repair_need": self._short_text(cluster.get("repair_need", "") or cluster.get("repair_hint", ""), 360),
            }
        )

    @staticmethod
    def _dominant_counter_key(counter_like, default=""):
        rows = dict(counter_like or {})
        if not rows:
            return str(default or "")
        return str(sorted(rows.items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0])))[0][0] or default or "")

    def _fallback_route_scope_for_cluster(self, cluster_packet, layer, path):
        layer = str(layer or "")
        path = dict(path or {})
        if layer == "what":
            return (
                str(path.get('why', "") or "")
                or self._dominant_counter_key(cluster_packet.get("trigger_signature_distribution", {}), "default")
                or "default"
            )
        if layer == "how":
            return str(path.get("what", "") or "") or "default"
        if layer == "who":
            return str(path.get("how", "") or "") or "default"
        return "default"

    def _route_injection_operation_for_patch(self, patch, result, cluster_packet, user_id):
        patch = dict(patch or {})
        result = dict(result or {})
        cluster_packet = dict(cluster_packet or {})
        layer = str(patch.get("layer", "") or cluster_packet.get("layer", "") or "").strip()
        if layer not in {"what", "how", "who"}:
            return None
        node = str(result.get("node_id", "") or patch.get("new_node_id", "") or "").strip()
        if not node:
            return None
        user_paths = dict(cluster_packet.get("support_user_paths", {}) or {})
        path = dict(user_paths.get(str(user_id), {}) or {})
        scope = self._route_scope(path, layer) if path else self._fallback_route_scope_for_cluster(cluster_packet, layer, path)
        before = str(patch.get("parent_node", "") or "").strip()
        if not before:
            before = str(
                patch.get("trial_anchor_node", "")
                or cluster_packet.get("reference_node", "")
                or ""
            ).strip()
        if not before and path:
            before = str(path.get(layer, "") or "").strip()
        return {
            "operation": "insert_before",
            "level": layer,
            "scope": scope or "default",
            "node": node,
            "before": before,
            "reason": (
                "cluster_active_tree_node_injection:"
                f"{cluster_packet.get('cluster_id', '')}"
            ),
        }

    def _inject_active_tree_node_into_route_skills(self, engine, patch, result):
        if not bool((result or {}).get("applied", False)):
            return []
        if str(getattr(engine.args, "run_stage", "train") or "train").strip().lower() != "train":
            return []
        patch = dict(patch or {})
        if str(patch.get("status", "") or "active") != "active":
            return []
        cluster_packet = dict(patch.get("_route_injection_cluster", {}) or {})
        user_ids = [str(x) for x in list(cluster_packet.get("support_user_ids", []) or []) if str(x).strip()]
        if not user_ids:
            return []
        store = getattr(engine, "user_policy_store", None)
        if store is None:
            return []
        events = []
        for user_id in user_ids:
            op = self._route_injection_operation_for_patch(patch, result, cluster_packet, user_id)
            if not op:
                continue
            try:
                policy, source = store.load_full_policy(user_id, stage="train")
                if hasattr(engine, "_sanitize_policy_for_dataset_tree"):
                    policy = engine._sanitize_policy_for_dataset_tree(
                        user_raw=user_id,
                        policy=policy,
                        policy_source=source,
                        stage="train",
                        history_summary="",
                    )
                updated, route = self._ensure_route_skill(policy)
                route, changed = self._apply_route_operation(route, op)
                if not changed:
                    continue
                updated["communication_route_skill"] = route
                store.save_full_policy(updated, snapshot_reason="tree_cluster_route_injection")
                event = {
                    "event": "tree_cluster_route_injection",
                    "ts": int(time.time()),
                    "dataset": self._dataset_slug(engine),
                    "user_id": user_id,
                    "policy_source": str(source or ""),
                    "cluster_id": str(cluster_packet.get("cluster_id", "") or ""),
                    "node_id": str(result.get("node_id", "") or ""),
                    "layer": str(op.get("level", "") or ""),
                    "scope": str(op.get("scope", "") or ""),
                    "before": str(op.get("before", "") or ""),
                    "operation": op,
                }
                store.append_evolution_log(user_id, event)
                events.append(event)
            except Exception as exc:
                events.append(
                    {
                        "event": "tree_cluster_route_injection_failed",
                        "ts": int(time.time()),
                        "user_id": user_id,
                        "cluster_id": str(cluster_packet.get("cluster_id", "") or ""),
                        "node_id": str(result.get("node_id", "") or ""),
                        "reason": str(exc),
                    }
                )
        if events:
            append_jsonl(
                engine.public_tree_store.index_dir / "tree_route_injections.jsonl",
                {
                    "event": "tree_route_injection_batch",
                    "ts": int(time.time()),
                    "dataset": self._dataset_slug(engine),
                    "cluster_id": str(cluster_packet.get("cluster_id", "") or ""),
                    "node_id": str(result.get("node_id", "") or ""),
                    "attempted_users": len(user_ids),
                    "changed_users": sum(1 for row in events if str(row.get("event", "")) == "tree_cluster_route_injection"),
                    "events": events[:200],
                },
            )
        return events

    def _route_contains_injected_node(self, route, level, scope, node):
        route = dict(route or {})
        level = str(level or "").strip().lower()
        scope = str(scope or "").strip()
        node = str(node or "").strip()
        if "|" in scope:
            parts = [x for x in scope.split("|") if x]
            if level == "what" and parts:
                scope = parts[0]
            elif level in ["how", "who"] and parts:
                scope = parts[-1]
        bucket = {
            "what": "what_by_why",
            "how": "how_by_what",
            "who": "who_by_how",
        }.get(level)
        if not bucket or not scope or not node:
            return False
        values = (route.get(bucket, {}) or {}).get(scope, [])
        values = values if isinstance(values, list) else [values]
        return node in [str(x) for x in values]

    def repair_active_tree_route_injections(self, engine):
        index_path = engine.public_tree_store.index_dir / "tree_route_injections.jsonl"
        if not index_path.exists():
            return {"checked": 0, "repaired": 0, "reason": "no_tree_route_injections_log"}
        try:
            tree = engine.public_tree_store.load_tree(force_reload=True)
        except Exception:
            tree = {}
        store = getattr(engine, "user_policy_store", None)
        if store is None:
            return {"checked": 0, "repaired": 0, "reason": "missing_user_policy_store"}
        checked = 0
        already_present = 0
        repaired = 0
        skipped_inactive = 0
        failed = 0
        seen = set()
        repair_events = []
        for batch in load_jsonl(index_path):
            for event in list((batch or {}).get("events", []) or []):
                if str((event or {}).get("event", "") or "") != "tree_cluster_route_injection":
                    continue
                op = dict((event or {}).get("operation", {}) or {})
                user_id = str((event or {}).get("user_id", "") or "").strip()
                level = str(op.get("level", "") or (event or {}).get("layer", "") or "").strip().lower()
                scope = str(op.get("scope", "") or (event or {}).get("scope", "") or "").strip()
                node = str(op.get("node", "") or (event or {}).get("node_id", "") or "").strip()
                if level not in {"what", "how", "who"} or not user_id or not node:
                    continue
                key = "|".join([user_id, level, scope, node])
                if key in seen:
                    continue
                seen.add(key)
                checked += 1
                node_status = str(((tree.get(level, {}) or {}).get(node, {}) or {}).get("status", "") or "")
                if node_status != "active":
                    skipped_inactive += 1
                    continue
                try:
                    policy, source = store.load_full_policy(user_id, stage="train")
                    if hasattr(engine, "_sanitize_policy_for_dataset_tree"):
                        policy = engine._sanitize_policy_for_dataset_tree(
                            user_raw=user_id,
                            policy=policy,
                            policy_source=source,
                            stage="train",
                            history_summary="",
                        )
                    updated, route = self._ensure_route_skill(policy)
                    if self._route_contains_injected_node(route, level, scope, node):
                        already_present += 1
                        continue
                    if not op:
                        op = {
                            "operation": "insert_before",
                            "level": level,
                            "scope": scope or "default",
                            "node": node,
                            "before": str((event or {}).get("before", "") or ""),
                            "reason": "tree_route_injection_repair",
                        }
                    route, changed = self._apply_route_operation(route, op)
                    if not changed:
                        failed += 1
                        continue
                    updated["communication_route_skill"] = route
                    store.save_full_policy(updated, snapshot_reason="tree_route_injection_repair")
                    repaired += 1
                    repair_events.append(
                        {
                            "event": "tree_route_injection_repaired",
                            "ts": int(time.time()),
                            "dataset": self._dataset_slug(engine),
                            "user_id": user_id,
                            "node_id": node,
                            "layer": level,
                            "scope": scope,
                            "operation": op,
                        }
                    )
                except Exception as exc:
                    failed += 1
                    repair_events.append(
                        {
                            "event": "tree_route_injection_repair_failed",
                            "ts": int(time.time()),
                            "dataset": self._dataset_slug(engine),
                            "user_id": user_id,
                            "node_id": node,
                            "layer": level,
                            "scope": scope,
                            "reason": str(exc),
                        }
                    )
        summary = {
            "checked": int(checked),
            "already_present": int(already_present),
            "repaired": int(repaired),
            "skipped_inactive": int(skipped_inactive),
            "failed": int(failed),
        }
        if repair_events:
            append_jsonl(
                engine.public_tree_store.index_dir / "tree_route_injection_repairs.jsonl",
                {"event": "tree_route_injection_repair_batch", "ts": int(time.time()), **summary, "events": repair_events[:500]},
            )
        return summary

    def _apply_tree_patch_proposals(self, engine, proposal, rows):
        proposal = dict(proposal or {})
        forbidden_terms = self._tree_patch_forbidden_terms(rows)
        applied = []
        rejected = []
        for reused in list(proposal.get("reused_nodes", []) or []):
            reused = dict(reused or {})
            result = {
                "applied": True,
                "operation": "reuse_existing_node",
                "node_id": str(reused.get("node_id", "") or reused.get("new_node_id", "") or ""),
            }
            route_injections = self._inject_active_tree_node_into_route_skills(engine, reused, result)
            row = {
                "patch": reused,
                "result": result,
                "reused_existing_node": True,
                "route_injections": route_injections[:12],
                "route_injection_count": sum(
                    1 for event in route_injections
                    if str((event or {}).get("event", "")) == "tree_cluster_route_injection"
                ),
            }
            applied.append(row)
        for patch in list(proposal.get("patches", []) or []):
            result = engine.public_tree_store.apply_tree_patch(
                patch,
                forbidden_terms=forbidden_terms,
                source="llm_tree_patch",
            )
            row = {"patch": patch, "result": result}
            if result.get("applied"):
                route_injections = self._inject_active_tree_node_into_route_skills(engine, patch, result)
                if route_injections:
                    row["route_injections"] = route_injections[:12]
                    row["route_injection_count"] = sum(
                        1 for event in route_injections
                        if str((event or {}).get("event", "")) == "tree_cluster_route_injection"
                    )
                applied.append(row)
            else:
                rejected.append(row)
        for prompt_io in list(proposal.get("_prompt_ios", []) or []):
            prompt_io = dict(prompt_io or {})
            signal = str(prompt_io.get("signal", "") or "")
            related_results = [
                row for row in applied + rejected
                if signal and str((row.get("patch", {}) or {}).get("_source_signal", "") or "") == signal
            ]
            prompt_io["validation_result"] = related_results[:6]
            prompt_io["applied_patch_count"] = sum(1 for row in related_results if (row.get("result", {}) or {}).get("applied"))
            append_jsonl(engine.public_tree_store.index_dir / "tree_patch_prompt_io.jsonl", prompt_io)
        append_jsonl(
            engine.public_tree_store.index_dir / "evolution_log.jsonl",
            {
                "event": "communication_tree_llm_patch",
                "ts": int(time.time()),
                "summary": str(proposal.get("summary", "") or ""),
                "problem_signals": list(proposal.get("problem_signals", []) or [])[:8],
                "failure_clusters": list(proposal.get("failure_clusters", []) or [])[:8],
                "reused_node_count": len(list(proposal.get("reused_nodes", []) or [])),
                "patch_count": len(list(proposal.get("patches", []) or [])),
                "applied_count": len(applied),
                "rejected_count": len(rejected),
                "route_injection_count": sum(int(row.get("route_injection_count", 0) or 0) for row in applied),
                "applied": applied[:8],
                "rejected": rejected[:8],
            },
        )
        return {
            "summary": str(proposal.get("summary", "") or ""),
            "patch_count": len(list(proposal.get("patches", []) or [])),
            "reused_node_count": len(list(proposal.get("reused_nodes", []) or [])),
            "applied_count": len(applied),
            "rejected_count": len(rejected),
            "route_injection_count": sum(int(row.get("route_injection_count", 0) or 0) for row in applied),
            "applied_nodes": [str((x.get("result", {}) or {}).get("node_id", "")) for x in applied],
            "failure_clusters": list(proposal.get("failure_clusters", []) or [])[:8],
        }

    @staticmethod
    def _parent_node_id(node_id):
        node_id = str(node_id or "").strip()
        return "/".join(node_id.split("/")[:-1]) if "/" in node_id else ""

    @staticmethod
    def _sprout_trial_key(layer, node_id):
        return f"{str(layer or '')}/{str(node_id or '')}"

    @staticmethod
    def _is_severe_quality_issue(issue):
        issue = str(issue or "").lower()
        return any(
            token in issue
            for token in [
                "protocol",
                "advisor_not",
                "feedback",
                "evidence_lost",
                "pool_empty",
                "missing_advisor",
                "silent_focus",
                "no_candidate",
            ]
        )

    def _trial_effect_for_context(self, session_outcome, ctx, last_round_id):
        session_outcome = str(session_outcome or "")
        if session_outcome in ["TW", "WW"]:
            try:
                return self._round_effect(session_outcome, ctx, int(ctx.get("round_id", 0) or 0) == int(last_round_id or 0))
            except Exception:
                return "harmful" if session_outcome == "TW" else "ineffective"
        if session_outcome == "WT":
            return "helpful"
        if session_outcome == "TT":
            return "neutral"
        return "neutral"

    def _sprout_nodes_in_path(self, engine, path):
        tree = engine.public_tree_store.load_tree(force_reload=True)
        out = []
        for level in ['why', "what", "who", "how"]:
            node_id = str((path or {}).get(level, "") or "")
            node = dict((tree.get(level, {}) or {}).get(node_id, {}) or {})
            if node and str(node.get("status", "") or "") == "sprout":
                out.append((level, node_id, node))
        who_branch = str((path or {}).get("who_branch", "") or "")
        if who_branch:
            node = dict((tree.get("who", {}) or {}).get(who_branch, {}) or {})
            if node and str(node.get("status", "") or "") == "sprout":
                out.append(("who", who_branch, node))
        return out

    def _trial_advisor_summary(self, execution_packet):
        rows = []
        for fb in list((execution_packet or {}).get("advisor_feedbacks", []) or [])[:4]:
            if not isinstance(fb, dict):
                continue
            rows.append(
                f"{fb.get('advisor_id', fb.get('advisor', ''))}: "
                f"{fb.get('stance', fb.get('advice', ''))} | "
                f"{fb.get('endorsed_item', fb.get('suggested_item', fb.get('evidence_item', '')))} | "
                f"{self._short_text(fb.get('task_answer', fb.get('reason', fb.get('raw_text', ''))), 140)}"
            )
        return self._short_text(" ; ".join([x for x in rows if x.strip(" ;|")]), 700)

    def _trial_candidate_evidence_summary(self, execution_packet):
        committee = dict((execution_packet or {}).get("committee_packet", {}) or {})
        summary = committee.get("candidate_evidence_text") or committee.get("evidence_summary") or committee.get("discussion_summary") or ""
        return self._short_text(summary, 700)

    def _trial_user_redecision_summary(self, ctx):
        packet = dict((ctx or {}).get("redecision_packet", {}) or {})
        return self._short_text(
            packet.get("revised_reason", "")
            or packet.get("reason", "")
            or packet.get("feedback_to_advisors", "")
            or packet.get("prompt_decision_context", ""),
            500,
        )

    def _update_lifecycle_with_trial(self, engine, layer, node_id, record):
        node_dir = engine.public_tree_store._node_dir_for_id(layer, node_id)
        lifecycle_path = node_dir / "references" / "lifecycle.json"
        lifecycle = load_json(lifecycle_path, default={}) or {}
        lifecycle.setdefault("status", "sprout")
        lifecycle.setdefault("parent_node", self._parent_node_id(node_id))
        lifecycle["trial_count"] = int(lifecycle.get("trial_count", lifecycle.get("trial_rounds", 0)) or 0) + 1
        lifecycle["trial_rounds"] = int(lifecycle.get("trial_rounds", 0) or 0) + 1
        outcome = str(record.get("session_outcome", "") or "").lower()
        if outcome in ["tt", "wt", "tw", "ww"]:
            lifecycle[outcome] = int(lifecycle.get(outcome, 0) or 0) + 1
        lifecycle["useful_final_t_count"] = int(lifecycle.get("tt", 0) or 0) + int(lifecycle.get("wt", 0) or 0)
        effect = str(record.get("round_effect", "") or "")
        if effect == "helpful":
            lifecycle["helpful_count"] = int(lifecycle.get("helpful_count", 0) or 0) + 1
        elif effect == "harmful":
            lifecycle["harmful_count"] = int(lifecycle.get("harmful_count", 0) or 0) + 1
        elif effect == "ineffective":
            lifecycle["ineffective_count"] = int(lifecycle.get("ineffective_count", 0) or 0) + 1
        else:
            lifecycle["neutral_count"] = int(lifecycle.get("neutral_count", 0) or 0) + 1
        pattern = str(record.get("success_pattern", "") or "")
        if pattern:
            rows = list(lifecycle.get("success_patterns", []) or [])
            if pattern not in rows:
                rows.append(pattern)
            lifecycle["success_patterns"] = rows[-12:]
        severe = sum(1 for issue in list(record.get("quality_issues", []) or []) if self._is_severe_quality_issue(issue))
        lifecycle["severe_quality_issue_count"] = int(lifecycle.get("severe_quality_issue_count", 0) or 0) + int(severe)
        if effect in ["harmful", "ineffective", "uncertain_failure"]:
            failure_pattern = str(record.get("failure_pattern", "") or record.get("why_failed", "") or "")
            if failure_pattern:
                patterns = list(lifecycle.get("failure_patterns", []) or [])
                if failure_pattern not in patterns:
                    patterns.append(failure_pattern)
                lifecycle["failure_patterns"] = patterns[-12:]
            repair_hint = str(record.get("repair_hint", "") or "")
            if repair_hint:
                hints = list(lifecycle.get("repair_hints", []) or [])
                if repair_hint not in hints:
                    hints.append(repair_hint)
                lifecycle["repair_hints"] = hints[-12:]
            lifecycle.setdefault("revision_memory", []).append(
                {
                    "ts": int(time.time()),
                    "round_effect": effect,
                    "failure_type": str(record.get("failure_type", "") or ""),
                    "failed_stage": str(record.get("failed_stage", "") or ""),
                    "failure_pattern": failure_pattern,
                    "repair_hint": repair_hint,
                    "quality_issues": list(record.get("quality_issues", []) or [])[:8],
                }
            )
            lifecycle["revision_memory"] = list(lifecycle.get("revision_memory", []) or [])[-20:]
        dump_json(lifecycle_path, lifecycle)

    def _update_sprout_stats_file(self, engine, layer, node_id, record):
        stats_path = engine.public_tree_store.index_dir / "sprout_trial_stats.json"
        stats = load_json(stats_path, default={}) or {}
        key = self._sprout_trial_key(layer, node_id)
        row = dict(stats.get(key, {}) or {})
        row.setdefault("node_id", node_id)
        row.setdefault("layer", layer)
        row.setdefault("parent_node", self._parent_node_id(node_id))
        row["support"] = int(row.get("support", 0) or 0) + 1
        outcomes = Counter(row.get("outcomes", {}) or {})
        outcomes[str(record.get("session_outcome", "") or "")] += 1
        row["outcomes"] = dict(outcomes)
        effects = Counter(row.get("round_effects", {}) or {})
        effects[str(record.get("round_effect", "") or "")] += 1
        row["round_effects"] = dict(effects)
        quality = Counter(row.get("quality_issues", {}) or {})
        for issue in list(record.get("quality_issues", []) or []):
            if str(issue).strip():
                quality[str(issue).strip()] += 1
        row["quality_issues"] = dict(quality)
        row["helpful_count"] = int(effects.get("helpful", 0) or 0)
        row["harmful_count"] = int(effects.get("harmful", 0) or 0)
        row["ineffective_count"] = int(effects.get("ineffective", 0) or 0)
        row["neutral_count"] = int(effects.get("neutral", 0) or 0)
        row["useful_final_t_count"] = int(outcomes.get("WT", 0) or 0) + int(outcomes.get("TT", 0) or 0)
        row["non_harmful_count"] = int(row["helpful_count"] + row["neutral_count"])
        row["severe_quality_issue_count"] = sum(
            count for issue, count in quality.items() if self._is_severe_quality_issue(issue)
        )
        row.setdefault("failure_patterns", [])
        row.setdefault("repair_hints", [])
        if record.get("failure_pattern"):
            row["failure_patterns"] = (list(row.get("failure_patterns", []) or []) + [str(record.get("failure_pattern"))])[-12:]
        if record.get("repair_hint"):
            row["repair_hints"] = (list(row.get("repair_hints", []) or []) + [str(record.get("repair_hint"))])[-12:]
        stats[key] = row
        dump_json(stats_path, stats)

    def _record_sprout_trials(self, engine, tree_diagnosis):
        if not bool(getattr(engine.args, "com_enable_tree_trial_exploration", False)):
            return []
        if str(getattr(engine.args, "run_stage", "train") or "train").lower() != "train":
            return []
        row = dict(tree_diagnosis or {})
        contexts = list(row.get("round_trace_contexts", []) or [])
        if not contexts:
            contexts = [
                {
                    "round_id": 1,
                    "path": dict(row.get("path", {}) or {}),
                    "execution_packet": {},
                    "evaluation_result": {"outcome_signal": str(row.get("outcome_signal", "") or "")},
                }
            ]
        last_round_id = max([int((ctx or {}).get("round_id", idx + 1) or idx + 1) for idx, ctx in enumerate(contexts)] or [1])
        session_outcome = str(row.get("outcome_signal", "") or "")
        records = []
        for idx, ctx in enumerate(contexts, start=1):
            ctx = dict(ctx or {})
            path = dict(ctx.get("path", {}) or row.get("path", {}) or {})
            sprout_nodes = self._sprout_nodes_in_path(engine, path)
            if not sprout_nodes:
                continue
            effect = self._trial_effect_for_context(session_outcome, ctx, last_round_id)
            try:
                quality_issues = self.analyzer._communication_quality_issues(ctx)
            except Exception:
                quality_issues = list(row.get("communication_quality_issues", []) or [])
            failed_level = str(row.get("failed_level", "") or "")
            failure_type = self._failure_type_from_round(row, ctx, quality_issues, failed_level, session_outcome) if effect in ["harmful", "ineffective", "uncertain_failure"] else ""
            failed_stage = self._failed_stage_for_type(failure_type) if failure_type else ""
            execution = dict(ctx.get("execution_packet", {}) or {})
            user_task = str(execution.get("task") or execution.get("user_task") or path.get("user_task", "") or "")
            advisor_summary = self._trial_advisor_summary(execution)
            evidence_summary = self._trial_candidate_evidence_summary(execution)
            redecision_summary = self._trial_user_redecision_summary(ctx)
            why_failed = self._short_text(
                str(row.get("failure_reason", "") or "")
                or ("; ".join([str(x) for x in quality_issues[:5]]) if quality_issues else failure_type),
                420,
            )
            repair_hint = self._short_text(
                f"Revise the node so it handles {failure_type or 'the observed trial'} without losing UserTask, CandidateView, or previous discussion memory.",
                300,
            )
            success_pattern = self._short_text(
                f"{session_outcome} with {path.get('what', '')}/{path.get('how', '')}: advisor evidence helped or preserved the final decision.",
                260,
            ) if session_outcome in ["WT", "TT"] else ""
            for layer, node_id, node in sprout_nodes:
                record = {
                    "event": "sprout_trial",
                    "ts": int(time.time()),
                    "node_id": node_id,
                    "parent_node": self._parent_node_id(node_id),
                    "layer": layer,
                    "status_at_trial": str(node.get("status", "") or "sprout"),
                    "trial_path": path,
                    "round_index": int(ctx.get("round_id", idx) or idx),
                    "session_outcome": session_outcome,
                    "round_effect": effect,
                    "failure_type": failure_type,
                    "failed_stage": failed_stage,
                    "quality_issues": list(quality_issues or []),
                    "user_task": self._short_text(user_task, 320),
                    "advisor_summary": advisor_summary,
                    "candidate_evidence_summary": evidence_summary,
                    "user_redecision_summary": redecision_summary,
                    "why_failed": why_failed if effect != "helpful" else "",
                    "failure_pattern": why_failed if effect != "helpful" else "",
                    "repair_hint": repair_hint if effect != "helpful" else "",
                    "success_pattern": success_pattern,
                }
                append_jsonl(engine.public_tree_store.index_dir / "sprout_trial_records.jsonl", record)
                self._update_sprout_stats_file(engine, layer, node_id, record)
                self._update_lifecycle_with_trial(engine, layer, node_id, record)
                records.append(record)
        return records

    def _update_sprout_lifecycle(self, engine, node_stats):
        if not bool(getattr(engine.args, "com_enable_tree_trial_exploration", False)):
            return []
        tree = engine.public_tree_store.load_tree(force_reload=True)
        updates = []
        sprout_trial_stats = load_json(engine.public_tree_store.index_dir / "sprout_trial_stats.json", default={}) or {}
        for level in ['why', "what", "who", "how"]:
            for node_id, node in dict(tree.get(level, {}) or {}).items():
                if str(node.get("status", "") or "") != "sprout":
                    continue
                stat = dict(sprout_trial_stats.get(f"{level}/{node_id}", {}) or {})
                if not stat:
                    continue
                support = int(stat.get("support", 0) or 0)
                harmful = int(stat.get("harmful_count", 0) or 0)
                helpful = int(stat.get("helpful_count", 0) or 0)
                ineffective = int(stat.get("ineffective_count", 0) or 0)
                severe_quality = int(stat.get("severe_quality_issue_count", 0) or 0)
                outcomes = Counter(stat.get("outcomes", {}) or {})
                useful_final_t = int(
                    stat.get("useful_final_t_count", outcomes.get("WT", 0) + outcomes.get("TT", 0)) or 0
                )
                quality = Counter(stat.get("quality_issues", {}) or {})
                repeated_failure = any(
                    int(count or 0) >= 2 and self._is_severe_quality_issue(issue)
                    for issue, count in quality.items()
                )
                new_status = ""
                reason = ""
                if useful_final_t >= 2:
                    new_status = "active"
                    reason = "sprout trial met promotion criteria: used node reached final target outcome twice (WT or TT)"
                elif harmful >= 1 or ineffective >= 2 or repeated_failure:
                    new_status = "withered"
                    reason = "sprout trial met wither criteria: harmful, ineffective, or repeated severe communication failure"
                if not new_status:
                    continue
                result = engine.public_tree_store.apply_tree_patch(
                    {
                        "operation": "mark_withered",
                        "layer": level,
                        "node_id": node_id,
                        "new_status": "withered",
                        "why_withered": reason,
                        "failure_pattern": "; ".join(list(stat.get("failure_patterns", []) or [])[:3]),
                        "repair_hint": "; ".join(list(stat.get("repair_hints", []) or [])[:3]),
                        "replacement_hint": "generate revised sprout sibling from lifecycle revision_memory",
                    },
                    forbidden_terms=[],
                    source="lifecycle_auto_update",
                ) if new_status == "withered" else self._promote_tree_node(engine, level, node_id, reason)
                updates.append({
                    "level": level,
                    "node_id": node_id,
                    "to": new_status,
                    "reason": reason,
                    "support": support,
                    "useful_final_t_count": useful_final_t,
                    "helpful_count": helpful,
                    "result": result,
                })
        dump_json(engine.public_tree_store.index_dir / "sprout_trial_stats.json", sprout_trial_stats)
        if updates:
            append_jsonl(
                engine.public_tree_store.index_dir / "lifecycle_updates.jsonl",
                {
                    "event": "sprout_lifecycle_batch_update",
                    "ts": int(time.time()),
                    "updates": updates,
                },
            )
        return updates

    def _promote_tree_node(self, engine, layer, node_id, reason):
        node_dir = engine.public_tree_store._node_dir_for_id(layer, node_id)
        if node_dir is None:
            return {"applied": False, "reason": "invalid node"}
        spec_path = node_dir / "references" / "skill.json"
        spec = load_json(spec_path, default={}) or {}
        spec["status"] = "active"
        spec["selection_prior"] = max(0.3, float(spec.get("selection_prior", 0.15) or 0.15))
        dump_json(spec_path, spec)
        lifecycle_path = node_dir / "references" / "lifecycle.json"
        lifecycle = load_json(lifecycle_path, default={}) or {}
        previous = str(lifecycle.get("status", "sprout") or "sprout")
        lifecycle["status"] = "active"
        lifecycle.setdefault("promotion_history", []).append(
            {"ts": int(time.time()), "from": previous, "to": "active", "reason": str(reason or "")}
        )
        dump_json(lifecycle_path, lifecycle)
        skill_path = node_dir / "SKILL.md"
        body = safe_read_text(skill_path, default="")
        if body:
            body = re.sub(r"(?m)^status:\s*\S+\s*$", "status: active", body, count=1)
            body = body.replace("This node is sprout and may only be trialed during train.", "This node is active and may be selected during normal path selection.")
            skill_path.write_text(body, encoding="utf-8")
        engine.public_tree_store._cache = None
        return {"applied": True, "operation": "promote_sprout", "node_id": node_id}

    def evolve_tree_batch(self, engine, stage="train"):
        buffer_path = self._tree_evolution_buffer_path(engine)
        rows = self._load_dataset_tree_evolution_buffer(engine)
        if not rows:
            lifecycle_updates = self._update_sprout_lifecycle(engine, {})
            route_injection_repair = self.repair_active_tree_route_injections(engine)
            return {
                "processed": 0,
                "message": "no tree evolution evidence",
                "dataset": self._dataset_slug(engine),
                "lifecycle_updates": list(lifecycle_updates or []),
                "route_injection_repair": dict(route_injection_repair or {}),
            }

        prefix_rows, path_rows, node_rows, fine_prefix_rows, fine_path_rows = self._aggregate_batch(rows)
        new_branch_stats = {key: self._stats_for_rows(value) for key, value in sorted(prefix_rows.items())}
        new_path_stats = {key: self._stats_for_rows(value) for key, value in sorted(path_rows.items())}
        new_node_stats = {key: self._stats_for_rows(value) for key, value in sorted(node_rows.items())}
        new_fine_branch_stats = {key: self._stats_for_rows(value) for key, value in sorted(fine_prefix_rows.items())}
        new_fine_path_stats = {key: self._stats_for_rows(value) for key, value in sorted(fine_path_rows.items())}
        branch_stats = self._merge_stat_maps(
            load_json(engine.public_tree_store.index_dir / "branch_stats.json", default={}) or {},
            new_branch_stats,
        )
        path_stats = self._merge_stat_maps(
            load_json(engine.public_tree_store.index_dir / "path_stats.json", default={}) or {},
            new_path_stats,
        )
        node_stats = self._merge_stat_maps(
            load_json(engine.public_tree_store.index_dir / "node_stats.json", default={}) or {},
            new_node_stats,
        )
        fine_branch_stats = self._merge_stat_maps(
            load_json(engine.public_tree_store.index_dir / "fine_branch_stats.json", default={}) or {},
            new_fine_branch_stats,
        )
        fine_path_stats = self._merge_stat_maps(
            load_json(engine.public_tree_store.index_dir / "fine_path_stats.json", default={}) or {},
            new_fine_path_stats,
        )
        risky_paths = {
            key: value
            for key, value in {**branch_stats, **fine_branch_stats}.items()
            if str(value.get("status", "")) == "risky"
        }

        dump_json(engine.public_tree_store.index_dir / "branch_stats.json", branch_stats)
        dump_json(engine.public_tree_store.index_dir / "path_stats.json", path_stats)
        dump_json(engine.public_tree_store.index_dir / "node_stats.json", node_stats)
        dump_json(engine.public_tree_store.index_dir / "fine_branch_stats.json", fine_branch_stats)
        dump_json(engine.public_tree_store.index_dir / "fine_path_stats.json", fine_path_stats)
        dump_json(engine.public_tree_store.index_dir / "risky_paths.json", risky_paths)
        lifecycle_updates = self._update_sprout_lifecycle(engine, node_stats)

        proposed_new = Counter(
            str(row.get("suggested_new_branch", "") or "")
            for row in rows
            if str(row.get("suggested_new_branch", "") or "")
        )
        for row in rows:
            for signal in list((row or {}).get("tree_need_signals", []) or []):
                if not isinstance(signal, dict):
                    continue
                level = str(signal.get("level", "") or "").strip()
                hint = str(signal.get("suggested_node_hint", "") or "").strip()
                if level and hint:
                    proposed_new[f"{level}/{hint}"] += 1
        for node_id, count in proposed_new.items():
            if count >= 1:
                append_jsonl(
                    engine.public_tree_store.index_dir / "node_proposals.jsonl",
                    {
                        "event": "batch_candidate_branch",
                        "node_id": node_id,
                        "support": int(count),
                        "status": "candidate",
                        "stage": str(stage or "train"),
                    },
                )

        llm_patch_update = {}
        if str(stage or "train") == "train":
            proposal = self._llm_generate_tree_patch_proposals(
                engine,
                rows,
                branch_stats,
                fine_branch_stats,
                path_stats,
                fine_path_stats,
                risky_paths,
                proposed_new,
                node_stats=node_stats,
            )
            llm_patch_update = self._apply_tree_patch_proposals(engine, proposal, rows)

        batch_id = int(time.time())
        append_jsonl(
            engine.public_tree_store.index_dir / "evolution_log.jsonl",
            {
                "event": "communication_tree_batch_update",
                "batch_id": batch_id,
                "dataset": self._dataset_slug(engine),
                "stage": str(stage or "train"),
                "processed": int(len(rows)),
                "branch_count": int(len(branch_stats)),
                "fine_branch_count": int(len(fine_branch_stats)),
                "path_count": int(len(path_stats)),
                "fine_path_count": int(len(fine_path_stats)),
                "risky_branch_count": int(len(risky_paths)),
                "proposed_new_branches": dict(proposed_new),
                "llm_patch_update": dict(llm_patch_update or {}),
                "lifecycle_updates": list(lifecycle_updates or []),
            },
        )

        archive_dir = engine.public_tree_store.index_dir / "processed_batches"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"tree_evolution_buffer_{self._dataset_slug(engine)}_{batch_id}.jsonl"
        with open(archive_path, "w", encoding="utf-8") as f:
            for row in rows:
                import json

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        Path(buffer_path).write_text("", encoding="utf-8")
        engine.public_tree = engine.public_tree_store.load_tree(force_reload=True)
        route_injection_repair = self.repair_active_tree_route_injections(engine)
        return {
            "processed": int(len(rows)),
            "branch_count": int(len(branch_stats)),
            "fine_branch_count": int(len(fine_branch_stats)),
            "path_count": int(len(path_stats)),
            "fine_path_count": int(len(fine_path_stats)),
            "risky_branch_count": int(len(risky_paths)),
            "archive_path": str(archive_path),
            "llm_patch_update": dict(llm_patch_update or {}),
            "lifecycle_updates": list(lifecycle_updates or []),
            "route_injection_repair": dict(route_injection_repair or {}),
        }
