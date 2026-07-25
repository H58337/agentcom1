from pathlib import Path
import re
import threading
import shutil
import time

from recommender.agent.COM.tree_engine.advisor_pool import AdvisorPoolManager
from recommender.agent.COM.tree_engine.disclosure import UserDisclosureBuilder
from recommender.agent.COM.tree_engine.evaluator import InteractionEvaluator
from recommender.agent.COM.tree_engine.evolver import TrainOnlyEvolver
from recommender.agent.COM.tree_engine.path_executor import PathExecutor
from recommender.agent.COM.tree_engine.path_selector import PublicTreePathSelector
from recommender.agent.COM.tree_engine.public_tree import PublicTreeStore, infer_communication_family
from recommender.agent.COM.tree_engine.redecision import ReDecisionMaker
from recommender.agent.COM.tree_engine.schemas import build_communication_path, build_decision_state
from recommender.agent.COM.tree_engine.task_planner import (
    generate_first_round_task,
    task_packet_from_feedback,
)
from recommender.agent.COM.tree_engine.trigger import infer_primary_trigger
from recommender.agent.COM.tree_engine.user_policy import UserPolicyStore
from recommender.agent.COM.tree_engine.utils import load_json, merge_unique
from recommender.agent.COM.utils.com_agent import (
    clear_llm_prompt_trace,
    llm_request,
    set_llm_prompt_trace,
    update_llm_prompt_trace_context,
)

_DISTILL_SYSTEM_PROMPT = (
    "You are a user preference profiler. Compress the following user skill data into a minimal structured profile "
    "for tie-breaking in item ranking.\n\n"
    "Extract:\n"
    "1. likes: preference themes most strongly distinguishing this user. Prefer domain-specific labels over generic descriptions.\n"
    "2. style: One phrase describing their overall preference pattern or decision style.\n"
    "3. rules: If there are reasoning rules, synthesize compact procedural keypoints across the relevant rules. "
    "Use the item-selection slots choose_by=..., preserve=..., minority=..., avoid_overbias=... when possible. "
    "Do not copy one stored rule verbatim. If none, write 'none'.\n\n"
    "Only include patterns with medium or high confidence. Skip vague entries like 'long-term repeated listening anchors' or 'mid-popularity affinity'.\n"
    "Do not output exact item names in likes. Item names are evidence only; convert them into transferable domain signals. "
    "Use the dataset-specific guidance below when converting evidence into transferable labels.\n"
    "Output format:\n"
    "likes: [theme1], [theme2], [theme3] | style: [decision style phrase] | rules: [choose_by=...; preserve=...; minority=...; avoid_overbias=... or none]"
)

_FULL_POLICY_RETRIEVAL_SYSTEM_PROMPT = (
    "You are the COM User Skill retriever. Your only source of skill knowledge is FullUserPolicy. "
    "Use CurrentContext only to decide which parts of the full skill are relevant for the current phase. "
    "Do not update the skill, invent new preferences, or use supervised target knowledge.\n\n"
    "Exact item names in FullUserPolicy or CurrentContext are evidence only. Convert them into transferable domain labels; never output exact current candidate, proposal, target, final, or prior-hint names in likes or rules. "
    "Use the dataset-specific guidance below when converting evidence into transferable labels.\n\n"
    "Treat weak_generic=true entries as weak tie-breakers only. Prefer concrete domain signals over popularity, name-shape, long-term-anchor, recent-anchor, statistical-cluster, or raw name-marker signals. "
    "Fields named risky_preferences or status=risky/evolution_weakened describe over-biases to avoid; they are not likes.\n"
    "If a stored rule names a current item, generalize it to a transferable style/cluster signal or ignore it.\n\n"
    "There are two phase-specific slim skills:\n"
    "1. item_selection/proposal: retrieve item-selection taste signals and procedural rules for choosing one item and preserving a hesitation shortlist. Focus on item_selection_skill preferences, recent_signals, active_rules, and risky_rules. Ignore advisor routing unless it directly affects item uncertainty.\n"
    "2. communication_selection: retrieve who/how/trigger/path/reliability skill for the abstract current condition. "
    "Do not retrieve a separate post-feedback skill; post-feedback item decisions reuse the item_selection slim skill.\n\n"
    "Prefer procedural rules that change behavior, especially 'when X, do Y' rules, recently evolved strategy rules, "
    "shortlist-preservation rules, and tie-break rules. Do not select a rule that merely repeats a like/preference "
    "already listed. Skip generic entries unless the current candidates make them directly relevant.\n"
    "For item_selection rules output, do not copy one stored rule verbatim. Synthesize compressed keypoints from "
    "multiple relevant active/risky rules when available. Use these exact slots when possible: choose_by=..., "
    "preserve=..., minority=..., avoid_overbias=.... Do not output primary=... because it over-strengthens the "
    "dominant cluster. preserve/minority must name concrete transferable style signals, not vague phrases like plausible candidates, diverse candidates, or generic exploration. For communication rules, use who=..., how=..., switch=..., avoid=....\n\n"
    "Output one compact line only. No JSON, no markdown.\n"
    "For item_selection/proposal, use:\n"
    "likes: <relevant domain preference signals, no exact item names> | style: <decision style> | rules: choose_by=<main evidence>; preserve=<specific non-favorite signals>; minority=<specific weak/minority signals>; avoid_overbias=<dominant-cluster risk>\n"
    "For communication_selection, use:\n"
    "condition: <abstract current uncertainty condition, no exact item names> | who: <advisor source preference> | how: <communication mode preference> | rules: <compressed communication keypoints>"
)


def _dataset_specific_prompt_guidance(args):
    dataset = str(getattr(args, "dataset", "") or "").lower()
    if "librarything" in dataset:
        return (
            "Dataset-specific guidance for LibraryThing book recommendation: "
            "use fiction/non-fiction genre, literary form, topic/subject, author style, narrative tone, "
            "era/setting, culture/language signal, audience/age category, series/franchise relation, "
            "adjacent theme bridge, canon/award/niche level, and recent reading drift. "
            "Do not translate book evidence into music/artist/listening labels."
        )
    if "epinions" in dataset:
        return (
            "Dataset-specific guidance for Epinions product recommendation: "
            "use product category, use case, brand/manufacturer family, feature/function, price/value, "
            "quality/durability, reliability, design/form factor, compatibility/accessory relation, "
            "review/rating sentiment, substitute/complement bridge, popularity/niche level, and recent need drift."
        )
    return (
        "Dataset-specific guidance for LastFM artist recommendation: "
        "use genre, scene, mood, era, energy, vocal/instrumental style, region/language signal, "
        "popularity level, co-listening cluster, and recent listening drift."
    )


def _distill_system_prompt(args):
    return f"{_DISTILL_SYSTEM_PROMPT}\n\n{_dataset_specific_prompt_guidance(args)}"


def _full_policy_retrieval_system_prompt(args):
    return f"{_FULL_POLICY_RETRIEVAL_SYSTEM_PROMPT}\n\n{_dataset_specific_prompt_guidance(args)}"


class ComTreeEngine:
    @staticmethod
    def _dataset_slug_value(dataset):
        dataset = str(dataset or "default").strip()
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset)
        return safe or "default"

    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.dataset = str(getattr(args, "dataset", "default") or "default")
        self.dataset_slug = self._dataset_slug_value(self.dataset)
        self.recommender_source = self._infer_recommender_source_from_prior_args(args)
        self.base_dir = Path(__file__).resolve().parents[1]
        self.public_tree_store = PublicTreeStore(
            self.base_dir / "public_tree" / self.dataset_slug,
            refresh_layout_on_load=bool(getattr(args, "com_refresh_public_tree_layout", False)),
            dataset=self.dataset_slug,
        )
        self.initial_skills_dir = self.base_dir / "initial_skills"
        initial_user_policy_dir = self.initial_skills_dir / "user_policy" if bool(getattr(args, "com_preserve_initial_skills", True)) else None
        self.user_policy_store = UserPolicyStore(
            self.base_dir / "user_policy",
            dataset=self.dataset,
            initial_base_dir=initial_user_policy_dir,
            recommender_source=self.recommender_source,
        )
        self.path_selector = PublicTreePathSelector(self.public_tree_store, args=self.args)
        self.advisor_pool = AdvisorPoolManager(self.args)
        self.path_executor = PathExecutor(self.args)
        self.disclosure_builder = UserDisclosureBuilder()
        self.redecision = ReDecisionMaker()
        self.evaluator = InteractionEvaluator()
        self.evolver = TrainOnlyEvolver()
        self.public_tree = self.public_tree_store.load_tree()
        self._evolution_lock = threading.Lock()
        self._skill_state_reset_done = False
        self._initial_public_tree_done = False
        self._rebuilt_user_policy_users = set()

    def _should_force_rebuild_user_policy(self, user_raw, stage):
        if str(stage or "").lower() != "train":
            return False
        if not bool(getattr(self.args, "com_rebuild_initial_user_policy", False)):
            return False
        key = str(user_raw)
        if key in self._rebuilt_user_policy_users:
            return False
        self._rebuilt_user_policy_users.add(key)
        return True

    @staticmethod
    def _infer_recommender_source_from_prior_args(args):
        for attr in ["com_prior_val_csv_path", "com_prior_csv_path"]:
            raw_path = str(getattr(args, attr, "") or "").strip()
            if not raw_path:
                continue
            stem = Path(raw_path).stem
            if stem.lower().endswith("_val"):
                stem = stem[:-4]
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
            if stem:
                return stem
        fallback = str(getattr(args, "tool_name", "") or "").strip()
        fallback = re.sub(r"[^A-Za-z0-9._-]+", "_", fallback).strip("._-")
        return fallback or "default_prior"

    def _distill_slim_policy(self, verbose_slim_policy, args=None):
        import json as _json
        import logging
        logger = logging.getLogger(__name__)
        llm_args = args or self.args
        try:
            user_prompt = _json.dumps(verbose_slim_policy, ensure_ascii=False)
        except Exception as e:
            logger.warning("[distill] JSON serialization failed: %s", e)
            return str(verbose_slim_policy)
        try:
            result = llm_request(_distill_system_prompt(llm_args), user_prompt, llm_args)
            if result and isinstance(result, str) and len(result.strip()) > 10:
                return result.strip()
            logger.warning("[distill] LLM returned empty/short result: %s", repr(result)[:200])
        except Exception as e:
            logger.warning("[distill] LLM call exception: %s", e)
        return str(verbose_slim_policy)

    @staticmethod
    def _weak_generic_skill_signal(text):
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

    @staticmethod
    def _rule_like_skill_text(text):
        low = " ".join(str(text or "").strip().lower().split())
        if not low:
            return False
        markers = [
            "when ",
            "if ",
            "evidence:",
            "reason:",
            "reinforce transferable item-selection signal",
            "add transferable item-selection signal",
            "weaken over-bias",
            "use this as a positive clue",
            "preserve future candidates",
            "before excluding",
            "compare that transferable signal",
            "do not let this bias",
        ]
        return any(marker in low for marker in markers)

    @staticmethod
    def _preference_explanation_rule(text):
        low = " ".join(str(text or "").strip().lower().split())
        if not low:
            return False
        prefixes = [
            "reinforce transferable item-selection signal",
            "add transferable item-selection signal",
            "weaken over-bias",
        ]
        if any(low.startswith(prefix) for prefix in prefixes):
            return True
        if ". evidence:" in low and not low.startswith("when "):
            return True
        return False

    @staticmethod
    def _actionable_strategy_rule(text):
        low = " ".join(str(text or "").strip().lower().split())
        if not low:
            return False
        strategy_markers = [
            "when ",
            "if ",
            "shortlist",
            "hesitation",
            "uncertain",
            "uncertainty",
            "near-miss",
            "tie breaker",
            "tie-breaker",
            "primary:",
            "secondary:",
            "ranking criteria",
            "preserve",
            "compare",
            "before excluding",
            "do not let",
            "prioritize",
            "choose_by=",
            "preserve=",
            "minority=",
            "avoid_overbias=",
        ]
        generic_preference_markers = [
            "give preference to artists with",
            "give preference to books with",
        ]
        has_strategy_marker = any(marker in low for marker in strategy_markers)
        is_generic_preference = any(marker in low for marker in generic_preference_markers)
        has_conditional_or_shortlist = any(marker in low for marker in ["when ", "if ", "shortlist", "hesitation", "uncertainty"])
        return has_strategy_marker and not (is_generic_preference and not has_conditional_or_shortlist)

    @staticmethod
    def _collect_current_item_names(decision_context):
        names = []

        def add(value):
            if isinstance(value, str):
                value = value.strip()
                if value:
                    names.append(value)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    add(item)
            elif isinstance(value, dict):
                for key in ["candidate", "item", "name", "proposal_item", "prior_hint"]:
                    if key in value:
                        add(value.get(key))

        ctx = dict(decision_context or {})
        for key in [
            "candidate_items", "prior_hint", "proposal_item", "final_item", "target_items",
            "hesitation_shortlist", "candidate_shortlist", "shortlist",
        ]:
            add(ctx.get(key))
        return sorted({x for x in names if len(x) >= 2}, key=len, reverse=True)

    @classmethod
    def _text_contains_current_item_name(cls, text, current_item_names):
        low = str(text or "").lower()
        for name in current_item_names or []:
            name = str(name or "").strip()
            if len(name) >= 2 and name.lower() in low:
                return True
        return False

    @classmethod
    def _compact_policy_for_skill_retrieval(cls, full_policy, decision_context=None):
        """Keep learned skill attributes, but remove history/evidence fields that make LLM copy artist names."""
        full_policy = dict(full_policy or {})
        current_item_names = cls._collect_current_item_names(decision_context)

        def compact_pref(row):
            row = dict(row or {})
            attribute = str(row.get("attribute", "") or row.get("rule", "") or "")
            return {
                "attribute": attribute,
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "source": str(row.get("source", "") or ""),
                "status": str(row.get("status", "") or ""),
                "reinforce_count": int(row.get("reinforce_count", 0) or 0),
                "weaken_count": int(row.get("weaken_count", 0) or 0),
                "weak_generic": bool(cls._weak_generic_skill_signal(attribute)),
            }

        def medium_or_high_pref(row):
            try:
                return float((row or {}).get("confidence", 0.0) or 0.0) >= 0.45
            except Exception:
                return False

        def compact_rule(row):
            row = dict(row or {})
            rule = str(row.get("rule", "") or "")
            try:
                confidence = float(row.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            if confidence < 0.45:
                return None
            if cls._text_contains_current_item_name(rule, current_item_names):
                return None
            if cls._preference_explanation_rule(rule):
                return None
            return {
                "rule": rule,
                "confidence": confidence,
                "status": str(row.get("status", "") or ""),
                "source": str(row.get("source", "") or ""),
                "reinforce_count": int(row.get("reinforce_count", 0) or 0),
                "weaken_count": int(row.get("weaken_count", 0) or 0),
                "weak_generic": bool(cls._weak_generic_skill_signal(rule)),
                "strategy_like": bool(cls._actionable_strategy_rule(rule)),
            }

        def is_risky_pref(row):
            row = dict(row or {})
            status = str(row.get("status", "") or "").lower()
            source = str(row.get("source", "") or "").lower()
            return status == "risky" or "weakened" in source

        def compact_pref_rows(rows, limit, include_risky=False):
            compacted = [
                compact_pref(row)
                for row in list(rows or [])
                if (
                    isinstance(row, dict)
                    and bool(is_risky_pref(row)) == bool(include_risky)
                    and medium_or_high_pref(row)
                    and not cls._text_contains_current_item_name(row.get("attribute", "") or row.get("rule", ""), current_item_names)
                    and not cls._rule_like_skill_text(row.get("attribute", "") or row.get("rule", ""))
                    and "rule_migration" not in str(row.get("source", "") or "")
                )
            ]
            compacted.sort(
                key=lambda row: (
                    bool(row.get("weak_generic", False)),
                    0 if "evolution" in str(row.get("source", "") or "") else 1,
                    -float(row.get("confidence", 0.0) or 0.0),
                    -int(row.get("reinforce_count", 0) or 0),
                )
            )
            selected = []
            weak_count = 0
            for row in compacted:
                if row.get("weak_generic", False):
                    weak_count += 1
                    if weak_count > 2:
                        continue
                selected.append(row)
                if len(selected) >= limit:
                    break
            return selected

        def compact_rule_rows(rows, limit):
            compacted = []
            for row in list(rows or []):
                if not isinstance(row, dict):
                    continue
                compacted_row = compact_rule(row)
                if compacted_row:
                    compacted.append(compacted_row)
            compacted.sort(
                key=lambda row: (
                    bool(row.get("weak_generic", False)),
                    not bool(row.get("strategy_like", False)),
                    0 if "evolution" in str(row.get("source", "") or "") else 1,
                    -int(row.get("reinforce_count", 0) or 0),
                    -float(row.get("confidence", 0.0) or 0.0),
                )
            )
            selected = []
            weak_count = 0
            for row in compacted:
                if row.get("weak_generic", False):
                    weak_count += 1
                    if weak_count > 1:
                        continue
                selected.append(row)
                if len(selected) >= limit:
                    break
            return selected

        item = dict(full_policy.get("item_selection_skill", {}) or {})
        comm = dict(full_policy.get("communication_selection_skill", {}) or {})
        route = dict(full_policy.get("communication_route_skill", {}) or {})
        state = dict(full_policy.get("policy_evolution_state", {}) or {})
        phase = str((decision_context or {}).get("phase", "") or "").strip().lower()
        try:
            policy_version = int(full_policy.get("version", 1) or 1)
        except Exception:
            policy_version = 1
        base = {
            "user_id": str(full_policy.get("user_id", "") or ""),
            "version": policy_version,
            "policy_evolution_state": {
                "num_updates": int(state.get("num_updates", 0) or 0),
                "last_updated_stage": str(state.get("last_updated_stage", "") or ""),
            },
        }
        item_payload = {
            "item_selection_skill": {
                "preferences": compact_pref_rows(item.get("preferences", []), 16, include_risky=False),
                "recent_signals": compact_pref_rows(item.get("recent_signals", []), 8, include_risky=False),
                "risky_preferences": compact_pref_rows(item.get("preferences", []), 6, include_risky=True),
                "decision_style": str(item.get("decision_style", "") or ""),
                "active_rules": compact_rule_rows(item.get("active_rules", []), 12),
                "risky_rules": compact_rule_rows(item.get("risky_rules", []), 6),
            }
        }
        comm_payload = {
            "communication_route_skill": {
                "version": int(route.get("version", 1) or 1) if str(route.get("version", 1) or 1).isdigit() else 1,
                "template_id": str(route.get("template_id", "") or ""),
                "template_features": dict(route.get("template_features", {}) or {}),
                "signature_order": list(route.get("signature_order", []) or [])[:12],
                "what_by_why": dict(route.get("what_by_why", {}) or {}),
                "how_by_what": dict(route.get("how_by_what", {}) or {}),
                "who_by_how": dict(route.get("who_by_how", {}) or {}),
                "child_order_memory": dict(route.get("child_order_memory", {}) or {}),
                "demotions": list(route.get("demotions", []) or [])[-16:],
                "unmapped_task_memory": list(route.get("unmapped_task_memory", []) or [])[-12:],
                "exploration_slots": list(route.get("exploration_slots", []) or [])[-12:],
                "exploration_history": list(route.get("exploration_history", []) or [])[-12:],
            }
        }
        if phase in ["communication", "comm", "communication_selection"]:
            base.update(comm_payload)
        else:
            base.update(item_payload)
        return base

    @classmethod
    def _sanitize_retrieved_skill_output(cls, text, decision_context):
        output = " ".join(str(text or "").strip().split())
        if not output:
            return output
        context = dict(decision_context or {})
        replacements = {}
        prior = str(context.get("prior_hint", "") or "").strip()
        proposal = str(context.get("proposal_item", "") or "").strip()
        if prior:
            replacements[prior] = "current prior item"
        if proposal:
            replacements[proposal] = "current proposal"
        for name in cls._collect_current_item_names(context):
            replacements.setdefault(name, "current candidate")
        for name, label in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            if not name:
                continue
            output = re.sub(re.escape(name), label, output, flags=re.IGNORECASE)
        if str(context.get("phase", "") or "").strip().lower() in ["communication", "comm", "communication_selection"]:
            output = re.sub(r"(?i)\bitem_context\s*:", "condition:", output, count=1)

        segments = [seg.strip() for seg in output.split("|")]
        cleaned_segments = []
        for seg in segments:
            if seg.lower().startswith("likes:"):
                prefix, _, rest = seg.partition(":")
                likes = []
                weak_count = 0
                for part in rest.split(","):
                    item = part.strip()
                    if not item:
                        continue
                    if "current candidate" in item.lower() or "current proposal" in item.lower() or "current prior item" in item.lower():
                        continue
                    is_weak = cls._weak_generic_skill_signal(item)
                    if is_weak and weak_count >= 1:
                        continue
                    weak_count += 1 if is_weak else 0
                    likes.append(item)
                if likes:
                    cleaned_segments.append(f"{prefix.strip()}: {', '.join(likes[:4])}")
                else:
                    cleaned_segments.append(f"{prefix.strip()}: transferable domain preference clusters")
            else:
                cleaned_segments.append(seg)
        return " | ".join(cleaned_segments)

    def _retrieve_relevant_skill_from_full_policy(self, full_policy, decision_context, args=None):
        import json as _json
        import logging
        logger = logging.getLogger(__name__)
        llm_args = args or self.args
        retrieval_context = dict(decision_context or {})
        retrieval_context.pop("history", None)
        retrieval_context.pop("history_str", None)
        retrieval_context.pop("user_history", None)
        payload = {
            "CurrentContext": retrieval_context,
            "FullUserPolicy": self._compact_policy_for_skill_retrieval(full_policy, retrieval_context),
        }
        try:
            user_prompt = _json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            logger.warning("[skill-retrieval] JSON serialization failed: %s", e)
            return self._distill_slim_policy(
                self.user_policy_store.build_slim_policy(full_policy, decision_context=decision_context),
                args=llm_args,
            )
        try:
            result = llm_request(_full_policy_retrieval_system_prompt(llm_args), user_prompt, llm_args)
            if result and isinstance(result, str) and len(result.strip()) > 10:
                return self._sanitize_retrieved_skill_output(result.strip(), retrieval_context)
            logger.warning("[skill-retrieval] LLM returned empty/short result: %s", repr(result)[:200])
        except Exception as e:
            logger.warning("[skill-retrieval] LLM call exception: %s", e)
        fallback_slim = self.user_policy_store.build_slim_policy(full_policy, decision_context=decision_context)
        return self._distill_slim_policy(fallback_slim, args=llm_args)

    @staticmethod
    def _communication_condition_summary(primary_trigger, uncertainty_points, self_confidence, shortlist_names, candidate_evidence):
        def evidence_signal(row):
            if not isinstance(row, dict):
                return ""
            text = " ".join(
                str(row.get(key, "") or "")
                for key in ["fit", "reason", "decision"]
            ).lower()
            signals = []
            keyword_groups = [
                ("portable/audio-device", ["portable", "mp3", "audio", "music player", "player", "pda", "handheld"]),
                ("camera/photo-equipment", ["camera", "photo", "dslr", "lens", "powershot", "coolpix"]),
                ("phone/mobile-device", ["phone", "cellular", "mobile"]),
                ("home-appliance", ["appliance", "washer", "dryer", "vacuum", "microwave", "mixer"]),
                ("vehicle/auto", ["auto", "car", "vehicle", "make-"]),
                ("game/toy", ["game", "toy", "learning", "fisher", "mattel"]),
                ("technical-specification", ["technical", "specification", "model", "features", "capacity"]),
                ("brand-loyalty", ["brand", "manufacturer", "loyalty"]),
                ("reliability/quality", ["reliable", "reliability", "quality", "durability"]),
            ]
            for label, keys in keyword_groups:
                if any(k in text for k in keys):
                    signals.append(label)
            return ", ".join(dict.fromkeys(signals))

        selected = []
        hesitation = []
        for row in list(candidate_evidence or []):
            signal = evidence_signal(row)
            if not signal:
                continue
            decision = str((row or {}).get("decision", "") or "").lower()
            if "selected" in decision:
                selected.append(signal)
            else:
                hesitation.append(signal)
        selected = list(dict.fromkeys(selected))[:2]
        hesitation = list(dict.fromkeys(hesitation))[:3]
        confidence_band = "high" if int(self_confidence or 0) >= 85 else "medium" if int(self_confidence or 0) >= 60 else "low"
        parts = [
            f"trigger={str(primary_trigger or 'unknown')}",
            f"uncertainty={','.join(str(x) for x in list(uncertainty_points or [])[:4]) or 'none'}",
            f"confidence={confidence_band}",
            f"focus_set_size={len(list(shortlist_names or []))}",
        ]
        if selected:
            parts.append(f"favorite_signal={'; '.join(selected)}")
        if hesitation:
            parts.append(f"hesitation_signals={'; '.join(hesitation)}")
        return "; ".join(parts)

    def bootstrap(self, stage="test"):
        self._ensure_initial_skill_snapshots()
        self._maybe_reset_skill_state()
        refresh_public_tree = bool(getattr(self.args, "com_refresh_public_tree_layout", False))
        if refresh_public_tree:
            self.public_tree_store.reset_to_initial_layout()
            self.public_tree = self.public_tree_store.load_tree(force_reload=True, refresh_layout=False)
        else:
            self.public_tree_store.ensure_layout(refresh_existing=False)
            self.public_tree = self.public_tree_store.load_tree(force_reload=True, refresh_layout=False)
        return {
            "stage": str(stage or "test"),
            "public_tree_root": str(self.public_tree_store.base_dir),
            "user_policy_root": str(self.user_policy_store.base_dir),
            "initial_skill_root": str(self.initial_skills_dir),
            "recommender_source": str(self.recommender_source),
        }

    def _ensure_initial_skill_snapshots(self):
        if not bool(getattr(self.args, "com_preserve_initial_skills", True)):
            return
        if self._initial_public_tree_done:
            return
        initial_public_tree = self.initial_skills_dir / "public_tree" / self.dataset_slug
        initial_store = PublicTreeStore(initial_public_tree, dataset=self.dataset_slug)
        initial_store.ensure_layout(refresh_existing=False)
        if not (initial_store.index_dir / "active_nodes.json").exists():
            initial_store.reset_runtime_indexes()
        self._initial_public_tree_done = True

    def _archive_path(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        return self.base_dir / "skill_state_archive" / f"{self.dataset}_{ts}"

    def _maybe_reset_skill_state(self):
        if self._skill_state_reset_done:
            return
        if not bool(getattr(self.args, "com_reset_skill_state", False)):
            return
        archive_root = self._archive_path()
        runtime_user_dir = Path(self.user_policy_store.base_dir)
        initial_user_dir = (
            Path(self.user_policy_store.initial_base_dir)
            if self.user_policy_store.initial_base_dir is not None
            else None
        )

        archive_root.mkdir(parents=True, exist_ok=True)
        if runtime_user_dir.exists():
            shutil.move(str(runtime_user_dir), str(archive_root / "user_policy"))
        if initial_user_dir is not None and initial_user_dir.exists():
            shutil.copytree(str(initial_user_dir), str(runtime_user_dir), dirs_exist_ok=True)
        else:
            runtime_user_dir.mkdir(parents=True, exist_ok=True)

        public_index_dir = Path(self.public_tree_store.index_dir)
        if public_index_dir.exists():
            shutil.copytree(str(public_index_dir), str(archive_root / "public_tree_indexes"), dirs_exist_ok=True)
        self.public_tree_store.ensure_layout()
        self.public_tree_store.reset_runtime_indexes()
        self._skill_state_reset_done = True

    def infer_primary_trigger(self, proposal_name, shortlist, uncertainty_points, prior_hint="", history_count=0):
        return infer_primary_trigger(
            args=self.args,
            proposal_name=proposal_name,
            shortlist=shortlist,
            uncertainty_points=uncertainty_points,
            prior_hint=prior_hint,
            history_count=history_count,
        )

    def _route_tree_marker(self):
        return {
            "tree_dataset": str(self.dataset_slug),
            "tree_root": str(self.public_tree_store.base_dir),
        }

    @staticmethod
    def _route_order_values(table):
        if isinstance(table, dict):
            for values in table.values():
                if isinstance(values, str):
                    value_iter = [values]
                else:
                    value_iter = list(values or [])
                for value in value_iter:
                    value = str(value or "").strip()
                    if value:
                        yield value

    def _invalid_route_nodes_for_current_tree(self, route):
        route = dict(route or {})
        tree = self.public_tree_store.load_tree()
        level_buckets = {
            "what": ["what_by_why"],
            "how": ["how_by_what"],
            "who": ["who_by_how"],
        }
        invalid = []
        for level, buckets in level_buckets.items():
            valid = set((tree.get(level, {}) or {}).keys())
            for bucket in buckets:
                for node_id in self._route_order_values(route.get(bucket, {})):
                    if node_id in ["none", "skip"]:
                        continue
                    if node_id not in valid:
                        invalid.append({"level": level, "bucket": bucket, "node": node_id})
        for row in list(route.get("demotions", []) or []) + list(route.get("exploration_slots", []) or []) + list(route.get("exploration_history", []) or []):
            if not isinstance(row, dict):
                continue
            level = str(row.get("level", "") or "")
            node_id = str(row.get("node", "") or "").strip()
            if level not in ["what", "how", "who"] or not node_id or node_id in ["none", "skip"]:
                continue
            if node_id not in set((tree.get(level, {}) or {}).keys()):
                invalid.append({"level": level, "bucket": "route_memory", "node": node_id})
        return invalid

    def _sanitize_policy_for_dataset_tree(self, user_raw, policy, policy_source, stage, history_summary=""):
        policy = dict(policy or {})
        route = dict(policy.get("communication_route_skill", {}) or {})
        marker = self._route_tree_marker()
        marker_mismatch = (
            str(route.get("tree_dataset", "") or "") != marker["tree_dataset"]
            or str(route.get("tree_root", "") or "") != marker["tree_root"]
        )
        invalid_nodes = self._invalid_route_nodes_for_current_tree(route) if route and not marker_mismatch else []
        if not marker_mismatch and not invalid_nodes:
            route.update(marker)
            policy["communication_route_skill"] = route
            return policy

        item_skill = dict(policy.get("item_selection_skill", {}) or {})
        communication_evidence = dict(policy.get("communication_initial_evidence", {}) or {})
        new_route = self.user_policy_store._initial_communication_route_skill(
            communication_evidence=communication_evidence,
            item_skill=item_skill,
            history_summary=history_summary,
        )
        new_route.update(marker)
        policy["communication_route_skill"] = new_route
        policy.pop("communication_absorption_skill", None)
        if str(stage or "").lower() == "train":
            self.user_policy_store.save_full_policy(policy, snapshot_reason="dataset_tree_communication_reset")
            self.user_policy_store.append_evolution_log(
                user_raw,
                {
                    "event": "dataset_tree_communication_reset",
                    "dataset": str(self.dataset_slug),
                    "policy_source": str(policy_source or ""),
                    "tree_root": marker["tree_root"],
                    "marker_mismatch": bool(marker_mismatch),
                    "invalid_nodes": invalid_nodes[:20],
                    "item_selection_skill_preserved": True,
                    "slim_cache_preserved": True,
                },
            )
        return policy

    def load_user_context(
        self,
        user_raw,
        history_str,
        target_profile,
        decision_context,
        stage="test",
    ):
        full_policy, source = self.user_policy_store.load_full_policy(
            user_raw=user_raw,
            history_summary=history_str,
            target_profile=target_profile,
            stage=stage,
            force_bootstrap=self._should_force_rebuild_user_policy(user_raw, stage),
        )
        full_policy = self._sanitize_policy_for_dataset_tree(
            user_raw=user_raw,
            policy=full_policy,
            policy_source=source,
            stage=stage,
            history_summary=history_str,
        )
        retrieval_context = dict(decision_context or {})
        retrieval_context.setdefault("task", "retrieve relevant user skill for the current COM phase")
        slim_policy_compact = self._retrieve_relevant_skill_from_full_policy(
            full_policy,
            retrieval_context,
            args=None,
        )
        if str(stage or "").lower() == "train":
            self.user_policy_store.cache_slim_policy(
                user_raw,
                {"source": "full_policy_retrieval", "retrieval_context": retrieval_context},
                phase="communication",
                compact_slim_policy=slim_policy_compact,
            )
        return {
            "full_user_policy": full_policy,
            "slim_user_policy": slim_policy_compact,
            "policy_source": source,
        }

    def build_decision_state(
        self,
        proposal_item,
        proposal_reason,
        shortlist,
        uncertainty_points,
        self_confidence,
        prior_hint="",
        history_count=0,
        slim_user_policy=None,
        communication_need=True,
    ):
        primary_trigger = self.infer_primary_trigger(
            proposal_name=proposal_item,
            shortlist=shortlist,
            uncertainty_points=uncertainty_points,
            prior_hint=prior_hint,
            history_count=history_count,
        )
        return build_decision_state(
            proposal_item=proposal_item,
            proposal_reason=proposal_reason,
            shortlist=shortlist,
            uncertainty_points=uncertainty_points,
            self_confidence=self_confidence,
            primary_trigger=primary_trigger,
            communication_need=communication_need,
            prior_item=prior_hint,
            slim_user_policy=slim_user_policy,
        )

    def select_path(self, decision_state, slim_user_policy, path_choice=None):
        return self.path_selector.select(
            decision_state=decision_state,
            slim_user_policy=slim_user_policy,
            path_choice=path_choice,
        )

    def _advisor_who_order(self, slim_user_policy):
        if isinstance(slim_user_policy, str):
            slim_user_policy = {}
        route = dict((slim_user_policy or {}).get("communication_route_skill", {}) or {})
        ordered = []
        for _scope, rows in dict(route.get("who_by_how", {}) or route.get("who_by_signature_what_how", {}) or {}).items():
            for who in list(rows or []):
                who = str(who or "").strip()
                if who and who not in ordered:
                    ordered.append(who)
        for row in list(route.get("exploration_slots", []) or []) + list(route.get("exploration_history", []) or []):
            who = str((row or {}).get("node", "") or "") if isinstance(row, dict) else ""
            who = str(who or "").strip()
            if who in ["trusted-advisors", "similar-users", "experienced-users", "topk-advisors"] and who not in ordered:
                ordered.append(who)
        if ordered:
            for key in ["trusted-advisors", "similar-users", "experienced-users", "topk-advisors"]:
                if key not in ordered:
                    ordered.append(key)
            return ordered
        pref = dict((slim_user_policy or {}).get("retrieved_preference", {}) or {})
        full = dict(pref.get("who_preference_full", {}) or {})
        if not full:
            comm_skill = dict((slim_user_policy or {}).get("communication_selection_skill", {}) or {})
            for row in list(comm_skill.get("top_who_preferences", []) or []):
                if isinstance(row, dict) and str(row.get("attribute", "") or "").strip():
                    full[str(row.get("attribute"))] = float(row.get("confidence", 0.0) or 0.0)
        defaults = [
            "trusted-advisors",
            "similar-users",
            "experienced-users",
            "topk-advisors",
        ]
        for key in defaults:
            full.setdefault(key, 0.0)
        return [key for key, _ in sorted(full.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))]

    def _reroute_if_advisor_pool_empty(
        self,
        host,
        path,
        slim_user_policy,
        u_raw,
        u_int,
        cands_int,
        proposal_iid,
        shortlist_names,
    ):
        path = dict(path or {})
        attempts = []
        original_profiles = self.advisor_pool.retrieve(
            host=host,
            path=path,
            u_raw=u_raw,
            u_int=u_int,
            cands_int=cands_int,
            proposal_iid=proposal_iid,
            shortlist_names=shortlist_names,
        )
        if original_profiles or str(path.get("who", "") or "") in ["none", "skip", ""]:
            return path, original_profiles, {
                "original_advisor_pool_empty": False,
                "advisor_pool_rerouted": False,
                "final_advisor_pool_empty": False,
                "final_advisor_count": int(len(original_profiles or [])),
                "advisor_pool_empty": False,
                "rerouted": False,
                "attempts": attempts,
                "who_branch_selection_status": dict(path.get("who_branch_selection_status", {}) or {}),
            }

        attempts.append(
            {
                "path": dict(path),
                "advisor_count": 0,
                "reason": "selected advisor source returned no available advisors",
                "trust_first_attempt_failed": str(path.get("who", "") or "") == "trusted-advisors",
            }
        )
        original_who = str(path.get("who", "") or "")
        who_order = [x for x in self._advisor_who_order(slim_user_policy) if x and x != original_who]
        for who in who_order:
            candidate_path = dict(path)
            candidate_path["who"] = who
            candidate_path["path_reason"] = list(candidate_path.get("path_reason", []) or []) + [
                "advisor_pool_empty_reroute",
                f"original who={original_who} had no available advisors",
                "trust-first path attempted before fallback" if original_who == "trusted-advisors" else "",
                "rerouted by available advisor source using communication route skill order",
            ]
            candidate_path["path_reason"] = [x for x in candidate_path["path_reason"] if str(x).strip()]
            candidate_path["risk_marks"] = list(candidate_path.get("risk_marks", []) or []) + ["advisor_pool_empty_rerouted"]
            profiles = self.advisor_pool.retrieve(
                host=host,
                path=candidate_path,
                u_raw=u_raw,
                u_int=u_int,
                cands_int=cands_int,
                proposal_iid=proposal_iid,
                shortlist_names=shortlist_names,
            )
            attempts.append(
                {
                    "path": dict(candidate_path),
                    "advisor_count": int(len(profiles or [])),
                }
            )
            if profiles:
                candidate_path["path_skill_payload"] = self.public_tree_store.build_path_skill_payload(candidate_path)
                return candidate_path, profiles, {
                    "original_advisor_pool_empty": True,
                    "advisor_pool_rerouted": True,
                    "final_advisor_pool_empty": False,
                    "final_advisor_count": int(len(profiles or [])),
                    "advisor_pool_empty": False,
                    "rerouted": True,
                    "attempts": attempts,
                    "original_path": dict(path),
                    "trust_first_attempt_failed": original_who == "trusted-advisors",
                    "fallback_who": str(candidate_path.get("who", "") or ""),
                    "who_branch_selection_status": dict(candidate_path.get("who_branch_selection_status", {}) or {}),
                }

        path["risk_marks"] = list(path.get("risk_marks", []) or []) + ["advisor_pool_empty"]
        return path, [], {
            "original_advisor_pool_empty": True,
            "advisor_pool_rerouted": False,
            "final_advisor_pool_empty": True,
            "final_advisor_count": 0,
            "advisor_pool_empty": True,
            "rerouted": False,
            "attempts": attempts,
            "original_path": dict(path),
            "who_branch_selection_status": dict(path.get("who_branch_selection_status", {}) or {}),
        }

    @staticmethod
    def _clamp_confidence(value, default=60):
        try:
            return max(0, min(100, int(value)))
        except Exception:
            return int(default)

    @staticmethod
    def _normalize_text(text):
        return " ".join(str(text or "").strip().lower().split())

    @staticmethod
    def _safe_list(values):
        return [str(x) for x in (values or []) if str(x or "").strip()]

    def _contains_target_name(self, values, target_names):
        targets = {self._normalize_text(x) for x in (target_names or []) if self._normalize_text(x)}
        if not targets:
            return False
        for value in values or []:
            norm = self._normalize_text(value)
            if norm and norm in targets:
                return True
        return False

    def _target_alignment(self, candidate_names, target_names, focus_names=None):
        candidate_names = [str(x) for x in (candidate_names or []) if str(x or "").strip()]
        target_names = [str(x) for x in (target_names or []) if str(x or "").strip()]
        focus_names = [str(x) for x in (focus_names or []) if str(x or "").strip()]
        target_norms = {self._normalize_text(x) for x in target_names if self._normalize_text(x)}
        candidate_norms = {self._normalize_text(x) for x in candidate_names if self._normalize_text(x)}
        focus_norms = {self._normalize_text(x) for x in focus_names if self._normalize_text(x)}
        missing = []
        for name in target_names:
            norm = self._normalize_text(name)
            if norm and norm not in candidate_norms:
                missing.append(name)
        return {
            "target_item_names": list(target_names),
            "candidate_target_overlap": bool(target_norms & candidate_norms),
            "focus_target_overlap": bool(target_norms & focus_norms) if focus_names else False,
            "missing_target_item_names": missing,
        }

    def _committee_state(self, execution_packet):
        committee = dict((execution_packet or {}).get("committee_packet", {}) or {})
        return {
            "advisor_pool_empty": bool(committee.get("final_advisor_pool_empty", committee.get("advisor_pool_empty", False))),
            "original_advisor_pool_empty": bool(committee.get("original_advisor_pool_empty", False)),
            "advisor_pool_rerouted": bool(committee.get("advisor_pool_rerouted", False)),
            "final_advisor_pool_empty": bool(committee.get("final_advisor_pool_empty", committee.get("advisor_pool_empty", False))),
            "final_advisor_count": int(committee.get("final_advisor_count", len((execution_packet or {}).get("advisor_feedbacks", []) or [])) or 0),
            "committee_aggregation_mode": str(committee.get("aggregation_mode", "summary_agent_v1") or "summary_agent_v1"),
            "committee_decision_policy": str(committee.get("decision_policy", "information_only_no_vote") or "information_only_no_vote"),
        }

    def _augment_evaluation(
        self,
        evaluation_result,
        candidate_names,
        target_names,
        focus_names,
        proposal_name,
        final_item_name,
        prior_hint,
        execution_packet,
        target_injected_into_focus=False,
        proposal_iid=None,
        final_iid=None,
        target_item_ids=None,
    ):
        out = dict(evaluation_result or {})
        alignment = self._target_alignment(candidate_names, target_names, focus_names=focus_names)
        out.update(alignment)
        target_ids = set()
        for value in target_item_ids or []:
            try:
                target_ids.add(int(value))
            except (TypeError, ValueError):
                continue

        def is_target_id(item_id):
            try:
                return int(item_id) in target_ids
            except (TypeError, ValueError):
                return False

        id_outcome_available = bool(target_ids) and proposal_iid is not None and final_iid is not None
        out.update(
            {
                "initial_hit": is_target_id(proposal_iid) if id_outcome_available else self._contains_target_name([proposal_name], target_names),
                "final_hit": is_target_id(final_iid) if id_outcome_available else self._contains_target_name([final_item_name], target_names),
                "outcome_id_based": bool(id_outcome_available),
                "prior_hit": self._contains_target_name([prior_hint], target_names),
                "proposal_equals_prior": bool(self._normalize_text(proposal_name) == self._normalize_text(prior_hint)),
                "final_equals_prior": bool(self._normalize_text(final_item_name) == self._normalize_text(prior_hint)),
                "target_injected_into_focus": bool(target_injected_into_focus),
            }
        )
        out.update(self._committee_state(execution_packet))
        return out

    def _sorted_candidate_ids(self, cands_int, score_dict):
        indexed = []
        for idx, iid in enumerate(cands_int or []):
            score = float((score_dict or {}).get(int(iid), 0.0))
            indexed.append((score, idx, int(iid)))
        indexed.sort(key=lambda row: (-float(row[0]), int(row[1])))
        return [iid for _, _, iid in indexed]

    def _resolve_proposal(self, host, proposal_name, cands_int, sorted_candidate_ids, prior_hint="", candidate_shortlist=None):
        proposal_iid = host._match_name_to_iid(proposal_name, cands_int)
        if proposal_iid is not None:
            return int(proposal_iid), str(host._get_item_name(int(proposal_iid))), {
                "proposal_repaired": False,
                "original_proposal": str(proposal_name or ""),
                "repair_source": "exact_or_fuzzy_candidate_match",
            }
        candidate_names = [str(host._get_item_name(int(iid))) for iid in (cands_int or [])]
        for source, names in [
            ("candidate_shortlist", list(candidate_shortlist or [])),
            ("prior_hint", [prior_hint] if prior_hint else []),
        ]:
            for name in names:
                iid = host._match_name_to_iid(name, cands_int)
                if iid is not None:
                    repaired_name = str(host._get_item_name(int(iid)))
                    return int(iid), repaired_name, {
                        "proposal_repaired": True,
                        "original_proposal": str(proposal_name or ""),
                        "repaired_proposal": repaired_name,
                        "repair_source": source,
                        "candidate_count": int(len(candidate_names)),
                    }
        fallback_iid = None
        if sorted_candidate_ids:
            fallback_iid = int(sorted_candidate_ids[0])
        elif cands_int:
            fallback_iid = int(cands_int[0])
        if fallback_iid is not None:
            repaired_name = str(host._get_item_name(int(fallback_iid)))
            return int(fallback_iid), repaired_name, {
                "proposal_repaired": True,
                "original_proposal": str(proposal_name or ""),
                "repaired_proposal": repaired_name,
                "repair_source": "top_scored_candidate_fallback",
                "candidate_count": int(len(candidate_names)),
            }
        raise ValueError(
            "User Reasoning Skill selected an item outside the candidate set and no candidate fallback exists: "
            f"proposal={proposal_name!r}; candidates={candidate_names}"
        )

    def _resolve_candidate_name_list(self, host, candidate_names, cands_int, limit):
        ordered = []
        seen = set()
        for name in candidate_names or []:
            iid = host._match_name_to_iid(name, cands_int)
            if iid is None:
                continue
            exact = str(host._get_item_name(int(iid)))
            key = self._normalize_text(exact)
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(exact)
            if len(ordered) >= int(limit):
                break
        return ordered

    def _build_shortlist(self, host, sorted_candidate_ids, proposal_name, prior_hint, absorbed_memory, limit, candidate_shortlist=None):
        ordered = []
        seen = set()

        def push(name):
            key = str(name or "").strip()
            if not key or key in seen:
                return
            seen.add(key)
            ordered.append(key)

        push(proposal_name)
        for name in candidate_shortlist or []:
            push(name)
        if candidate_shortlist and len(ordered) >= min(2, max(1, int(limit))):
            return ordered[: max(1, int(limit))]
        for alt in list((absorbed_memory or {}).get("alternative_items", []) or []):
            push(alt)
        push(prior_hint)
        for iid in sorted_candidate_ids or []:
            push(host._get_item_name(int(iid)))
            if len(ordered) >= int(limit):
                break
        return ordered[: max(1, int(limit))]

    def _inject_targets_into_shortlist(self, shortlist_names, target_names, limit, protected_names=None):
        ordered = [str(x) for x in (shortlist_names or []) if str(x or "").strip()]
        limit = max(1, int(limit or len(ordered) or 1))
        protected = {self._normalize_text(x) for x in (protected_names or []) if self._normalize_text(x)}
        injected = []
        seen = {self._normalize_text(x) for x in ordered if self._normalize_text(x)}
        for target in [str(x) for x in (target_names or []) if str(x or "").strip()]:
            key = self._normalize_text(target)
            if not key or key in seen:
                continue
            if len(ordered) < limit:
                ordered.append(target)
            else:
                replace_idx = None
                for idx in range(len(ordered) - 1, -1, -1):
                    if self._normalize_text(ordered[idx]) not in protected:
                        replace_idx = idx
                        break
                if replace_idx is None:
                    replace_idx = len(ordered) - 1
                old_key = self._normalize_text(ordered[replace_idx])
                if old_key in seen:
                    seen.discard(old_key)
                ordered[replace_idx] = target
            seen.add(key)
            injected.append(target)
        return ordered[:limit], injected

    def _contains_any_clue(self, text, clues):
        norm = self._normalize_text(text)
        for clue in clues or []:
            key = self._normalize_text(clue)
            if key and key in norm:
                return True
        return False

    def _infer_uncertainty_points(
        self,
        host,
        history_str,
        target_profile,
        proposal_name,
        proposal_reason,
        shortlist_names,
        proposal_iid,
        sorted_candidate_ids,
        score_dict,
        prior_hint="",
        absorbed_memory=None,
    ):
        points = []
        for point in list((absorbed_memory or {}).get("remaining_uncertainty", []) or []):
            key = str(point or "").strip()
            if key:
                points.append(key)

        if prior_hint and self._normalize_text(prior_hint) != self._normalize_text(proposal_name):
            points.append("internal_prior_conflict")

        if len(shortlist_names or []) >= 2:
            proposal_score = float((score_dict or {}).get(int(proposal_iid), 0.0)) if proposal_iid is not None else 0.0
            competitor_score = None
            for iid in sorted_candidate_ids or []:
                if proposal_iid is not None and int(iid) == int(proposal_iid):
                    continue
                competitor_score = float((score_dict or {}).get(int(iid), 0.0))
                break
            compare_margin = float(getattr(self.args, "com_skill_compare_margin", 0.05) or 0.05)
            if competitor_score is None or abs(proposal_score - competitor_score) <= compare_margin or len(shortlist_names) >= 3:
                points.append("candidate_comparison")

        history_clues = host._extract_history_clues(history_str, max_n=4) if hasattr(host, "_extract_history_clues") else []
        profile_clues = host._extract_history_clues(target_profile, max_n=4) if hasattr(host, "_extract_history_clues") else []
        if not self._contains_any_clue(proposal_reason, history_clues + profile_clues):
            points.append("preference_alignment")

        if prior_hint and self._normalize_text(proposal_name) != self._normalize_text(prior_hint):
            if not self._contains_any_clue(proposal_reason, history_clues[:2] + profile_clues[:2]):
                points.append("novelty_justification")

        return merge_unique(points)

    def _slim_policy_preview(self, slim_user_policy):
        if isinstance(slim_user_policy, str):
            return {"compact_profile": slim_user_policy}
        item_skill = dict((slim_user_policy or {}).get("item_selection_skill", {}) or {})
        comm_skill = dict((slim_user_policy or {}).get("communication_selection_skill", {}) or {})
        route_skill = dict((slim_user_policy or {}).get("communication_route_skill", {}) or {})
        post_skill = dict((slim_user_policy or {}).get("post_feedback_skill", {}) or {})
        rules = dict((slim_user_policy or {}).get("retrieved_reasoning_rules", {}) or {})
        return {
            "communication_route_template": str(route_skill.get("template_id", "") or ""),
            "signature_order": list(route_skill.get("signature_order", []) or [])[:4],
            "route_demotions": list(route_skill.get("demotions", []) or [])[-4:],
            "route_unmapped_task_memory": list(route_skill.get("unmapped_task_memory", []) or [])[-4:],
            "route_exploration_slots": list(route_skill.get("exploration_slots", []) or [])[-4:],
            "trigger_strength": float(comm_skill.get("trigger_strength", 0.5) or 0.5),
            "top_who_preferences": list(comm_skill.get("top_who_preferences", []) or [])[:2],
            "top_how_preferences": list(comm_skill.get("top_how_preferences", []) or [])[:2],
            "path_memory": list(comm_skill.get("path_memory", []) or [])[:2],
            "communication_round_memory": list(comm_skill.get("communication_round_memory", []) or [])[:2],
            "post_feedback_trust_rules": list(post_skill.get("post_feedback_trust_rules", []) or [])[:2],
            "item_preferences": list(item_skill.get("preferences", []) or [])[:3],
            "item_selection_rules": list(rules.get("item_selection", rules.get("core_decision", [])) or [])[:2],
            "communication_rules": list(rules.get("communication_selection", rules.get("communication", [])) or [])[:2],
            "post_feedback_rules": list(rules.get("post_feedback", []) or [])[:2],
            "risky_rules": list(rules.get("risky", []) or [])[:2],
        }

    def _append_memory(self, memory_window, role, text):
        body = " ".join(str(text or "").split())
        if not body:
            return
        memory_window.append({"role": str(role or "system"), "text": body[:1200]})

    def _render_shared_memory(self, host, memory_window):
        if hasattr(host, "_render_shared_memory"):
            return host._render_shared_memory(memory_window)
        if not memory_window:
            return "none"
        return "\n".join([f"{idx}. [{row.get('role', 'system')}] {row.get('text', '')}" for idx, row in enumerate(memory_window, start=1)])

    def _build_skip_path(self):
        return build_communication_path(
            why="skip",
            who="none",
            how="none",
            what="none",
            path_reason=["communication skipped because the decision is already stable"],
            path_score=0.0,
            risk_marks=[],
            trial_flag=False,
            pattern_source="direct",
        )

    def _safe_target_profile(self, host, user_agent, u_raw, history_str, candidate_names, prior_hint):
        cache_key = f"profile::{self.dataset}::{u_raw}"
        if hasattr(host, "_get_profile_cached"):
            profile = host._get_profile_cached(
                cache_key=cache_key,
                u_raw=u_raw,
                history_str=history_str,
                candidate_names=candidate_names,
                prior_hint=prior_hint,
                profile_agent=user_agent,
            )
            profile = str(profile or "").strip()
            if profile:
                return profile
        if hasattr(host, "_fallback_target_profile"):
            return str(host._fallback_target_profile(history_str))
        return "Prefers items aligned with recent interactions and stable historical interests."

    def _load_cached_stage1_proposal_slim(self, user_raw):
        if not bool(getattr(self.args, "com_reuse_cached_stage1_slim", False)):
            return ""
        try:
            paths = self.user_policy_store._paths(user_raw)
            payload = load_json(paths["slim_cache_proposal_json"], default={}) or {}
        except Exception:
            return ""
        if isinstance(payload, dict):
            slim = str(payload.get("compact_slim_policy", "") or "").strip()
            if slim:
                return slim
        if isinstance(payload, str):
            return payload.strip()
        return ""

    def _discussion_result_from_execution(self, decision_state, path, execution_packet):
        committee = dict((execution_packet or {}).get("committee_packet", {}) or {})
        evidence_summary = dict(committee.get("evidence_summary", {}) or {})
        synthesis_packet = dict(committee.get("advisor_synthesis_packet", {}) or {})
        feedbacks = list((execution_packet or {}).get("advisor_feedbacks", []) or [])
        discussion_memory = list((execution_packet or {}).get("discussion_memory", []) or [])
        support_reasons = list(committee.get("proposal_support_reasons", []) or [])
        oppose_reasons = list(committee.get("proposal_oppose_reasons", []) or [])
        additional_info = list(committee.get("proposal_additional_info", []) or [])
        alternatives = list(committee.get("alternative_candidates", []) or [])
        protocol_issues = list(committee.get("protocol_issues", []) or [])
        interaction_summary = dict(synthesis_packet.get("interaction_summary", {}) or {})
        candidate_summaries = dict(synthesis_packet.get("candidate_summaries", {}) or {})
        task_specific_summary = dict(synthesis_packet.get("task_specific_summary", {}) or {})
        extra_task_summary = dict(synthesis_packet.get("extra_task_summary", {}) or {})
        extra_interaction_summary = dict(synthesis_packet.get("extra_interaction_summary", {}) or {})
        silent_focus_candidates = list(evidence_summary.get("silent_focus_candidates", []) or [])
        support_only_candidates = list(evidence_summary.get("support_only_candidates", []) or [])
        unresolved_questions = list(synthesis_packet.get("remaining_uncertainty", []) or evidence_summary.get("unresolved_questions", []) or [])
        key_conflicts = list(evidence_summary.get("key_conflicts", []) or [])

        direct_points = []
        for fb in feedbacks[:6]:
            defended = str(fb.get("defended_item", "") or fb.get("endorsed_item", "") or "")
            attacked = str(fb.get("attacked_item", "") or "")
            support = str(fb.get("support_reason", "") or "")
            oppose = str(fb.get("oppose_reason", "") or "")
            task_answer = str(fb.get("task_answer", "") or "")
            response_prev = str(fb.get("response_to_previous", "") or "")
            challenge_prev = str(fb.get("challenge_or_support_previous", "") or "")
            key_tradeoff = str(fb.get("key_tradeoff", "") or "")
            comparison_reason = str(fb.get("comparison_reason", "") or "")
            correction = str(fb.get("correction", "") or "")
            still_missing = str(fb.get("still_missing", "") or "")
            if any([defended, support, attacked, oppose, task_answer, response_prev, challenge_prev, key_tradeoff, comparison_reason, correction, still_missing]):
                direct_points.append(
                    {
                        "advisor": str(fb.get("advisor_id", "") or "advisor"),
                        "defended_item": defended,
                        "challenged_item": attacked,
                        "support_reason": support,
                        "oppose_reason": oppose,
                        "task_answer": task_answer,
                        "response_to_previous": response_prev,
                        "challenge_or_support_previous": challenge_prev,
                        "key_tradeoff": key_tradeoff,
                        "comparison_reason": comparison_reason,
                        "correction": correction,
                        "still_missing": still_missing,
                        "raw_text": str(fb.get("raw_text", "") or ""),
                    }
                )

        remaining_uncertainty = []
        if bool(committee.get("advisor_pool_empty", False)):
            remaining_uncertainty.append("advisor_pool_empty")
        if silent_focus_candidates:
            remaining_uncertainty.append("missing_advisor_evidence")
        if support_only_candidates:
            remaining_uncertainty.append("unchallenged_support")
        if protocol_issues:
            remaining_uncertainty.append("communication_protocol")
        if unresolved_questions:
            remaining_uncertainty.append("candidate_comparison")
        remaining_uncertainty = merge_unique(remaining_uncertainty)

        evidence_packet = dict(synthesis_packet or {})
        evidence_packet.update({
            "source": "direct_discussion_result",
            "decision_policy": "information_only_no_vote",
            "advisor_synthesis_packet": synthesis_packet,
            "support": support_reasons,
            "oppose": oppose_reasons,
            "additional": additional_info,
            "alternatives": alternatives,
            "evidence_summary": evidence_summary,
            "protocol_issues": protocol_issues,
            "silent_focus_candidates": list(silent_focus_candidates),
            "missing_advisor_evidence": list(evidence_summary.get("missing_advisor_evidence", []) or []),
            "support_only_candidates": list(support_only_candidates),
            "unchallenged_support_warning": str(evidence_summary.get("unchallenged_support_warning", "") or ""),
            "advisor_pool_empty": bool(committee.get("advisor_pool_empty", False)),
            "committee_packet": committee,
            "advisor_feedbacks": feedbacks,
            "current_round_discussion_memory": discussion_memory,
            "direct_advisor_points": direct_points,
        })
        return {
            "source": "direct_discussion_result",
            "communication_path": dict(path or {}),
            "advisor_feedbacks": feedbacks,
            "current_round_discussion_memory": discussion_memory,
            "committee_packet": committee,
            "evidence_summary": evidence_summary,
            "direct_advisor_points": direct_points,
            "candidate_comparison": {
                "decision_policy": "information_only_no_vote",
                "source": "advisor_summary_agent_v1",
                "candidate_summaries": candidate_summaries,
                "task_specific_summary": task_specific_summary,
                "extra_task_summary": extra_task_summary,
                "interaction_summary": interaction_summary,
                "extra_interaction_summary": extra_interaction_summary,
                "silent_focus_candidates": list(silent_focus_candidates),
                "support_only_candidates": list(support_only_candidates),
                "key_conflicts": key_conflicts,
                "unresolved_questions": list(unresolved_questions),
            },
            "alternative_items": alternatives,
            "remaining_uncertainty": remaining_uncertainty,
            "evidence_packet": evidence_packet,
            "accepted_points": direct_points,
            "silent_or_missing_evidence": list(evidence_summary.get("missing_advisor_evidence", []) or []),
            "feedback_to_advisors_seed": list(unresolved_questions)[:4],
        }

    def _build_round_trace(
        self,
        ridx,
        decision_state,
        communication_action,
        slim_user_policy,
        path,
        advisor_profiles,
        execution_packet,
        discussion_result,
        redecision_packet,
        evaluation_result,
        evolution_update=None,
    ):
        committee = dict((execution_packet or {}).get("committee_packet", {}) or {})
        executed_path = dict((execution_packet or {}).get("path", {}) or path or {})
        return {
            "round": int(ridx),
            "stage1_decision_and_trigger": {
                "proposal_item": str((decision_state or {}).get("proposal_item", "") or ""),
                "proposal_reason": str((decision_state or {}).get("proposal_reason", "") or ""),
                "shortlist": list((decision_state or {}).get("shortlist", []) or []),
                "uncertainty_points": list((decision_state or {}).get("uncertainty_points", []) or []),
                "self_confidence": int((decision_state or {}).get("self_confidence", 0) or 0),
                "candidate_evidence": list((decision_state or {}).get("candidate_evidence", []) or []),
                "shortlist_semantics": str((decision_state or {}).get("shortlist_semantics", "") or ""),
                "primary_trigger": str((decision_state or {}).get("primary_trigger", "") or ""),
                "communication_need": bool((decision_state or {}).get("communication_need", True)),
                "communication_action": str(communication_action or ""),
                "communication_training_gate": dict((decision_state or {}).get("communication_training_gate", {}) or {}),
                "why_matching": dict((decision_state or {}).get("why_matching", {}) or {}),
                "target_injected_into_shortlist": bool((decision_state or {}).get("target_injected_into_shortlist", False)),
                "injected_target_items": list((decision_state or {}).get("injected_target_items", []) or []),
                "slim_user_policy": self._slim_policy_preview(slim_user_policy),
            },
            "stage2_path_selection": {
                "selected_path": executed_path,
                "user_selected_path": dict(path or {}),
                "advisor_pool_status": dict((execution_packet or {}).get("advisor_pool_status", {}) or {}),
                "advisor_pool": [
                    {
                        "advisor_id": str(row.get("u_raw", "") or ""),
                        "advisor_type": str(row.get("advisor_type", "") or ""),
                        "reliability": float(row.get("reliability", 0.0) or 0.0),
                        "sim": float(row.get("sim", 0.0) or 0.0),
                        "experience_items": [str(x) for x in (row.get("experience_items", []) or [])],
                        "experience_item_names": [str(x) for x in (row.get("experience_item_names", []) or [])],
                        "experience_score": float(row.get("experience_score", 0.0) or 0.0),
                        "trust_relation": str(row.get("trust_relation", "") or "none"),
                        "trust_scope": str(row.get("trust_scope", "") or "none"),
                        "history_similarity_bucket": str(row.get("history_similarity_bucket", "") or "none"),
                        "trust_subbranch": str(row.get("trust_subbranch", "") or "none"),
                        "mutual_count": int(row.get("mutual_count", 0) or 0),
                        "hop": int(row.get("hop", 0) or 0),
                        "selection_reason": str(row.get("selection_reason", "") or ""),
                    }
                    for row in (advisor_profiles or [])
                ],
            },
            "stage3_communication_execution": {
                "round_type": str((execution_packet or {}).get("round_type", "") or ""),
                "previous_user_feedback": (execution_packet or {}).get("previous_user_feedback", {}) or {},
                "focus_candidates": list((execution_packet or {}).get("focus_candidates", []) or []),
                "requester_shareable_item_brief": dict((execution_packet or {}).get("requester_shareable_item_brief", {}) or {}),
                "committee_result": committee,
                "advisor_feedbacks": list((execution_packet or {}).get("advisor_feedbacks", []) or []),
            },
            "stage4_post_feedback_redecision": {
                "discussion_result": dict(discussion_result or {}),
                "prompt_communication_summary": str((redecision_packet or {}).get("prompt_communication_summary", "") or ""),
                "prompt_decision_context": str((redecision_packet or {}).get("prompt_decision_context", "") or ""),
                "revised_item": str((redecision_packet or {}).get("revised_name", "") or ""),
                "revised_reason": str((redecision_packet or {}).get("revised_reason", "") or ""),
                "arbitration": dict((redecision_packet or {}).get("arbitration", {}) or {}),
            },
            "stage6_evaluation": dict(evaluation_result or {}),
            "stage7_train_evolution": dict(evolution_update or {}),
        }

    def _build_evolution_trace_context(
        self,
        stage,
        user_raw,
        ridx,
        candidate_names,
        gt_item_names,
        decision_state,
        path,
        execution_packet,
        absorbed_memory,
        redecision_packet,
        evaluation_result,
        proposal_name,
        final_item_name,
    ):
        focus_names = list((execution_packet or {}).get("focus_candidates", []) or [])
        alignment = self._target_alignment(candidate_names, gt_item_names, focus_names=focus_names)
        decision_state = dict(decision_state or {})
        evaluation_result = dict(evaluation_result or {})
        communication_gate = dict(decision_state.get("communication_training_gate", {}) or {})
        path = dict(path or {})
        execution_packet = dict(execution_packet or {})
        redecision_packet = dict(redecision_packet or {})
        communication_continuation = bool(
            decision_state.get("communication_continuation", False)
            or evaluation_result.get("communication_continuation", False)
            or communication_gate.get("communication_continuation", False)
            or (
                int(ridx) > 1
                and str((redecision_packet.get("arbitration", {}) or {}).get("decision_state", "") or "").strip().lower() == "continue"
                and bool(str(decision_state.get("previous_user_feedback", "") or "").strip())
            )
        )
        communication_target_gate_exempt = bool(
            decision_state.get("communication_target_gate_exempt", False)
            or evaluation_result.get("communication_target_gate_exempt", False)
            or communication_gate.get("communication_target_gate_exempt", False)
            or communication_continuation
        )
        communication_session_started = bool(
            str(path.get('why', "") or "").strip().lower() not in {"", "none", "skip", "no", "null", "n/a"}
            and not bool(execution_packet.get("communication_skipped_by_training_gate", False))
        )
        return {
            "stage": str(stage or "train"),
            "user_id": str(user_raw),
            "round_id": int(ridx),
            "candidate_item_names": [str(x) for x in candidate_names or []],
            "target_item_names": [str(x) for x in gt_item_names or []],
            "candidate_target_overlap": bool(alignment.get("candidate_target_overlap", False)),
            "focus_target_overlap": bool(alignment.get("focus_target_overlap", False)),
            "missing_target_item_names": list(alignment.get("missing_target_item_names", []) or []),
            "target_injected_into_focus": bool((evaluation_result or {}).get("target_injected_into_focus", False)),
            "communication_session_started": bool(communication_session_started),
            "communication_continuation": bool(communication_continuation),
            "communication_target_gate_exempt": bool(communication_target_gate_exempt),
            "decision_state": dict(decision_state or {}),
            "path": dict(path or {}),
            "execution_packet": dict(execution_packet or {}),
            "discussion_result": dict(absorbed_memory or {}),
            "absorbed_memory": dict(absorbed_memory or {}),
            "redecision_packet": dict(redecision_packet or {}),
            "evaluation_result": dict(evaluation_result or {}),
            "proposal_item_name": str(proposal_name or ""),
            "final_item_name": str(final_item_name or ""),
        }

    def run_interaction(
        self,
        host,
        user_agent,
        advisor_agent,
        sample,
        u_raw,
        u_int,
        cands_int,
        score_dict,
        prior_hint,
        gt_items=None,
        collect_trace=False,
        stage="test",
    ):
        history_str = str((sample or {}).get("seq_str", "") or "")
        sorted_candidate_ids = self._sorted_candidate_ids(cands_int, score_dict)
        candidate_names = [str(host._get_item_name(int(iid))) for iid in cands_int]
        # Do not build or inject a separate long-term/profile summary into LLM prompts.
        # Stage-1 decisions should rely on recent History, Candidates, PriorHint, and UserReasoningSkillSlim.
        target_profile = ""
        force_rebuild_policy = self._should_force_rebuild_user_policy(u_raw, stage)
        policy_exists = bool(self.user_policy_store.policy_exists(u_raw))
        communication_evidence = None
        core_payload = None
        if str(stage or "").lower() == "train" and (force_rebuild_policy or not policy_exists):
            if hasattr(host, "_build_communication_initial_evidence"):
                communication_evidence = host._build_communication_initial_evidence(
                    u_raw=str(u_raw),
                    u_int=u_int,
                    sample=sample,
                    history_str=history_str,
                )
            if (
                hasattr(host, "_build_stat_initial_core_skill")
                and hasattr(host, "_generate_llm_initial_core_skill")
                and hasattr(host, "_merge_llm_label_payload")
            ):
                stat_payload = host._build_stat_initial_core_skill(
                    u_raw=str(u_raw),
                    sample=sample,
                    history_str=history_str,
                    target_profile=target_profile,
                )
                llm_payload = host._generate_llm_initial_core_skill(
                    u_raw=str(u_raw),
                    history_str=history_str,
                    target_profile=target_profile,
                    stat_payload=stat_payload,
                )
                core_payload = host._merge_llm_label_payload(stat_payload, llm_payload)
        full_user_policy, policy_source = self.user_policy_store.load_full_policy(
            user_raw=u_raw,
            history_summary=history_str,
            target_profile=target_profile,
            stage=stage,
            communication_evidence=communication_evidence,
            core_rules=(core_payload or {}).get("core_rules"),
            core_preference=(core_payload or {}).get("core_preference"),
            core_initial_evidence=(core_payload or {}).get("core_initial_evidence"),
            force_bootstrap=force_rebuild_policy,
        )
        full_user_policy = self._sanitize_policy_for_dataset_tree(
            user_raw=u_raw,
            policy=full_user_policy,
            policy_source=policy_source,
            stage=stage,
            history_summary=history_str,
        )

        max_round_arg = getattr(self.args, "com_user_advisor_rounds", None)
        if max_round_arg is None:
            max_round_arg = getattr(self.args, "com_max_communication_rounds", None)
        if max_round_arg is None:
            max_round_arg = getattr(self.args, "com_max_rounds", 2)
        max_rounds = max(1, int(max_round_arg or 2))
        shortlist_limit = max(2, int(getattr(self.args, "com_skill_shortlist_limit", 5) or 5))
        history_count = len(list((sample or {}).get("seq", []) or []))
        memory_window = []
        structured_rounds = []
        raw_trace = []
        train_round_trace_contexts = []
        if collect_trace:
            set_llm_prompt_trace(
                raw_trace,
                {
                    "stage": str(stage or "test"),
                    "user_id": str(u_raw),
                    "user_int": int(u_int),
                },
            )
        else:
            clear_llm_prompt_trace()
        final_iid = None
        gt_item_names = [str(host._get_item_name(int(iid))) for iid in (gt_items or [])]
        absorbed_memory = None
        stage1_round_cache = None
        proposal_slim_cache = None
        communication_slim_cache = None
        advisor_profiles_cache = None
        discussion_memory_cache = []
        first_communication_how = ""
        first_communication_path_cache = None
        active_hesitation_set = None
        initial_target_alignment = self._target_alignment(candidate_names, gt_item_names)
        force_train_target_focus = (
            str(stage or "").lower() == "train"
            and bool(getattr(self.args, "com_train_force_target_in_focus", True))
            and bool(gt_item_names)
        )

        def add_raw_event(event, **payload):
            row = {"event": str(event or "")}
            row.update(payload)
            raw_trace.append(row)

        def compact_previous_user_feedback(arbitration, memory):
            arbitration = dict(arbitration or {})
            memory = dict(memory or {})
            requests = [str(x) for x in list(arbitration.get("feedback_to_advisors", []) or []) if str(x).strip()]
            if requests:
                text = " ".join(requests).strip()
                return text[:900] + "..." if len(text) > 903 else text
            remaining = [str(x) for x in list(memory.get("remaining_uncertainty", []) or []) if str(x).strip()]
            evidence_packet = dict(memory.get("evidence_packet", {}) or {})
            missing = [str(x) for x in list(evidence_packet.get("silent_focus_candidates", []) or []) if str(x).strip()]
            evidence_summary = dict(evidence_packet.get("evidence_summary", {}) or {})
            by_candidate = dict(evidence_summary.get("by_candidate", {}) or {})
            parts = []
            discussion_result = str(evidence_summary.get("discussion_result", "") or "").strip()
            if discussion_result:
                parts.append("Please continue from the previous unresolved advisor discussion.")
            if remaining:
                parts.append("Focus on remaining uncertainty: " + ", ".join(remaining[:3]))
            if missing:
                parts.append("Please cover missing or silent evidence for: " + ", ".join(missing[:3]))
            text = " ".join(parts).strip()
            return text[:900] + "..." if len(text) > 903 else text

        add_raw_event(
            "interaction_started",
            stage=str(stage or "test"),
            user_id=str(u_raw),
            user_int=int(u_int),
            policy_source=str(policy_source),
            user_advisor_round_limit=int(max_rounds),
            advisor_discussion_round_limit=max(1, int(getattr(self.args, "com_advisor_discussion_rounds", 1) or 1)),
            candidate_count=int(len(cands_int or [])),
            target_alignment=dict(initial_target_alignment),
        )
        add_raw_event("target_profile_disabled", reason="profile summary is not injected into LLM prompts")

        for ridx in range(1, max_rounds + 1):
            update_llm_prompt_trace_context(
                round=int(ridx),
                phase="stage1_item_skill_retrieval",
                advisor_index="",
                advisor_id="",
                advisor_type="",
                path_why="",
                path_who="",
                path_how="",
            )
            shared_memory = self._render_shared_memory(host, memory_window)
            proposal_skill_context = {
                "phase": "item_selection",
                "task": "choose one current favorite item and preserve a genuine hesitation shortlist from the current candidates",
                "candidate_items": list(candidate_names),
                "prior_hint": prior_hint,
                "shortlist": [],
                "uncertainty_points": list((absorbed_memory or {}).get("remaining_uncertainty", []) or []),
                "updated_memory": dict(absorbed_memory or {}),
                "primary_trigger": "",
            }
            if proposal_slim_cache is None:
                cached_stage1_slim = self._load_cached_stage1_proposal_slim(u_raw)
                if cached_stage1_slim:
                    proposal_slim_compact = cached_stage1_slim
                    add_raw_event(
                        "proposal_slim_loaded_from_cache",
                        round=int(ridx),
                        source="slim_cache_proposal",
                        cache_mode="user_cached_stage1_slim",
                        chars=int(len(str(proposal_slim_compact))),
                    )
                else:
                    proposal_slim_compact = self._retrieve_relevant_skill_from_full_policy(
                        full_user_policy,
                        proposal_skill_context,
                        args=user_agent.args,
                    )
                proposal_slim_cache = proposal_slim_compact
            else:
                proposal_slim_compact = proposal_slim_cache
                add_raw_event("proposal_slim_reused_after_continue", round=int(ridx))
            update_llm_prompt_trace_context(round=int(ridx), phase="stage1_item_proposal")
            stage_key_for_log = str(stage or "").lower()
            should_log_proposal_slim = (
                int(ridx) == 1
                and (
                stage_key_for_log == "train"
                or (
                    stage_key_for_log == "test"
                    and bool(getattr(self.args, "com_stage1_only", False))
                )
                )
            )
            if should_log_proposal_slim:
                proposal_skill_log_source = "full_policy_retrieval"
                self.user_policy_store.cache_slim_policy(
                    u_raw,
                    {"source": proposal_skill_log_source, "retrieval_context": proposal_skill_context},
                    phase="proposal",
                    round_info={
                        "round": ridx,
                        "user_id": str(u_raw),
                        "stage": stage_key_for_log,
                        "call_site": "stage1_proposal_skill_retrieval",
                        "policy_source": str(policy_source),
                        "target_item_names": list(gt_item_names),
                    },
                    compact_slim_policy=proposal_slim_compact,
                )
            if stage1_round_cache is not None:
                previous_arbitration = dict((redecision_packet or {}).get("arbitration", {}) or {})
                previous_name = str((redecision_packet or {}).get("revised_name", "") or stage1_round_cache.get("proposal_name", ""))
                previous_reason = str((redecision_packet or {}).get("revised_reason", "") or stage1_round_cache.get("proposal_reason", ""))
                previous_confidence = previous_arbitration.get(
                    "decision_confidence",
                    stage1_round_cache.get("proposal_confidence", 60),
                )
                proposal_response = (
                    previous_reason,
                    previous_name,
                    previous_confidence,
                    list(active_hesitation_set or stage1_round_cache.get("shortlist_names", []) or []),
                    list(stage1_round_cache.get("candidate_evidence", []) or []),
                    str(stage1_round_cache.get("hesitation_reason", "") or ""),
                )
                add_raw_event(
                    "stage1_proposal_reused_after_continue",
                    round=int(ridx),
                    reused_item=str(previous_name),
                    reused_hesitation_set=list(active_hesitation_set or stage1_round_cache.get("shortlist_names", []) or []),
                )
            else:
                proposal_response = user_agent.propose_decision(
                    history_str=history_str,
                    candidate_names=candidate_names,
                    prior_hint=prior_hint,
                    target_profile=target_profile,
                    slim_user_policy=proposal_slim_compact,
                    updated_memory=str(absorbed_memory or "none"),
                )
            if len(proposal_response) >= 6:
                proposal_reason, proposal_name, proposal_confidence, candidate_shortlist, candidate_evidence, hesitation_reason = proposal_response[:6]
            elif len(proposal_response) >= 5:
                proposal_reason, proposal_name, proposal_confidence, candidate_shortlist, candidate_evidence = proposal_response[:5]
                hesitation_reason = ""
            elif len(proposal_response) >= 4:
                proposal_reason, proposal_name, proposal_confidence, candidate_shortlist = proposal_response[:4]
                candidate_evidence = []
                hesitation_reason = ""
            else:
                proposal_reason, proposal_name, proposal_confidence = proposal_response[:3]
                candidate_shortlist = []
                candidate_evidence = []
                hesitation_reason = ""
            proposal_iid, proposal_name, proposal_repair = self._resolve_proposal(
                host=host,
                proposal_name=proposal_name,
                cands_int=cands_int,
                sorted_candidate_ids=sorted_candidate_ids,
                prior_hint=prior_hint,
                candidate_shortlist=candidate_shortlist,
            )
            if proposal_repair.get("proposal_repaired"):
                add_raw_event("proposal_repaired_to_allowed_candidate", round=int(ridx), **proposal_repair)
            if not proposal_reason:
                proposal_reason = f"{proposal_name} is the strongest candidate under the current user preference evidence."
            candidate_shortlist = self._resolve_candidate_name_list(
                host=host,
                candidate_names=candidate_shortlist,
                cands_int=cands_int,
                limit=shortlist_limit,
            )

            shortlist_names = self._build_shortlist(
                host=host,
                sorted_candidate_ids=sorted_candidate_ids,
                proposal_name=proposal_name,
                prior_hint=prior_hint,
                absorbed_memory=absorbed_memory,
                limit=shortlist_limit,
                candidate_shortlist=candidate_shortlist,
            )
            injected_target_items = []
            if force_train_target_focus and not active_hesitation_set:
                shortlist_names, injected_target_items = self._inject_targets_into_shortlist(
                    shortlist_names=shortlist_names,
                    target_names=gt_item_names,
                    limit=shortlist_limit,
                    protected_names=[proposal_name, prior_hint],
                )
            if active_hesitation_set:
                shortlist_names = list(active_hesitation_set)
            if stage1_round_cache is None:
                stage1_round_cache = {
                    "proposal_name": str(proposal_name or ""),
                    "proposal_reason": str(proposal_reason or ""),
                    "proposal_confidence": int(self._clamp_confidence(proposal_confidence, default=60)),
                    "shortlist_names": list(shortlist_names or []),
                    "candidate_evidence": list(candidate_evidence or []),
                    "hesitation_reason": str(hesitation_reason or ""),
                    "injected_target_items": list(injected_target_items or []),
                }
            uncertainty_points = self._infer_uncertainty_points(
                host=host,
                history_str=history_str,
                target_profile=target_profile,
                proposal_name=proposal_name,
                proposal_reason=proposal_reason,
                shortlist_names=shortlist_names,
                proposal_iid=proposal_iid,
                sorted_candidate_ids=sorted_candidate_ids,
                score_dict=score_dict,
                prior_hint=prior_hint,
                absorbed_memory=absorbed_memory,
            )
            self_confidence = self._clamp_confidence(proposal_confidence, default=60)

            decision_context = {
                "phase": "communication",
                "proposal_item": proposal_name,
                "proposal_reason": proposal_reason,
                "shortlist": shortlist_names,
                "uncertainty_points": uncertainty_points,
                "self_confidence": self_confidence,
                "candidate_shortlist": list(shortlist_names),
                "candidate_evidence": list(candidate_evidence or []),
                "shortlist_semantics": "hesitation_uncertainty_set",
                "primary_trigger": self.infer_primary_trigger(
                    proposal_name=proposal_name,
                    shortlist=shortlist_names,
                    uncertainty_points=uncertainty_points,
                    prior_hint=prior_hint,
                    history_count=history_count,
                ),
                "target_injected_into_shortlist": bool(injected_target_items),
                "injected_target_items": list(injected_target_items),
                "proposal_repair": dict(proposal_repair or {}),
            }
            slim_user_policy_compact = proposal_slim_compact
            slim_user_policy = proposal_slim_compact
            if not bool(getattr(self.args, "com_stage1_only", False)):
                update_llm_prompt_trace_context(round=int(ridx), phase="communication_planning_view_retrieval")
                communication_condition = self._communication_condition_summary(
                    primary_trigger=str(decision_context.get("primary_trigger", "") or ""),
                    uncertainty_points=uncertainty_points,
                    self_confidence=self_confidence,
                    shortlist_names=shortlist_names,
                    candidate_evidence=candidate_evidence,
                )
                previous_user_feedback_for_planning = {}
                if int(ridx) > 1:
                    previous_arbitration = dict((redecision_packet or {}).get("arbitration", {}) or {})
                    previous_user_feedback_for_planning = compact_previous_user_feedback(previous_arbitration, absorbed_memory)
                communication_skill_context = {
                    "phase": "communication_selection",
                    "task": "retrieve structured communication planning view for deterministic planner",
                    "condition": communication_condition,
                    "planning_condition": {
                        "round_type": "repair" if int(ridx) > 1 else "initial",
                        "primary_trigger": str(decision_context.get("primary_trigger", "") or ""),
                        "uncertainty_shape": (
                            "candidate-conflict" if "candidate_comparison" in list(uncertainty_points or []) or len(shortlist_names or []) >= 2
                            else ("internal-prior-conflict" if "internal_prior_conflict" in list(uncertainty_points or []) else "proposal-risk-check")
                        ),
                        "confidence_band": "high" if int(self_confidence) >= 75 else ("medium" if int(self_confidence) >= 50 else "low"),
                        "focus_set_size": int(len(shortlist_names or [])),
                        "history_count": int(history_count or 0),
                        "history_sparsity": (
                            "sparse" if int(history_count or 0) <= 3
                            else ("medium" if int(history_count or 0) <= 8 else "rich")
                        ),
                        "prior_relation": "proposal_differs_from_prior" if prior_hint and self._normalize_text(prior_hint) != self._normalize_text(proposal_name) else "proposal_equals_prior",
                        "previous_feedback_exists": bool(previous_user_feedback_for_planning),
                    },
                    "uncertainty_points": list(uncertainty_points or []),
                    "self_confidence": self_confidence,
                    "focus_set_size": int(len(shortlist_names or [])),
                    "primary_trigger": str(decision_context.get("primary_trigger", "") or ""),
                    "updated_memory": dict(absorbed_memory or {}),
                }
                if communication_slim_cache is None:
                    slim_user_policy_compact = self.user_policy_store.build_slim_policy(
                        full_user_policy,
                        decision_context=communication_skill_context,
                    )
                    communication_slim_cache = slim_user_policy_compact
                else:
                    slim_user_policy_compact = communication_slim_cache
                    add_raw_event("communication_slim_reused_after_continue", round=int(ridx))
                slim_user_policy = slim_user_policy_compact
                if str(stage or "").lower() == "train" and int(ridx) == 1:
                    self.user_policy_store.cache_slim_policy(
                        u_raw,
                        {"source": "full_policy_retrieval", "retrieval_context": communication_skill_context},
                        phase="communication",
                        round_info={"round": ridx, "user_id": str(u_raw)},
                        compact_slim_policy=slim_user_policy_compact,
                    )
                update_llm_prompt_trace_context(round=int(ridx), phase="stage1_item_selection")

            preliminary_decision_state = self.build_decision_state(
                proposal_item=proposal_name,
                proposal_reason=proposal_reason,
                shortlist=shortlist_names,
                uncertainty_points=uncertainty_points,
                self_confidence=self_confidence,
                prior_hint=prior_hint,
                history_count=history_count,
                slim_user_policy=slim_user_policy_compact,
                communication_need=True,
            )
            preliminary_decision_state["target_injected_into_shortlist"] = bool(injected_target_items)
            preliminary_decision_state["injected_target_items"] = list(injected_target_items)
            preliminary_decision_state["candidate_shortlist"] = list(shortlist_names)
            preliminary_decision_state["round_type"] = "repair" if int(ridx) > 1 else "initial"
            if int(ridx) > 1:
                previous_arbitration = dict((redecision_packet or {}).get("arbitration", {}) or {})
                preliminary_decision_state["previous_user_feedback"] = compact_previous_user_feedback(previous_arbitration, absorbed_memory)
            preliminary_decision_state["candidate_evidence"] = list(candidate_evidence or [])
            preliminary_decision_state["shortlist_semantics"] = "hesitation_uncertainty_set"
            preliminary_decision_state["proposal_repair"] = dict(proposal_repair or {})
            preliminary_decision_state["history_count"] = int(history_count or 0)
            preliminary_decision_state["history_sparsity"] = (
                "sparse" if int(history_count or 0) <= 3
                else ("medium" if int(history_count or 0) <= 8 else "rich")
            )
            stage1_hesitation_alignment = self._target_alignment(
                candidate_names,
                gt_item_names,
                focus_names=list(shortlist_names),
            )
            stage1_scope_names = merge_unique([proposal_name] + list(shortlist_names or []))
            stage1_scope_alignment = self._target_alignment(
                candidate_names,
                gt_item_names,
                focus_names=list(stage1_scope_names),
            )
            proposal_alignment = self._target_alignment(
                candidate_names,
                gt_item_names,
                focus_names=[proposal_name],
            )
            target_scope_eligible = bool(stage1_scope_alignment.get("focus_target_overlap", False))
            previous_arbitration_for_gate = dict((redecision_packet or {}).get("arbitration", {}) or {}) if int(ridx) > 1 else {}
            previous_feedback_text_for_gate = str(preliminary_decision_state.get("previous_user_feedback", "") or "").strip()
            previous_decision_state_for_gate = str(previous_arbitration_for_gate.get("decision_state", "") or "").strip().lower()
            followup_requested_by_redecision = (
                int(ridx) > 1
                and previous_decision_state_for_gate == "continue"
                and bool(previous_feedback_text_for_gate)
            )
            communication_train_eligible = bool(target_scope_eligible or followup_requested_by_redecision)
            if target_scope_eligible:
                communication_gate_reason = "target_in_stage1_decision_scope"
            elif followup_requested_by_redecision:
                communication_gate_reason = "followup_requested_by_redecision"
            else:
                communication_gate_reason = "target_not_in_stage1_proposal_or_hesitation"
            communication_gate = {
                "eligible": bool(communication_train_eligible),
                "reason": communication_gate_reason,
                "target_scope_eligible": bool(target_scope_eligible),
                "followup_requested_by_redecision": bool(followup_requested_by_redecision),
                "communication_continuation": bool(followup_requested_by_redecision),
                "communication_target_gate_exempt": bool(followup_requested_by_redecision),
                "previous_feedback_present": bool(previous_feedback_text_for_gate),
                "previous_decision_state": previous_decision_state_for_gate,
                "target_absent_but_followup_allowed": bool(followup_requested_by_redecision and not target_scope_eligible),
                "target_in_base_candidates": bool(stage1_scope_alignment.get("candidate_target_overlap", False)),
                "target_in_initial_proposal": bool(proposal_alignment.get("focus_target_overlap", False)),
                "target_in_hesitation_shortlist": bool(stage1_hesitation_alignment.get("focus_target_overlap", False)),
                "target_in_stage1_decision_scope": bool(stage1_scope_alignment.get("focus_target_overlap", False)),
                "target_in_focus_candidates": bool(stage1_scope_alignment.get("focus_target_overlap", False)),
            }
            preliminary_decision_state["communication_training_gate"] = dict(communication_gate)

            stage_key_for_gate = str(stage or "").lower()
            skip_ineligible_communication = (
                bool(gt_item_names)
                and bool(getattr(self.args, "com_skip_ineligible_advisor_cost", True))
                and not bool(getattr(self.args, "com_stage1_only", False))
                and not communication_train_eligible
                and (
                    stage_key_for_gate != "train"
                    or bool(getattr(self.args, "com_train_communication_eligible_only", True))
                )
            )
            if skip_ineligible_communication:
                communication_action = "training_gate_skip"
                decision_state = dict(preliminary_decision_state)
                decision_state["communication_need"] = False
                decision_state["communication_training_gate"] = dict(communication_gate)
                final_iid = int(proposal_iid) if proposal_iid is not None else int(sorted_candidate_ids[0])
                path = self._build_skip_path()
                path["pattern_source"] = "communication_training_gate"
                path["path_reason"] = ["communication gate skipped advisor calls because target was not in the Stage1 proposal or hesitation set"]
                redecision_packet = {
                    "revised_name": proposal_name,
                    "revised_reason": proposal_reason,
                    "revised_iid": final_iid,
                    "arbitration": {
                        "current_decision": "keep_initial_proposal",
                        "decision_item": proposal_name,
                        "decision_confidence": int(self_confidence),
                        "decision_state": "final",
                        "remaining_uncertainty": list(uncertainty_points),
                        "stop_reason": "target_not_in_stage1_proposal_or_hesitation",
                    },
                }
                execution_packet = {
                    "path": dict(path),
                    "focus_candidates": list(shortlist_names),
                    "proposal_slim_user_policy": proposal_slim_compact,
                    "communication_training_gate": dict(communication_gate),
                    "communication_train_eligible": False,
                    "communication_skipped_by_training_gate": True,
                    "skip_reason": "target_not_in_stage1_proposal_or_hesitation",
                    "advisor_profiles": [],
                    "advisor_feedbacks": [],
                    "committee_packet": {
                        "aggregation_mode": "summary_agent_v1",
                        "decision_policy": "information_only_no_vote",
                        "advisor_synthesis_packet": {
                            "decision_policy": "information_only_no_vote",
                            "source": "advisor_summary_agent_skipped",
                            "what_was_answered": "communication gate skipped advisor communication",
                            "candidate_summaries": {},
                            "task_specific_summary": {},
                            "interaction_summary": {
                                "main_agreements": [],
                                "main_disagreements": [],
                                "corrections_or_rebuttals": [],
                                "unresolved_conflicts": [],
                            },
                            "remaining_uncertainty": list(uncertainty_points),
                            "do_not_decide_winner": True,
                        },
                        "legacy_aggregation_used": False,
                        "proposal_support_count": 0,
                        "proposal_oppose_count": 0,
                        "discussion_result": "training_gate_skip",
                        "advisor_pool_empty": True,
                        "original_advisor_pool_empty": True,
                        "advisor_pool_rerouted": False,
                        "final_advisor_pool_empty": True,
                        "final_advisor_count": 0,
                    },
                    "advisor_pool_status": {
                        "original_advisor_pool_empty": True,
                        "advisor_pool_rerouted": False,
                        "final_advisor_pool_empty": True,
                        "final_advisor_count": 0,
                        "advisor_pool_empty": True,
                        "rerouted": False,
                        "attempts": [],
                    },
                }
                absorbed_memory = {
                    "helpful_points": [],
                    "rejected_points": [],
                    "candidate_comparison": {},
                    "alternative_items": [],
                    "advisor_reliability_observation": {},
                    "remaining_uncertainty": list(uncertainty_points),
                    "evidence_packet": {
                        "support": [],
                        "oppose": [],
                        "additional": [],
                        "alternatives": list(shortlist_names),
                        "decision_policy": "information_only_no_vote",
                        "advisor_pool_empty": True,
                    },
                }
                evaluation_result = self.evaluator.evaluate(
                    initial_item_name=proposal_name,
                    final_item_name=proposal_name,
                    gt_items=gt_item_names,
                    initial_confidence=self_confidence,
                    final_confidence=self_confidence,
                    prev_uncertainty=uncertainty_points,
                    curr_uncertainty=uncertainty_points,
                    path=path,
                    initial_item_id=proposal_iid,
                    final_item_id=proposal_iid,
                    gt_item_ids=gt_items,
                )
                evaluation_result = self._augment_evaluation(
                    evaluation_result=evaluation_result,
                    candidate_names=candidate_names,
                    target_names=gt_item_names,
                    focus_names=list(shortlist_names),
                    proposal_name=proposal_name,
                    final_item_name=proposal_name,
                    prior_hint=prior_hint,
                    execution_packet=execution_packet,
                    target_injected_into_focus=bool(injected_target_items),
                    proposal_iid=proposal_iid,
                    final_iid=proposal_iid,
                    target_item_ids=gt_items,
                )
                evaluation_result.update(
                    {
                        "communication_train_eligible": False,
                        "communication_skipped_by_training_gate": True,
                        "skip_reason": "target_not_in_stage1_proposal_or_hesitation",
                    }
                )
                evolution_update = {
                    "policy_updated": False,
                    "updated_stage": str(stage or "test"),
                    "evolution_skipped": True,
                    "skip_reason": "target_not_in_stage1_proposal_or_hesitation",
                }
                if str(stage or "").lower() == "train":
                    trace_context = self._build_evolution_trace_context(
                        stage=stage,
                        user_raw=u_raw,
                        ridx=ridx,
                        candidate_names=candidate_names,
                        gt_item_names=gt_item_names,
                        decision_state=decision_state,
                        path=path,
                        execution_packet=execution_packet,
                        absorbed_memory=absorbed_memory,
                        redecision_packet=redecision_packet,
                        evaluation_result=evaluation_result,
                        proposal_name=proposal_name,
                        final_item_name=proposal_name,
                    )
                    train_round_trace_contexts.append(dict(trace_context))
                    trace_context["round_trace_contexts"] = list(train_round_trace_contexts)
                    with self._evolution_lock:
                        full_user_policy, diagnosis = self.evolver.evolve(
                            engine=self,
                            user_raw=u_raw,
                            full_user_policy=full_user_policy,
                            path=path,
                            evaluation_result=evaluation_result,
                            trace_context=trace_context,
                        )
                    evolution_skipped = bool((diagnosis or {}).get("skipped", False))
                    evolution_update = {
                        "policy_updated": not evolution_skipped,
                        "updated_stage": "skipped" if evolution_skipped else "train_item_gate_skip",
                        "path": dict(path or {}),
                        "outcome_signal": str((evaluation_result or {}).get("outcome_signal", "") or ""),
                        "diagnosis": dict(diagnosis or {}),
                        "communication_skipped_by_training_gate": True,
                    }
                    event_name = (
                        "communication_gate_item_evolution_skipped"
                        if evolution_skipped
                        else "communication_gate_item_evolution_applied"
                    )
                    add_raw_event(event_name, round=int(ridx), evolution_update=dict(evolution_update))
                add_raw_event(
                    "communication_training_gate_skipped",
                    round=int(ridx),
                    communication_training_gate=dict(communication_gate),
                    stage1_proposal_item=str(proposal_name),
                    stage1_hesitation_shortlist=list(shortlist_names),
                )
                structured_rounds.append(
                    self._build_round_trace(
                        ridx=ridx,
                        decision_state=decision_state,
                        communication_action=communication_action,
                        slim_user_policy=proposal_slim_compact,
                        path=path,
                        advisor_profiles=[],
                        execution_packet=execution_packet,
                        discussion_result=absorbed_memory,
                        redecision_packet=redecision_packet,
                        evaluation_result=evaluation_result,
                        evolution_update=evolution_update,
                    )
                )
                break

            if bool(getattr(self.args, "com_stage1_only", False)):
                communication_action = "stage1_only"
                decision_state = dict(preliminary_decision_state)
                decision_state["communication_need"] = False
                final_iid = int(proposal_iid) if proposal_iid is not None else int(sorted_candidate_ids[0])
                path = self._build_skip_path()
                path["path_reason"] = ["stage1-only user-skill training; communication/advisor/redecision skipped"]
                redecision_packet = {
                    "revised_name": proposal_name,
                    "revised_reason": proposal_reason,
                    "revised_iid": final_iid,
                    "arbitration": {
                        "current_decision": "keep_initial_proposal",
                        "decision_item": proposal_name,
                        "decision_confidence": int(self_confidence),
                        "decision_state": "final",
                        "remaining_uncertainty": list(uncertainty_points),
                        "stop_reason": "stage1-only user-skill training",
                    },
                }
                execution_packet = {
                    "path": dict(path),
                    "focus_candidates": list(shortlist_names),
                    "proposal_slim_user_policy": proposal_slim_compact,
                    "advisor_profiles": [],
                    "advisor_feedbacks": [],
                    "committee_packet": {
                        "aggregation_mode": "summary_agent_v1",
                        "decision_policy": "information_only_no_vote",
                        "advisor_synthesis_packet": {
                            "decision_policy": "information_only_no_vote",
                            "source": "advisor_summary_agent_skipped",
                            "what_was_answered": "stage1-only; no advisor communication executed",
                            "candidate_summaries": {},
                            "task_specific_summary": {},
                            "interaction_summary": {
                                "main_agreements": [],
                                "main_disagreements": [],
                                "corrections_or_rebuttals": [],
                                "unresolved_conflicts": [],
                            },
                            "remaining_uncertainty": list(uncertainty_points),
                            "do_not_decide_winner": True,
                        },
                        "legacy_aggregation_used": False,
                        "proposal_support_count": 0,
                        "proposal_oppose_count": 0,
                        "discussion_result": "stage1_only",
                        "advisor_pool_empty": True,
                        "original_advisor_pool_empty": True,
                        "advisor_pool_rerouted": False,
                        "final_advisor_pool_empty": True,
                        "final_advisor_count": 0,
                    },
                    "advisor_pool_status": {
                        "original_advisor_pool_empty": True,
                        "advisor_pool_rerouted": False,
                        "final_advisor_pool_empty": True,
                        "final_advisor_count": 0,
                        "advisor_pool_empty": True,
                        "rerouted": False,
                        "attempts": [],
                    },
                }
                absorbed_memory = {
                    "helpful_points": ["stage1-only user initial selection evaluated"],
                    "rejected_points": [],
                    "candidate_comparison": {},
                    "alternative_items": [],
                    "advisor_reliability_observation": {},
                    "remaining_uncertainty": list(uncertainty_points),
                    "evidence_packet": {
                        "support": [],
                        "oppose": [],
                        "additional": [],
                        "alternatives": list(shortlist_names),
                        "decision_policy": "information_only_no_vote",
                    },
                }
                evaluation_result = self.evaluator.evaluate(
                    initial_item_name=proposal_name,
                    final_item_name=proposal_name,
                    gt_items=gt_item_names,
                    initial_confidence=self_confidence,
                    final_confidence=self_confidence,
                    prev_uncertainty=uncertainty_points,
                    curr_uncertainty=uncertainty_points,
                    path=path,
                    initial_item_id=proposal_iid,
                    final_item_id=proposal_iid,
                    gt_item_ids=gt_items,
                )
                evaluation_result = self._augment_evaluation(
                    evaluation_result=evaluation_result,
                    candidate_names=candidate_names,
                    target_names=gt_item_names,
                    focus_names=list(shortlist_names),
                    proposal_name=proposal_name,
                    final_item_name=proposal_name,
                    prior_hint=prior_hint,
                    execution_packet=execution_packet,
                    target_injected_into_focus=bool(injected_target_items),
                    proposal_iid=proposal_iid,
                    final_iid=proposal_iid,
                    target_item_ids=gt_items,
                )
                evaluation_result["stage1_only"] = True
                evaluation_result["target_in_hesitation_shortlist"] = bool(evaluation_result.get("focus_target_overlap", False))
                add_raw_event(
                    "stage1_only_evaluation_ready",
                    round=int(ridx),
                    proposal_item=str(proposal_name),
                    shortlist=list(shortlist_names),
                    evaluation_result=dict(evaluation_result or {}),
                )

                evolution_update = {}
                if str(stage or "").lower() == "train":
                    trace_context = self._build_evolution_trace_context(
                        stage=stage,
                        user_raw=u_raw,
                        ridx=ridx,
                        candidate_names=candidate_names,
                        gt_item_names=gt_item_names,
                        decision_state=decision_state,
                        path=path,
                        execution_packet=execution_packet,
                        absorbed_memory=absorbed_memory,
                        redecision_packet=redecision_packet,
                        evaluation_result=evaluation_result,
                        proposal_name=proposal_name,
                        final_item_name=proposal_name,
                    )
                    trace_context["stage1_only"] = True
                    with self._evolution_lock:
                        full_user_policy, diagnosis = self.evolver.evolve(
                            engine=self,
                            user_raw=u_raw,
                            full_user_policy=full_user_policy,
                            path=path,
                            evaluation_result=evaluation_result,
                            trace_context=trace_context,
                        )
                    evolution_skipped = bool((diagnosis or {}).get("skipped", False))
                    evolution_update = {
                        "policy_updated": not evolution_skipped,
                        "updated_stage": "skipped" if evolution_skipped else "train_stage1_only",
                        "path": dict(path or {}),
                        "outcome_signal": str((evaluation_result or {}).get("outcome_signal", "") or ""),
                        "diagnosis": dict(diagnosis or {}),
                    }
                    event_name = "stage1_only_user_skill_evolution_skipped" if evolution_skipped else "stage1_only_user_skill_evolution_applied"
                    add_raw_event(event_name, round=int(ridx), evolution_update=dict(evolution_update))
                structured_rounds.append(
                    self._build_round_trace(
                        ridx=ridx,
                        decision_state=decision_state,
                        communication_action=communication_action,
                        slim_user_policy=slim_user_policy,
                        path=path,
                        advisor_profiles=[],
                        execution_packet=execution_packet,
                        discussion_result=absorbed_memory,
                        redecision_packet=redecision_packet,
                        evaluation_result=evaluation_result,
                        evolution_update=evolution_update,
                    )
                )
                add_raw_event("stage1_only_finalized", round=int(ridx), final_iid=int(final_iid), final_item=str(proposal_name))
                break

            update_llm_prompt_trace_context(
                round=int(ridx),
                phase="communication_path_planning",
                advisor_index="",
                advisor_id="",
                advisor_type="",
            )
            if int(ridx) == 1:
                preliminary_decision_state["user_id"] = str(u_raw or "")
                why_preview = self.path_selector.match_why_set(preliminary_decision_state, slim_user_policy)
                preliminary_decision_state["primary_why"] = str(why_preview.get("primary_why", "") or "")
                preliminary_decision_state["matched_why"] = list(why_preview.get("matched_why", []) or [])
                preliminary_decision_state["why_reasons"] = list(why_preview.get("why_reasons", []) or [])
                task_packet = generate_first_round_task(
                    args=self.args,
                    hesitation_set=shortlist_names,
                    hesitation_reason=hesitation_reason,
                    hesitation_evidence=candidate_evidence,
                    uncertainty_points=uncertainty_points,
                    item_slim_skill=proposal_slim_compact,
                    matched_why=str(why_preview.get("primary_why", "") or preliminary_decision_state.get("primary_trigger", "") or ""),
                    matched_whens=list(why_preview.get("matched_why", []) or []),
                    primary_why=str(why_preview.get("primary_why", "") or ""),
                )
                preliminary_decision_state["task_packet"] = dict(task_packet)
                preliminary_decision_state["user_task"] = str(task_packet.get("user_task", "") or "")
                preliminary_decision_state["task_type_hint"] = str(task_packet.get("task_type_hint", "") or "")
                preliminary_decision_state["what"] = str(task_packet.get("what", "") or "none")
                preliminary_decision_state["secondary_what"] = list(task_packet.get("secondary_what", []) or [])
                preliminary_decision_state["criteria"] = list(task_packet.get("criteria", []) or [])
                preliminary_decision_state["mapping_confidence"] = str(task_packet.get("mapping_confidence", "") or "")
                preliminary_decision_state["task_source"] = str(task_packet.get("task_source", "") or "")
                preliminary_decision_state["unmapped_task"] = bool(task_packet.get("unmapped_task", False))
                add_raw_event(
                    "user_task_generated",
                    round=int(ridx),
                    task_packet=dict(task_packet),
                )
                path = self.select_path(preliminary_decision_state, slim_user_policy, path_choice=None)
                first_communication_path_cache = dict(path or {})
            else:
                previous_arbitration = dict((redecision_packet or {}).get("arbitration", {}) or {})
                feedback_task = compact_previous_user_feedback(previous_arbitration, absorbed_memory)
                task_packet = task_packet_from_feedback(
                    feedback_task,
                    previous_how=str((first_communication_path_cache or {}).get("how", "") or first_communication_how),
                    advisor_count=len(advisor_profiles_cache or []),
                    hesitation_set=shortlist_names,
                )
                base_path = dict(first_communication_path_cache or path or {})
                preliminary_decision_state["task_packet"] = dict(task_packet)
                preliminary_decision_state["user_task"] = str(task_packet.get("user_task", "") or "")
                preliminary_decision_state["task_type_hint"] = str(task_packet.get("task_type_hint", "") or "")
                preliminary_decision_state["what"] = ""
                preliminary_decision_state["secondary_what"] = list(task_packet.get("secondary_what", []) or [])
                preliminary_decision_state["criteria"] = list(task_packet.get("criteria", []) or [])
                preliminary_decision_state["mapping_confidence"] = str(task_packet.get("mapping_confidence", "") or "")
                preliminary_decision_state["task_source"] = "feedback_to_advisors"
                preliminary_decision_state["unmapped_task"] = bool(task_packet.get("unmapped_task", False))
                preliminary_decision_state["tree_need_signals"] = list(task_packet.get("tree_need_signals", []) or [])
                preliminary_decision_state["followup_base_path"] = dict(base_path)
                preliminary_decision_state["followup_of_round"] = int(ridx) - 1
                path = self.select_path(preliminary_decision_state, slim_user_policy, path_choice=None)
                if task_packet.get("tree_need_signals"):
                    path["tree_need_signals"] = list(task_packet.get("tree_need_signals", []) or [])
                    path["path_reason"] = list(path.get("path_reason", []) or []) + [
                        "unmapped follow-up task recorded as what-tree growth signal"
                    ]
                path.setdefault("planner_log", {})
                path["planner_log"]["task_packet"] = dict(task_packet)
                path["planner_log"]["advisor_group_source"] = str(path.get("advisor_group_source", "") or "route_skill_followup")
                preliminary_decision_state["what"] = str(path.get("what", "") or "none")
                preliminary_decision_state["secondary_what"] = list(task_packet.get("secondary_what", []) or [])
                preliminary_decision_state["criteria"] = list(task_packet.get("criteria", []) or [])
                preliminary_decision_state["mapping_confidence"] = str(task_packet.get("mapping_confidence", "") or "")
                preliminary_decision_state["task_source"] = "feedback_to_advisors"
                preliminary_decision_state["unmapped_task"] = bool(task_packet.get("unmapped_task", False))
                preliminary_decision_state["tree_need_signals"] = list(task_packet.get("tree_need_signals", []) or [])
                add_raw_event(
                    "continued_user_task_mapped",
                    round=int(ridx),
                    task_packet=dict(task_packet),
                    path=dict(path),
                )
            path_choice = {
                'why': str((path or {}).get('why', "") or ""),
                "what": str((path or {}).get("what", "") or ""),
                "who": str((path or {}).get("who", "") or ""),
                "how": str((path or {}).get("how", "") or ""),
                "primary_why": str((path or {}).get("primary_why", "") or (path or {}).get('why', "") or ""),
                "matched_why": list((path or {}).get("matched_why", []) or []),
                "why_reasons": list((path or {}).get("why_reasons", []) or []),
                "user_task": str((path or {}).get("user_task", "") or ""),
                "task_type_hint": str((path or {}).get("task_type_hint", "") or ""),
                "secondary_what": list((path or {}).get("secondary_what", []) or []),
                "criteria": list((path or {}).get("criteria", []) or []),
                "mapping_confidence": str((path or {}).get("mapping_confidence", "") or ""),
                "task_source": str((path or {}).get("task_source", "") or ""),
                "tree_need_signals": list((path or {}).get("tree_need_signals", []) or []),
                "confidence": float((path or {}).get("path_score", 0.0) or 0.0),
                "reasons": list((path or {}).get("path_reason", []) or []),
                "planner_log": dict((path or {}).get("planner_log", {}) or {}),
            }
            selected_why = str((path or {}).get('why', "") or "").strip().lower()
            if selected_why in ["", "none", "skip", "no", "null", "n/a"]:
                communication_action = "skip"
            else:
                communication_action = "start" if int(ridx) == 1 else "continue"
            communication_control = {
                "communication_action": communication_action,
                "selected_why": str((path or {}).get('why', "") or ""),
                "primary_why": str((path or {}).get("primary_why", "") or (path or {}).get('why', "") or ""),
                "matched_why_nodes": list((path or {}).get("matched_why", []) or ((path or {}).get("planner_log", {}) or {}).get("matched_why", []) or []),
                "why_reasons": list((path or {}).get("why_reasons", []) or []),
                "decision_confidence": int(self_confidence),
                "reasons": list((path or {}).get("path_reason", []) or []),
                "planner_log": dict((path or {}).get("planner_log", {}) or {}),
            }
            communication_need = communication_action in ["start", "continue"]

            decision_state = self.build_decision_state(
                proposal_item=proposal_name,
                proposal_reason=proposal_reason,
                shortlist=shortlist_names,
                uncertainty_points=uncertainty_points,
                self_confidence=self_confidence,
                prior_hint=prior_hint,
                history_count=history_count,
                slim_user_policy=slim_user_policy,
                communication_need=communication_need,
            )
            decision_state["target_injected_into_shortlist"] = bool(injected_target_items)
            decision_state["injected_target_items"] = list(injected_target_items)
            decision_state["candidate_shortlist"] = list(shortlist_names)
            decision_state["candidate_evidence"] = list(candidate_evidence or [])
            decision_state["hesitation_reason"] = str(hesitation_reason or "")
            decision_state["shortlist_semantics"] = "hesitation_uncertainty_set"
            decision_state["task_packet"] = dict(preliminary_decision_state.get("task_packet", {}) or {})
            decision_state["user_task"] = str((path or {}).get("user_task", "") or preliminary_decision_state.get("user_task", "") or "")
            decision_state["what"] = str((path or {}).get("what", "") or preliminary_decision_state.get("what", "") or "none")
            decision_state["secondary_what"] = list((path or {}).get("secondary_what", []) or preliminary_decision_state.get("secondary_what", []) or [])
            decision_state["criteria"] = list((path or {}).get("criteria", []) or preliminary_decision_state.get("criteria", []) or [])
            decision_state["mapping_confidence"] = str((path or {}).get("mapping_confidence", "") or preliminary_decision_state.get("mapping_confidence", "") or "")
            decision_state["task_source"] = str((path or {}).get("task_source", "") or preliminary_decision_state.get("task_source", "") or "")
            decision_state["unmapped_task"] = bool((path or {}).get("unmapped_task", preliminary_decision_state.get("unmapped_task", False)))
            decision_state["communication_training_gate"] = dict(communication_gate)
            decision_state["communication_continuation"] = bool(followup_requested_by_redecision)
            decision_state["communication_target_gate_exempt"] = bool(followup_requested_by_redecision)
            decision_state["round_type"] = "repair" if int(ridx) > 1 else "initial"
            if int(ridx) > 1:
                decision_state["previous_user_feedback"] = str(preliminary_decision_state.get("previous_user_feedback", "") or "")
            decision_state["why_matching"] = {
                "matched_why_nodes": list((communication_control or {}).get("matched_why_nodes", []) or (path_choice or {}).get("matched_why_nodes", []) or []),
                "selected_why": str((communication_control or {}).get("selected_why", "") or (path_choice or {}).get('why', "") or ""),
                "primary_why": str((communication_control or {}).get("primary_why", "") or (path_choice or {}).get("primary_why", "") or (path_choice or {}).get('why', "") or ""),
                "why_reasons": list((communication_control or {}).get("why_reasons", []) or (path_choice or {}).get("why_reasons", []) or []),
                "no_why_matched": not communication_need,
                "no_communication_reason": str((communication_control or {}).get("no_communication_reason", "") or ""),
            }
            add_raw_event(
                "decision_state_built",
                round=int(ridx),
                proposal_item=str(proposal_name),
                primary_trigger=str(decision_state.get("primary_trigger", "")),
                uncertainty_points=list(uncertainty_points),
                self_confidence=int(self_confidence),
                communication_action=str(communication_action),
                communication_control=dict(communication_control or {}),
                communication_path_choice=dict(path_choice or {}),
                why_matching=dict(decision_state.get("why_matching", {}) or {}),
                planner_log=dict((path or {}).get("planner_log", {}) or {}),
                target_injected_into_shortlist=bool(injected_target_items),
                injected_target_items=list(injected_target_items),
                candidate_shortlist=list(shortlist_names),
            )

            if not communication_need:
                final_iid = int(proposal_iid) if proposal_iid is not None else int(sorted_candidate_ids[0])
                if not path or str((path or {}).get('why', "") or "") != "skip":
                    path = self._build_skip_path()
                evaluation_result = self.evaluator.evaluate(
                    initial_item_name=proposal_name,
                    final_item_name=proposal_name,
                    gt_items=gt_item_names,
                    initial_confidence=self_confidence,
                    final_confidence=self_confidence,
                    prev_uncertainty=uncertainty_points,
                    curr_uncertainty=uncertainty_points,
                    path=path,
                    initial_item_id=proposal_iid,
                    final_item_id=final_iid,
                    gt_item_ids=gt_items,
                )
                redecision_packet = {
                    "revised_name": proposal_name,
                    "revised_reason": proposal_reason,
                    "revised_iid": final_iid,
                    "arbitration": {
                        "current_decision": "keep",
                        "decision_item": proposal_name,
                        "decision_confidence": int(self_confidence),
                        "decision_state": "final",
                        "remaining_uncertainty": list(uncertainty_points),
                        "stop_reason": "communication skipped because the decision is already stable",
                    },
                }
                execution_packet = {
                    "path": dict(path),
                    "focus_candidates": list(shortlist_names),
                    "advisor_profiles": [],
                    "advisor_feedbacks": [],
                    "committee_packet": {
                        "aggregation_mode": "summary_agent_v1",
                        "decision_policy": "information_only_no_vote",
                        "advisor_synthesis_packet": {
                            "decision_policy": "information_only_no_vote",
                            "source": "advisor_summary_agent_skipped",
                            "what_was_answered": "communication skipped because decision was already stable",
                            "candidate_summaries": {},
                            "task_specific_summary": {},
                            "interaction_summary": {
                                "main_agreements": [],
                                "main_disagreements": [],
                                "corrections_or_rebuttals": [],
                                "unresolved_conflicts": [],
                            },
                            "remaining_uncertainty": list(uncertainty_points),
                            "do_not_decide_winner": True,
                        },
                        "legacy_aggregation_used": False,
                        "proposal_support_count": 0,
                        "proposal_oppose_count": 0,
                    },
                }
                absorbed_memory = {
                    "helpful_points": ["internal proposal is already stable"],
                    "rejected_points": [],
                    "candidate_comparison": {},
                    "alternative_items": [],
                    "advisor_reliability_observation": {},
                    "remaining_uncertainty": list(uncertainty_points),
                    "evidence_packet": {"support": [], "oppose": [], "additional": [], "alternatives": [], "decision_policy": "information_only_no_vote"},
                }
                evaluation_result = self._augment_evaluation(
                    evaluation_result=evaluation_result,
                    candidate_names=candidate_names,
                    target_names=gt_item_names,
                    focus_names=list(shortlist_names),
                    proposal_name=proposal_name,
                    final_item_name=proposal_name,
                    prior_hint=prior_hint,
                    execution_packet=execution_packet,
                    target_injected_into_focus=bool(injected_target_items),
                    proposal_iid=proposal_iid,
                    final_iid=final_iid,
                    target_item_ids=gt_items,
                )
                evaluation_result["communication_train_eligible"] = bool(communication_train_eligible)
                evaluation_result["communication_skipped_by_training_gate"] = False
                evaluation_result["communication_continuation"] = bool(followup_requested_by_redecision)
                evaluation_result["communication_target_gate_exempt"] = bool(followup_requested_by_redecision)
                evolution_update = {}
                if str(stage or "").lower() == "train":
                    trace_context = self._build_evolution_trace_context(
                        stage=stage,
                        user_raw=u_raw,
                        ridx=ridx,
                        candidate_names=candidate_names,
                        gt_item_names=gt_item_names,
                        decision_state=decision_state,
                        path=path,
                        execution_packet=execution_packet,
                        absorbed_memory=absorbed_memory,
                        redecision_packet=redecision_packet,
                        evaluation_result=evaluation_result,
                        proposal_name=proposal_name,
                        final_item_name=proposal_name,
                    )
                    train_round_trace_contexts.append(dict(trace_context))
                    trace_context["round_trace_contexts"] = list(train_round_trace_contexts)
                    with self._evolution_lock:
                        full_user_policy, diagnosis = self.evolver.evolve(
                            engine=self,
                            user_raw=u_raw,
                            full_user_policy=full_user_policy,
                            path=path,
                            evaluation_result=evaluation_result,
                            trace_context=trace_context,
                        )
                    evolution_skipped = bool((diagnosis or {}).get("skipped", False))
                    evolution_update = {
                        "policy_updated": not evolution_skipped,
                        "updated_stage": "skipped" if evolution_skipped else "train",
                        "path": dict(path or {}),
                        "outcome_signal": str((evaluation_result or {}).get("outcome_signal", "") or ""),
                        "diagnosis": dict(diagnosis or {}),
                    }
                    event_name = "train_evolution_skipped" if evolution_skipped else "train_evolution_applied"
                    add_raw_event(event_name, round=int(ridx), evolution_update=dict(evolution_update))
                structured_rounds.append(
                    self._build_round_trace(
                        ridx=ridx,
                        decision_state=decision_state,
                        communication_action=communication_action,
                        slim_user_policy=slim_user_policy,
                        path=path,
                        advisor_profiles=[],
                        execution_packet=execution_packet,
                        discussion_result=absorbed_memory,
                        redecision_packet=redecision_packet,
                        evaluation_result=evaluation_result,
                        evolution_update=evolution_update,
                    )
                )
                add_raw_event("communication_skipped", round=int(ridx), final_item=str(proposal_name))
                break

            if int(ridx) > 1 and advisor_profiles_cache:
                advisor_profiles = list(advisor_profiles_cache)
                advisor_pool_status = {
                    "original_advisor_pool_empty": False,
                    "advisor_pool_rerouted": False,
                    "final_advisor_pool_empty": False,
                    "final_advisor_count": int(len(advisor_profiles)),
                    "advisor_pool_empty": False,
                    "rerouted": False,
                    "attempts": [],
                    "reused_previous_round_advisors": True,
                }
                advisor_pool_status["reused_previous_round_advisors"] = True
            else:
                path, advisor_profiles, advisor_pool_status = self._reroute_if_advisor_pool_empty(
                    host=host,
                    path=path,
                    slim_user_policy=slim_user_policy,
                    u_raw=u_raw,
                    u_int=u_int,
                    cands_int=cands_int,
                    proposal_iid=proposal_iid,
                    shortlist_names=shortlist_names,
                )
            round_type = "continued_user_task" if int(ridx) > 1 else "open_candidate_review"
            previous_user_feedback = {}
            if int(ridx) > 1:
                previous_arbitration = dict((redecision_packet or {}).get("arbitration", {}) or {})
                previous_user_feedback = compact_previous_user_feedback(previous_arbitration, absorbed_memory)
            elif not first_communication_how:
                first_communication_how = str(path.get("how", "") or "")
                first_communication_path_cache = dict(path or {})
            if advisor_profiles and advisor_profiles_cache is None:
                advisor_profiles_cache = list(advisor_profiles)
            add_raw_event(
                "path_selected",
                round=int(ridx),
                round_type=str(round_type),
                user_skill_path_choice=dict(path_choice or {}),
                path=dict(path),
                advisor_ids=[str(row.get("u_raw", "")) for row in (advisor_profiles or [])],
                advisor_trust_subbranches=[
                    {
                        "advisor_id": str(row.get("u_raw", "") or ""),
                        "trust_relation": str(row.get("trust_relation", "") or "none"),
                        "trust_scope": str(row.get("trust_scope", "") or "none"),
                        "history_similarity_bucket": str(row.get("history_similarity_bucket", "") or "none"),
                        "trust_subbranch": str(row.get("trust_subbranch", "") or "none"),
                    }
                    for row in (advisor_profiles or [])
                ],
                advisor_pool_status=dict(advisor_pool_status or {}),
            )

            requester_shareable_brief = None
            if bool(getattr(self.args, "com_enable_shareable_user_brief", True)):
                requester_shareable_brief = self.disclosure_builder.build_shareable_item_brief(
                    private_item_slim_skill=proposal_slim_compact,
                    stage1_state=decision_state,
                    focus_candidates=shortlist_names,
                    selected_why=path.get('why', ""),
                    selected_how=path.get("how", ""),
                )
            advisor_requester_context = requester_shareable_brief if requester_shareable_brief is not None else slim_user_policy
            update_llm_prompt_trace_context(
                round=int(ridx),
                phase="advisor_communication",
                path_why=str(path.get('why', "") or ""),
                path_what=str(path.get("what", "") or ""),
                path_who=str(path.get("who", "") or ""),
                path_how=str(path.get("how", "") or ""),
            )
            execution_packet = self.path_executor.execute(
                host=host,
                path=path,
                advisor_profiles=advisor_profiles,
                advisor_agent=advisor_agent,
                history_str=history_str,
                target_profile=target_profile,
                proposal_name=proposal_name,
                proposal_reason=proposal_reason,
                shortlist_names=shortlist_names,
                candidate_evidence=candidate_evidence,
                hesitation_reason=hesitation_reason,
                cands_int=cands_int,
                prior_hint=prior_hint,
                shared_memory=shared_memory,
                target_user_skill=advisor_requester_context,
                round_type=round_type,
                previous_user_feedback=previous_user_feedback,
                previous_discussion_memory=discussion_memory_cache if int(ridx) > 1 else None,
                previous_round_summary=absorbed_memory if int(ridx) > 1 else None,
            )
            if int(ridx) == 1:
                advisor_profiles_cache = list((execution_packet or {}).get("advisor_profiles", []) or advisor_profiles or [])
            discussion_memory_cache = list((execution_packet or {}).get("discussion_memory", []) or [])
            if requester_shareable_brief is not None:
                execution_packet["requester_shareable_item_brief"] = dict(requester_shareable_brief)
            if advisor_pool_status:
                execution_packet["advisor_pool_status"] = dict(advisor_pool_status)
                committee_packet = dict((execution_packet or {}).get("committee_packet", {}) or {})
                committee_packet["original_advisor_pool_empty"] = bool(advisor_pool_status.get("original_advisor_pool_empty", False))
                committee_packet["advisor_pool_rerouted"] = bool(advisor_pool_status.get("advisor_pool_rerouted", False))
                committee_packet["final_advisor_pool_empty"] = bool(advisor_pool_status.get("final_advisor_pool_empty", False))
                committee_packet["final_advisor_count"] = int(advisor_pool_status.get("final_advisor_count", len(advisor_profiles or [])) or 0)
                committee_packet["advisor_pool_empty"] = bool(advisor_pool_status.get("final_advisor_pool_empty", False))
                if advisor_pool_status.get("final_advisor_pool_empty"):
                    issues = list(committee_packet.get("protocol_issues", []) or [])
                    issues.append(
                        {
                            "advisor": "system",
                            "issue": "advisor_pool_empty",
                            "item": str(proposal_name),
                        }
                    )
                    committee_packet["protocol_issues"] = issues
                if advisor_pool_status.get("advisor_pool_rerouted"):
                    committee_packet["advisor_pool_rerouted"] = True
                execution_packet["committee_packet"] = committee_packet
            add_raw_event(
                "path_executed",
                round=int(ridx),
                executed_path=dict((execution_packet or {}).get("path", {}) or path or {}),
                focus_candidates=list((execution_packet or {}).get("focus_candidates", []) or []),
                committee_result=dict((execution_packet or {}).get("committee_packet", {}) or {}),
            )

            absorbed_memory = self._discussion_result_from_execution(
                decision_state=decision_state,
                path=path,
                execution_packet=execution_packet,
            )
            add_raw_event(
                "discussion_result_ready",
                round=int(ridx),
                discussion_result=dict(absorbed_memory or {}),
            )

            post_feedback_skill_context = {
                "phase": "redecision",
                "task": "choose after advisor discussion while applying any learned communication absorption rules",
                "proposal_item": str(proposal_name or ""),
                "proposal_reason": str(proposal_reason or ""),
                "shortlist": list(shortlist_names or []),
                "candidate_shortlist": list(shortlist_names or []),
                "uncertainty_points": list(uncertainty_points or []),
                "self_confidence": int(self_confidence or 0),
                "primary_trigger": str(decision_context.get("primary_trigger", "") or ""),
                "updated_memory": dict(absorbed_memory or {}),
                "planning_condition": {
                    "round_type": "repair" if int(ridx) > 1 else "initial",
                    "primary_trigger": str(decision_context.get("primary_trigger", "") or ""),
                    "uncertainty_shape": (
                        "candidate-conflict" if len(shortlist_names or []) >= 2
                        else str(decision_context.get("primary_trigger", "") or "proposal-risk-check")
                    ),
                    "confidence_band": "high" if int(self_confidence or 0) >= 75 else ("medium" if int(self_confidence or 0) >= 50 else "low"),
                    "focus_set_size": int(len(shortlist_names or [])),
                    "previous_feedback_exists": bool(int(ridx) > 1),
                },
            }
            # Reuse the compact skill distilled for the first item decision.
            # Rebuilding a redecision slim here can expand into a large dict
            # and dominate the post-feedback prompt.
            post_feedback_slim_compact = proposal_slim_compact or proposal_slim_cache or slim_user_policy_compact
            add_raw_event(
                "post_feedback_slim_ready",
                round=int(ridx),
                has_absorption_skill=bool(
                    isinstance(post_feedback_slim_compact, dict)
                    and post_feedback_slim_compact.get("communication_absorption_skill")
                ),
                absorption_rule_count=int(
                    len(
                        (
                            (post_feedback_slim_compact or {}).get("communication_absorption_skill", {})
                            if isinstance(post_feedback_slim_compact, dict)
                            else {}
                        ).get("active_rules", [])
                    )
                ),
                source="stage1_proposal_slim_reused_for_redecision",
            )
            if str(stage or "").lower() == "train":
                self.user_policy_store.cache_slim_policy(
                    u_raw,
                    {"source": "stage1_proposal_slim_reused_for_redecision", "retrieval_context": post_feedback_skill_context},
                    phase="redecision",
                    round_info={"round": ridx, "user_id": str(u_raw)},
                    compact_slim_policy=post_feedback_slim_compact,
                )

            update_llm_prompt_trace_context(
                round=int(ridx),
                phase="post_feedback_redecision",
                advisor_index="",
                advisor_id="",
                advisor_type="",
            )
            redecision_packet = self.redecision.redecide(
                host=host,
                user_agent=user_agent,
                decision_state=decision_state,
                discussion_result=absorbed_memory,
                path=path,
                candidate_names=candidate_names,
                cands_int=cands_int,
                prior_hint=prior_hint,
                target_profile=target_profile,
                history_str=history_str,
                shared_memory=shared_memory,
                slim_user_policy=post_feedback_slim_compact,
            )
            arbitration = dict((redecision_packet or {}).get("arbitration", {}) or {})
            if ridx >= max_rounds and str(arbitration.get("decision_state", "continue")) != "final":
                arbitration["decision_state"] = "final"
                arbitration["stop_reason"] = "maximum communication rounds reached; use the best current decision"
                arbitration["forced_final_due_to_round_limit"] = True
                redecision_packet["arbitration"] = arbitration
            add_raw_event(
                "user_redecision_ready",
                round=int(ridx),
                revised_item=str((redecision_packet or {}).get("revised_name", "") or ""),
                arbitration=dict(arbitration),
            )
            is_final_round = str(arbitration.get("decision_state", "continue") or "continue") == "final"
            if not is_final_round:
                next_focus_raw = list(arbitration.get("next_round_focus", []) or [])
                updated_focus = self._resolve_candidate_name_list(
                    host=host,
                    candidate_names=next_focus_raw,
                    cands_int=cands_int,
                    limit=shortlist_limit,
                )
                if not updated_focus:
                    removed_norm = {
                        self._normalize_text(x)
                        for x in list(arbitration.get("removed_from_hesitation", []) or [])
                        if self._normalize_text(x)
                    }
                    if removed_norm:
                        current_focus = list((execution_packet or {}).get("focus_candidates", []) or shortlist_names or [])
                        updated_focus = [
                            str(x)
                            for x in current_focus
                            if self._normalize_text(x) and self._normalize_text(x) not in removed_norm
                        ][:shortlist_limit]
                if updated_focus:
                    active_hesitation_set = list(updated_focus)
                    add_raw_event(
                        "hesitation_set_updated_after_feedback",
                        round=int(ridx),
                        next_hesitation_set=list(active_hesitation_set),
                        removed_from_hesitation=list(arbitration.get("removed_from_hesitation", []) or []),
                    )

            final_iid = int((redecision_packet or {}).get("revised_iid", proposal_iid) or proposal_iid or sorted_candidate_ids[0])
            final_item_name = str((redecision_packet or {}).get("revised_name", proposal_name) or proposal_name)
            evaluation_result = self.evaluator.evaluate(
                initial_item_name=proposal_name,
                final_item_name=final_item_name,
                gt_items=gt_item_names,
                initial_confidence=self_confidence,
                final_confidence=int(arbitration.get("decision_confidence", self_confidence) or self_confidence),
                prev_uncertainty=uncertainty_points,
                curr_uncertainty=list(arbitration.get("remaining_uncertainty", []) or []),
                path=path,
                initial_item_id=proposal_iid,
                final_item_id=final_iid,
                gt_item_ids=gt_items,
            )
            evaluation_result = self._augment_evaluation(
                evaluation_result=evaluation_result,
                candidate_names=candidate_names,
                target_names=gt_item_names,
                focus_names=list((execution_packet or {}).get("focus_candidates", []) or []),
                proposal_name=proposal_name,
                final_item_name=final_item_name,
                prior_hint=prior_hint,
                execution_packet=execution_packet,
                target_injected_into_focus=bool(injected_target_items),
                proposal_iid=proposal_iid,
                final_iid=final_iid,
                target_item_ids=gt_items,
            )
            evaluation_result["communication_train_eligible"] = bool(communication_train_eligible)
            evaluation_result["communication_skipped_by_training_gate"] = False
            evaluation_result["communication_continuation"] = bool(followup_requested_by_redecision)
            evaluation_result["communication_target_gate_exempt"] = bool(followup_requested_by_redecision)
            add_raw_event(
                "evaluation_ready",
                round=int(ridx),
                evaluation_result=dict(evaluation_result or {}),
            )

            evolution_update = {}
            if str(stage or "").lower() == "train" and is_final_round:
                trace_context = self._build_evolution_trace_context(
                    stage=stage,
                    user_raw=u_raw,
                    ridx=ridx,
                    candidate_names=candidate_names,
                    gt_item_names=gt_item_names,
                    decision_state=decision_state,
                    path=path,
                    execution_packet=execution_packet,
                    absorbed_memory=absorbed_memory,
                    redecision_packet=redecision_packet,
                    evaluation_result=evaluation_result,
                    proposal_name=proposal_name,
                    final_item_name=final_item_name,
                )
                train_round_trace_contexts.append(dict(trace_context))
                trace_context["round_trace_contexts"] = list(train_round_trace_contexts)
                with self._evolution_lock:
                    full_user_policy, diagnosis = self.evolver.evolve(
                        engine=self,
                        user_raw=u_raw,
                        full_user_policy=full_user_policy,
                        path=path,
                        evaluation_result=evaluation_result,
                        trace_context=trace_context,
                    )
                evolution_update = {
                    "policy_updated": True,
                    "updated_stage": "train",
                    "path": dict(path or {}),
                    "outcome_signal": str((evaluation_result or {}).get("outcome_signal", "") or ""),
                    "diagnosis": dict(diagnosis or {}),
                }
                add_raw_event("train_evolution_applied", round=int(ridx), evolution_update=dict(evolution_update))
            elif str(stage or "").lower() == "train":
                deferred_trace_context = self._build_evolution_trace_context(
                    stage=stage,
                    user_raw=u_raw,
                    ridx=ridx,
                    candidate_names=candidate_names,
                    gt_item_names=gt_item_names,
                    decision_state=decision_state,
                    path=path,
                    execution_packet=execution_packet,
                    absorbed_memory=absorbed_memory,
                    redecision_packet=redecision_packet,
                    evaluation_result=evaluation_result,
                    proposal_name=proposal_name,
                    final_item_name=final_item_name,
                )
                train_round_trace_contexts.append(dict(deferred_trace_context))
                evolution_update = {
                    "policy_updated": False,
                    "updated_stage": "train",
                    "evolution_deferred": True,
                    "defer_reason": "user requested continued communication; train evolution is applied only on final rounds",
                    "path": dict(path or {}),
                    "outcome_signal": str((evaluation_result or {}).get("outcome_signal", "") or ""),
                    "diagnosis": {},
                }
                add_raw_event("train_evolution_deferred", round=int(ridx), evolution_update=dict(evolution_update))

            self._append_memory(
                memory_window,
                "decision",
                f"Round {ridx} decision: proposal={proposal_name}; trigger={decision_state.get('primary_trigger', '')}; confidence={self_confidence}; uncertainty={', '.join(uncertainty_points) or 'none'}",
            )
            self._append_memory(
                memory_window,
                "path",
                f"Round {ridx} path: why={path.get('why', '')}; who={path.get('who', '')}; how={path.get('how', '')}",
            )
            committee = dict((execution_packet or {}).get("committee_packet", {}) or {})
            self._append_memory(
                memory_window,
                "committee",
                f"Round {ridx} advisor evidence: aggregation_mode={committee.get('aggregation_mode', 'summary_agent_v1')}; discussion_result={committee.get('discussion_result', 'unknown')}",
            )
            self._append_memory(
                memory_window,
                "discussion",
                f"Round {ridx} discussion result: source={(absorbed_memory or {}).get('source', 'direct_discussion_result')}; remaining={', '.join((absorbed_memory or {}).get('remaining_uncertainty', []) or []) or 'none'}",
            )
            self._append_memory(
                memory_window,
                "redecision",
                f"Round {ridx} user redecision: item={final_item_name}; decision={arbitration.get('current_decision', 'keep')}; confidence={arbitration.get('decision_confidence', self_confidence)}; state={arbitration.get('decision_state', 'continue')}",
            )

            structured_rounds.append(
                self._build_round_trace(
                    ridx=ridx,
                    decision_state=decision_state,
                    communication_action=communication_action,
                    slim_user_policy=slim_user_policy_compact,
                    path=path,
                    advisor_profiles=advisor_profiles,
                    execution_packet=execution_packet,
                    discussion_result=absorbed_memory,
                    redecision_packet=redecision_packet,
                    evaluation_result=evaluation_result,
                    evolution_update=evolution_update,
                )
            )

            if is_final_round:
                add_raw_event("interaction_finalized", round=int(ridx), final_iid=int(final_iid), final_item=str(final_item_name))
                break

        if final_iid is None:
            raise ValueError("COM interaction ended without a skill-driven final item.")

        first_round = dict(structured_rounds[0] if structured_rounds else {})
        first_eval = dict(first_round.get("stage6_evaluation", {}) or {})
        final_round = dict(structured_rounds[-1] if structured_rounds else {})
        final_eval = dict(final_round.get("stage6_evaluation", {}) or {})
        final_evolution = dict(final_round.get("stage7_train_evolution", {}) or {})
        final_diagnosis = dict(final_evolution.get("diagnosis", {}) or {})
        user_initial_hit = bool(first_eval.get("initial_hit", str(first_eval.get("outcome_signal", "") or "") in ["TT", "TW"]))
        user_final_hit = bool(final_eval.get("final_hit", str(final_eval.get("outcome_signal", "") or "") in ["TT", "WT"]))
        if user_initial_hit and user_final_hit:
            user_outcome_signal = "TT"
        elif user_initial_hit and not user_final_hit:
            user_outcome_signal = "TW"
        elif (not user_initial_hit) and user_final_hit:
            user_outcome_signal = "WT"
        else:
            user_outcome_signal = "WW"
        interaction_summary = {
            "outcome_signal": user_outcome_signal,
            "round_outcome_signal": str(final_eval.get("outcome_signal", "") or final_evolution.get("outcome_signal", "") or ""),
            "initial_hit": bool(user_initial_hit),
            "final_hit": bool(user_final_hit),
            "failure_attribution": str(final_diagnosis.get("failure_attribution", "") or ""),
            "primary_failure_level": str(final_diagnosis.get("primary_failure_level", "") or ""),
            "reason": str(
                final_diagnosis.get("path_effect_explanation")
                or final_diagnosis.get("communication_reflection_summary")
                or ((final_diagnosis.get("user_skill_diagnosis", {}) or {}).get("problem", ""))
                or ""
            ),
            "diagnosis_id": str(final_diagnosis.get("diagnosis_id", "") or ""),
            "communication_train_eligible": bool(
                final_eval.get("communication_train_eligible", final_diagnosis.get("communication_train_eligible", False))
            ),
        }
        clear_llm_prompt_trace()
        return int(final_iid), structured_rounds if collect_trace else [], raw_trace if collect_trace else [], interaction_summary
