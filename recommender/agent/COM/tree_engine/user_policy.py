import re
from pathlib import Path

from recommender.agent.COM.tree_engine.utils import append_jsonl, dump_json, load_json, merge_unique
from recommender.agent.COM.tree_engine.public_tree import PublicTreeStore


WHO_NODE_ALIASES = {}

HOW_NODE_ALIASES = {
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
    "debate": "multi-competitive",
    "pairwise-debate": "multi-competitive",
    "competitive": "multi-competitive",
    "multi-candidate-debate": "multi-competitive",
    "multi-competitive-warning": "multi-competitive",
    "multi-competitive-promotion": "multi-competitive",
    "feedback-competitive-repair": "multi-competitive",
}

ACTIVE_WHO_NODES = [
    "trusted-advisors",
    "similar-users",
    "experienced-users",
    "topk-advisors",
]

ACTIVE_WHEN_NODES = [
    "candidate-conflict",
    "cold-start",
    "internal-prior-conflict",
    "novelty-uncertainty",
]

ACTIVE_WHAT_NODES = [
    "reduce_hesitation_set",
    "find_interested_subset",
    "compare_remaining_candidates",
    "evidence_gap_check",
    "reasoning_check",
    "none",
]

ACTIVE_HOW_NODES = [
    "single-advisor",
    "multi-cooperative",
    "multi-competitive",
]

TRIGGER_SIGNATURE_PRIORITY = [
    "internal-prior-conflict",
    "candidate-conflict",
    "novelty-uncertainty",
    "cold-start",
    "missing-evidence",
]

DEFAULT_TRIGGER_SIGNATURES = [
    "internal-prior-conflict+candidate-conflict",
    "novelty-uncertainty+candidate-conflict",
    "cold-start+candidate-conflict",
    "candidate-conflict",
    "internal-prior-conflict",
    "novelty-uncertainty",
    "cold-start",
    "default",
]

ITEM_LAYER_ALIASES = {
    "core_decision_reasoning_skill": "item_selection_skill",
    "item_reasoning_skill": "item_selection_skill",
    "item_selection": "item_selection_skill",
}

COMMUNICATION_LAYER_ALIASES = {
    "communication_reasoning_skill": "communication_selection_skill",
    "communication_selection": "communication_selection_skill",
}

POST_FEEDBACK_LAYER_ALIASES = {
    "post_feedback_reasoning_skill": "post_feedback_skill",
    "post_feedback": "post_feedback_skill",
    "post-feedback": "post_feedback_skill",
    "feedback": "post_feedback_skill",
    "redecision": "post_feedback_skill",
}

ABSORPTION_LAYER_ALIASES = {
    "communication_absorption": "communication_absorption_skill",
    "advisor_absorption": "communication_absorption_skill",
    "absorption": "communication_absorption_skill",
}


class UserPolicyStore:
    def __init__(self, base_dir, dataset, initial_base_dir=None, recommender_source=None):
        self.dataset = str(dataset or "default")
        self.recommender_source = self._safe_path_slug(recommender_source or "default_prior")
        self.policy_root = Path(base_dir)
        self.base_dir = self.policy_root / self.dataset
        self.public_tree_dir = self.policy_root.parent / "public_tree" / self._safe_path_slug(self.dataset)
        self.initial_base_dir = (
            Path(initial_base_dir) / self.dataset
            if initial_base_dir
            else None
        )

    @staticmethod
    def _safe_path_slug(value):
        text = str(value or "default_prior").strip()
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = text.strip("._-")
        return text or "default_prior"

    def _is_product_domain(self):
        return "epinions" in str(self.dataset or "").lower()

    def _is_book_domain(self):
        return "librarything" in str(self.dataset or "").lower()

    def _user_slug(self, user_raw):
        return f"user_{str(user_raw)}"

    def user_dir(self, user_raw):
        return self.base_dir / self._user_slug(user_raw)

    def _path_set_for_dir(self, udir):
        return {
            "dir": udir,
            "skill_md": udir / "SKILL.md",
            "policy_json": udir / "references" / "policy.json",
            "snapshots_jsonl": udir / "references" / "snapshots.jsonl",
            "evolution_log_jsonl": udir / "references" / "evolution_log.jsonl",
            "interaction_diagnoses_jsonl": udir / "references" / "interaction_diagnoses.jsonl",
            "slim_cache_json": udir / "assets" / "slim_cache.json",
            "slim_cache_proposal_json": udir / "assets" / "slim_cache_proposal.json",
            "slim_cache_communication_json": udir / "assets" / "slim_cache_communication.json",
            "slim_cache_post_feedback_json": udir / "assets" / "slim_cache_post_feedback.json",
            "slim_policy_log_jsonl": udir / "assets" / "slim_policy_log.jsonl",
        }

    def _paths(self, user_raw):
        return self._path_set_for_dir(self.user_dir(user_raw))

    def _legacy_base_dirs(self, root_dir):
        root = Path(root_dir)
        dataset_dir = root / self.dataset
        seen = set()

        def add(path):
            path = Path(path)
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen or path == dataset_dir:
                return
            seen.add(key)
            yield path

        for path in add(dataset_dir / self.recommender_source):
            yield path
        try:
            children = list(dataset_dir.iterdir()) if dataset_dir.exists() else []
        except Exception:
            children = []
        for child in children:
            if not child.is_dir():
                continue
            if child.name.startswith("user_"):
                continue
            for path in add(child):
                yield path

    def _legacy_paths(self, user_raw, initial=False):
        root = Path(self.initial_base_dir).parent if initial and self.initial_base_dir is not None else self.policy_root
        user_slug = self._user_slug(user_raw)
        for base in self._legacy_base_dirs(root):
            yield self._path_set_for_dir(base / user_slug)

    def _existing_paths(self, user_raw, initial=False):
        if initial and self.initial_base_dir is None:
            return None
        primary = self._initial_paths(user_raw) if initial else self._paths(user_raw)
        if primary is not None and primary["policy_json"].exists():
            return primary
        for paths in self._legacy_paths(user_raw, initial=initial):
            if paths["policy_json"].exists():
                return paths
        return primary

    def policy_exists(self, user_raw, include_legacy=True):
        paths = self._paths(user_raw)
        if paths["policy_json"].exists():
            return True
        if include_legacy:
            existing = self._existing_paths(user_raw)
            return existing is not None and existing["policy_json"].exists()
        return False

    def _initial_paths(self, user_raw):
        if self.initial_base_dir is None:
            return None
        return self._path_set_for_dir(self.initial_base_dir / self._user_slug(user_raw))

    @staticmethod
    def _rule(rule, confidence, status="active"):
        return {
            "rule": str(rule or ""),
            "confidence": float(confidence),
            "status": str(status or "active"),
            "reinforce_count": 0,
            "weaken_count": 0,
        }

    @staticmethod
    def canonical_skill_layer(layer_name):
        layer_name = str(layer_name or "").strip()
        if layer_name in ITEM_LAYER_ALIASES:
            return ITEM_LAYER_ALIASES[layer_name]
        if layer_name in COMMUNICATION_LAYER_ALIASES:
            return COMMUNICATION_LAYER_ALIASES[layer_name]
        if layer_name in POST_FEEDBACK_LAYER_ALIASES:
            return POST_FEEDBACK_LAYER_ALIASES[layer_name]
        if layer_name in ABSORPTION_LAYER_ALIASES:
            return ABSORPTION_LAYER_ALIASES[layer_name]
        if layer_name in ["item_selection_skill", "communication_selection_skill", "post_feedback_skill", "communication_absorption_skill"]:
            return layer_name
        return "item_selection_skill"

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
    def _numeric_version(value, default=1):
        try:
            return int(value)
        except Exception:
            return int(default)

    def _preference(self, attribute, confidence=0.50, source="stat_init", evidence_artists=None, evidence="", status="active"):
        return {
            "attribute": str(attribute or "").strip(),
            "confidence": float(confidence),
            "confidence_label": self._confidence_label(confidence),
            "source": str(source or "stat_init"),
            "evidence_artists": [str(x) for x in list(evidence_artists or []) if str(x or "").strip()][:10],
            "evidence": str(evidence or "").strip(),
            "status": str(status or "active"),
            "reinforce_count": 0,
            "weaken_count": 0,
        }

    def _rule_to_preference(self, row, source="legacy_rule"):
        row = dict(row or {})
        text = str(row.get("rule", "") or "").strip()
        if not text:
            return None
        if self._is_generic_item_protocol_rule(text):
            return None
        conf = float(row.get("confidence", 0.50) or 0.50)
        return {
            "attribute": text,
            "confidence": conf,
            "confidence_label": self._confidence_label(conf),
            "source": str(row.get("source", source) or source),
            "evidence_artists": [str(x) for x in list(row.get("evidence_artists", []) or []) if str(x or "").strip()][:10],
            "evidence": str(row.get("evidence", "") or row.get("problem", "") or "").strip(),
            "status": str(row.get("status", "active") or "active"),
            "reinforce_count": int(row.get("reinforce_count", 0) or 0),
            "weaken_count": int(row.get("weaken_count", 0) or 0),
        }

    def _preference_to_rule(self, row):
        row = dict(row or {})
        attribute = str(row.get("attribute", "") or row.get("rule", "") or "").strip()
        if not attribute:
            return None
        evidence = str(row.get("evidence", "") or "").strip()
        rule = attribute if evidence in ["", attribute] else f"{attribute}. Evidence: {evidence}"
        conf = float(row.get("confidence", 0.50) or 0.50)
        return {
            "rule": rule,
            "confidence": conf,
            "status": str(row.get("status", "active") or "active"),
            "reinforce_count": int(row.get("reinforce_count", 0) or 0),
            "weaken_count": int(row.get("weaken_count", 0) or 0),
        }

    @staticmethod
    def _canonical_who(node_id):
        node_id = str(node_id or "").strip()
        return str(WHO_NODE_ALIASES.get(node_id, node_id))

    def _normalize_who_mapping(self, mapping, default=0.50):
        mapping = dict(mapping or {})
        out = {
            "trusted-advisors": 0.90,
            "similar-users": 0.58,
            "experienced-users": 0.56,
            "topk-advisors": 0.52,
        }
        for key, value in mapping.items():
            canonical = self._canonical_who(key)
            if canonical not in out:
                continue
            try:
                value = float(value)
            except Exception:
                value = float(default)
            if canonical == "trusted-advisors":
                out[canonical] = max(0.90, value)
            else:
                out[canonical] = min(0.74, max(float(out.get(canonical, default)), value))
        return out

    @staticmethod
    def _canonical_how(node_id):
        node_id = str(node_id or "").strip()
        return str(HOW_NODE_ALIASES.get(node_id, node_id))

    @staticmethod
    def _is_generic_item_protocol_rule(text):
        low = " ".join(str(text or "").strip().lower().split())
        if not low:
            return False
        protocol_markers = [
            "scan all 20",
            "rank all 20",
            "candidate shortlist",
            "candidateshortlist",
            "hesitationshortlist",
            "top 5",
            "top-5",
            "top-k",
            "listwise ranking",
            "listwise rank",
            "choose one current favorite",
            "priorhint is only one evidence source",
            "before accepting priorhint",
            "avoid independent fit/drop pruning",
            "prioritize candidates",
            "prioritize artists",
            "rank candidates",
            "infer this user's music taste as multiple evidence-backed artist clusters",
            "infer this user's product preferences as multiple evidence-backed category/use-case clusters",
            "infer this user's book preferences as multiple evidence-backed genre/topic/author-style clusters",
            "preserve stable minority taste clusters",
            "preserve stable minority product categories/use-cases",
            "preserve stable minority book genres/topics",
            "use item-level music evidence as preference clues",
            "use item-level product evidence as preference clues",
            "use item-level book evidence as preference clues",
            "when recent interactions differ from long-term history",
            "history-cluster grounded selection",
            "product-history grounded selection",
            "book-history grounded selection",
            "reading-history grounded selection",
        ]
        return any(marker in low for marker in protocol_markers)

    @staticmethod
    def _is_rule_like_preference_attribute(text):
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
            "raise that candidate",
            "do not let this bias",
        ]
        return any(marker in low for marker in markers)

    @staticmethod
    def _is_weak_meta_preference_attribute(text):
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
        ]
        return any(marker in low for marker in weak_markers)

    @staticmethod
    def _preference_family_key(text):
        text = " ".join(str(text or "").strip().lower().split())
        if not text:
            return ""
        text = re.sub(
            r"^(reinforce transferable item-selection signal|add transferable item-selection signal|weaken over-bias)\s*:\s*",
            "",
            text,
        )
        text = re.split(r"\b(evidence|reason)\s*:", text, maxsplit=1)[0]
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text)
        stop = {
            "a", "an", "and", "are", "as", "by", "for", "from", "in", "is", "of", "or", "the", "to", "with",
            "user", "users", "candidate", "candidates", "item", "items", "artist", "artists", "music",
            "book", "books", "author", "authors", "reading",
            "preference", "preferences", "signal", "signals", "selection", "transferable", "positive",
            "future", "current", "same", 'why', "history", "evidence", "cluster", "clusters", "style", "styles",
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

    def _diverse_preference_rows(self, rows, limit=5):
        out = []
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            attr = str(row.get("attribute", "") or row.get("rule", "") or "").strip()
            if not attr:
                continue
            duplicate_idx = None
            for idx, existing in enumerate(out):
                if self._preferences_similar(attr, existing.get("attribute", "") or existing.get("rule", "")):
                    duplicate_idx = idx
                    break
            if duplicate_idx is None:
                out.append(row)
            else:
                existing = dict(out[duplicate_idx] or {})
                if float(row.get("confidence", 0.0) or 0.0) > float(existing.get("confidence", 0.0) or 0.0):
                    merged = dict(row)
                    merged["reinforce_count"] = max(int(row.get("reinforce_count", 0) or 0), int(existing.get("reinforce_count", 0) or 0))
                    merged["weaken_count"] = max(int(row.get("weaken_count", 0) or 0), int(existing.get("weaken_count", 0) or 0))
                    out[duplicate_idx] = merged
            if len(out) >= limit:
                break
        return out

    def _normalize_how_mapping(self, mapping, default=0.50):
        mapping = dict(mapping or {})
        out = {key: float(default) for key in ACTIVE_HOW_NODES}
        for key, value in mapping.items():
            canonical = self._canonical_how(key)
            if canonical not in out:
                continue
            try:
                value = float(value)
            except Exception:
                value = float(default)
            out[canonical] = max(float(out.get(canonical, default)), value)
        return out

    def _initial_core_rules(self, history_summary="", target_profile=""):
        if self._is_product_domain():
            profile = str(target_profile or history_summary or "").strip()
            anchor_text = profile[:420] if profile else "the user's long-term product interaction history"
            return [
                self._rule(
                    "Infer this user's product preferences as multiple evidence-backed category/use-case clusters rather than one dominant category; "
                    "treat each cluster as a possible positive route when judging candidate products/items. "
                    f"Initial anchors: {anchor_text}",
                    0.62,
                ),
                self._rule(
                    "Preserve stable minority product categories/use-cases: an item matching a smaller but repeated history pattern can be a strong user fit even when it does not match the most obvious category.",
                    0.60,
                ),
                self._rule(
                    "Use item-level product evidence as preference clues: category, use case, brand/manufacturer family, feature/function, price/value, quality/durability, reliability, design/form factor, compatibility/accessory relation, review/rating sentiment, and popularity/niche level.",
                    0.58,
                ),
                self._rule(
                    "Why recent interactions differ from long-term history, treat recent products as a short-term need signal but require at least one bridge back to the user's stable preference clusters.",
                    0.54,
                ),
            ]
        if self._is_book_domain():
            profile = str(target_profile or history_summary or "").strip()
            anchor_text = profile[:420] if profile else "the user's long-term book/reading history"
            return [
                self._rule(
                    "Infer this user's book preferences as multiple evidence-backed genre/topic/author-style clusters rather than one dominant genre; "
                    "treat each cluster as a possible positive route when judging candidate books. "
                    f"Initial anchors: {anchor_text}",
                    0.62,
                ),
                self._rule(
                    "Preserve stable minority book genres/topics/author-style clusters: a book matching a smaller but repeated reading pattern can be a strong user fit even when it does not match the most obvious category.",
                    0.60,
                ),
                self._rule(
                    "Use item-level book evidence as preference clues: fiction/non-fiction genre, literary form, topic/subject, author style, narrative tone, era/setting, cultural-language signal, audience/age category, series/franchise relation, canonical/award/niche level, and adjacent theme bridges.",
                    0.58,
                ),
                self._rule(
                    "Why recent interactions differ from long-term history, treat recent books as a short-term reading drift signal but require at least one bridge back to the user's stable preference clusters.",
                    0.54,
                ),
            ]
        profile = str(target_profile or history_summary or "").strip()
        anchor_text = profile[:420] if profile else "the user's long-term artist history"
        stable_rule = (
            "Infer this user's music taste as multiple evidence-backed artist clusters rather than one dominant genre; "
            "treat each cluster as a possible positive route when judging candidate artists. "
            f"Initial anchors: {anchor_text}"
        )
        if profile:
            stable_rule = (
                "Infer this user's music taste as multiple evidence-backed artist clusters rather than one dominant genre; "
                "treat each cluster as a possible positive route when judging candidate artists. "
                f"Initial anchors: {anchor_text}"
            )
        return [
            self._rule(stable_rule, 0.62),
            self._rule("Preserve stable minority taste clusters: an artist matching a smaller but repeated history pattern can be a strong user fit even when it does not match the most obvious history pattern.", 0.60),
            self._rule("Use item-level music evidence as preference clues: genre, scene, era, mood, vocal/instrumental style, energy, cultural/region signals, popularity level, and likely co-listening bridges to evidence artists.", 0.58),
            self._rule("Why recent interactions differ from long-term history, treat recent artists as a recency signal but require at least one bridge back to the user's stable taste clusters.", 0.54),
        ]

    def _initial_item_preferences(self, core_preference=None, core_rules=None, target_profile=""):
        core_preference = dict(core_preference or {})
        preferences = []
        for row in list(core_preference.get("preferences", []) or []):
            if isinstance(row, dict):
                pref = self._preference(
                    attribute=row.get("attribute", "") or row.get("cluster", ""),
                    confidence=row.get("confidence", 0.52),
                    source=row.get("source", "stat_init"),
                    evidence_artists=row.get("evidence_artists", []),
                    evidence=row.get("evidence", "") or row.get("ranking_rule", ""),
                    status=row.get("status", "active"),
                )
                if pref["attribute"]:
                    preferences.append(pref)
        for row in list(core_preference.get("taste_clusters", []) or []):
            if isinstance(row, dict):
                pref = self._preference(
                    attribute=row.get("cluster", ""),
                    confidence=row.get("confidence", 0.56),
                    source=row.get("source", "stat_init"),
                    evidence_artists=row.get("evidence_artists", []),
                    evidence=row.get("ranking_rule", ""),
                )
                if pref["attribute"]:
                    preferences.append(pref)
        if not preferences and str(target_profile or "").strip():
            preferences.append(self._preference(str(target_profile)[:260], confidence=0.50, source="fallback_profile"))

        recent_signals = []
        for row in list(core_preference.get("recent_signals", []) or []):
            if isinstance(row, dict):
                source = str(row.get("source", "") or "")
                if source in ("llm_label_supplement", "llm_label_supplement_recent"):
                    continue
                pref = self._preference(
                    attribute=row.get("attribute", "") or row.get("cluster", ""),
                    confidence=row.get("confidence", 0.46),
                    source=source or "stat_init_recent",
                    evidence_artists=row.get("evidence_artists", []),
                    evidence=row.get("evidence", ""),
                )
                if pref["attribute"]:
                    recent_signals.append(pref)

        def dedupe(rows):
            out = []
            seen = set()
            for pref in rows:
                key = " ".join(str(pref.get("attribute", "") or "").lower().split())
                if not key or key in seen:
                    continue
                seen.add(key)
                pref["confidence_label"] = self._confidence_label(pref.get("confidence", 0.50))
                out.append(pref)
            return out

        preferences = dedupe(preferences)[:10]
        active_rules = []
        for rule in list(core_rules or []):
            if isinstance(rule, dict) and str(rule.get("rule", "") or "").strip():
                active_rules.append(dict(rule))
        for pref in preferences[:8]:
            rule = self._preference_to_rule(pref)
            if rule:
                active_rules.append(rule)
        decision_style = str(core_preference.get("decision_style", "") or "").strip()
        if not decision_style:
            if self._is_product_domain():
                decision_style = "product-history grounded selection with minority-use-case preservation"
            elif self._is_book_domain():
                decision_style = "book-history grounded selection with minority-genre/topic preservation"
            else:
                decision_style = "history-cluster grounded selection with minority-cluster preservation"
        return {
            "preferences": preferences,
            "recent_signals": dedupe(recent_signals)[:5],
            "decision_style": decision_style,
            "active_rules": active_rules,
            "candidate_rules": [],
            "risky_rules": [],
            "inactive_rules": [],
        }

    def _initial_communication_rules(self, communication_evidence=None):
        evidence = dict(communication_evidence or {})
        selected_who = str(evidence.get("selected_who", "") or "neutral")
        selected_reason = str(evidence.get("selected_who_reason", "") or "").strip()
        evidence_bits = []
        for key in ["direct_trust_count", "two_hop_count", "similar_user_count", "history_count"]:
            if key in evidence:
                evidence_bits.append(f"{key}={evidence.get(key)}")
        selected_rule = (
            f"Initial who preference is {selected_who}; evidence: "
            f"{'; '.join(evidence_bits) if evidence_bits else 'limited bootstrap evidence'}"
            f"{'; ' + selected_reason if selected_reason else ''}."
        )
        return [
            self._rule("At bootstrap, choose trusted-advisors as the first who path by default. Trust is the primary communication source; if no trusted advisor pool is available at execution time, reroute explicitly rather than silently choosing another who node.", 0.82),
            self._rule("Inside trusted-advisors, reason over trust subbranches: mutual-trust is stronger than one-way-trust; multi-trust is stronger than single-trust; history-similar trusted users are stronger domain evidence than history-dissimilar trusted users.", 0.78),
            self._rule("Use similar-users only after trusted-advisors is unavailable or insufficient; use experienced-users when item-level experience is needed; use topk-advisors as friend-of-friend exploratory evidence, not as recommender topK evidence.", 0.66),
            self._rule("Choose communication in two layers: what is the user's natural-language task type, while how is only the advisor organization style: single-advisor, multi-cooperative, or multi-competitive.", 0.64),
            self._rule(selected_rule, 0.60),
        ]

    def _initial_communication_selection_skill(self, communication_evidence=None, active_rules=None):
        who_preference, communication_evidence = self._initial_who_preference(communication_evidence)
        active_rules = list(active_rules or self._initial_communication_rules(communication_evidence=communication_evidence))
        return {
            "who_preferences": [
                {"attribute": key, "confidence": float(value), "confidence_label": self._confidence_label(value)}
                for key, value in sorted(who_preference.items(), key=lambda kv: kv[1], reverse=True)
            ],
            "how_preferences": [
                {"attribute": key, "confidence": 0.50, "confidence_label": "medium"}
                for key in ACTIVE_HOW_NODES
            ],
            "trigger_rules": [
                {
                    "condition": "multiple candidates match different user preference clusters",
                    "mode": "multi-competitive",
                    "confidence": 0.50,
                },
                {
                    "condition": "the HesitationSet needs risk evidence from one advisor",
                    "mode": "single-advisor",
                    "confidence": 0.50,
                },
            ],
            "active_rules": active_rules,
            "candidate_rules": [],
            "risky_rules": [],
            "inactive_rules": [],
        }

    def _initial_communication_absorption_skill(self):
        return {
            "version": 1,
            "active_rules": [],
            "candidate_rules": [],
            "risky_rules": [],
            "inactive_rules": [],
            "ignored_advisor_signal_cases": [],
            "confidence": 0.50,
        }

    def _has_absorption_skill_content(self, absorption_skill):
        absorption_skill = dict(absorption_skill or {})
        for bucket in ["active_rules", "candidate_rules", "risky_rules", "inactive_rules", "ignored_advisor_signal_cases"]:
            if list(absorption_skill.get(bucket, []) or []):
                return True
        return False

    def _has_communication_selection_skill_content(self, comm_skill):
        comm_skill = dict(comm_skill or {})
        for bucket in [
            "active_rules",
            "candidate_rules",
            "risky_rules",
            "inactive_rules",
            "path_memory",
            "advisor_reliability_memory",
            "communication_round_memory",
        ]:
            if list(comm_skill.get(bucket, []) or []):
                return True
        return False

    @staticmethod
    def _history_items_from_summary(history_summary):
        text = str(history_summary or "").strip()
        if not text:
            return []
        parts = re.split(r"[,;\n]+", text)
        return [str(x).strip() for x in parts if str(x).strip()]

    def _communication_history_features(self, item_skill=None, history_summary=""):
        item_skill = dict(item_skill or {})
        preferences = [row for row in list(item_skill.get("preferences", []) or []) if isinstance(row, dict)]
        recent_signals = [row for row in list(item_skill.get("recent_signals", []) or []) if isinstance(row, dict)]
        active_rules = [row for row in list(item_skill.get("active_rules", []) or []) if isinstance(row, dict)]
        history_count = len(self._history_items_from_summary(history_summary))
        preference_text = " ".join(
            [
                str(item_skill.get("decision_style", "") or ""),
                " ".join(str(row.get("attribute", "") or row.get("rule", "") or "") for row in preferences),
                " ".join(str(row.get("attribute", "") or row.get("rule", "") or "") for row in recent_signals),
                " ".join(str(row.get("rule", "") or "") for row in active_rules),
            ]
        ).lower()
        if history_count <= 3 or len(preferences) <= 1:
            sparsity = "sparse"
        elif history_count <= 8 or len(preferences) <= 3:
            sparsity = "medium"
        else:
            sparsity = "rich"
        novelty_terms = [
            "novel", "novelty", "explore", "exploration", "minority", "niche", "unusual", "new",
            "小众", "探索", "新颖", "少数", "冷门",
        ]
        strong_rule_terms = ["choose_by", "preserve=", "minority=", "avoid_overbias", "must", "prefer"]
        return {
            "history_count": history_count,
            "history_sparsity": sparsity,
            "preference_cluster_count": len(preferences),
            "recent_signal_count": len(recent_signals),
            "multi_cluster": len(preferences) >= 4,
            "novelty_or_minority": any(term in preference_text for term in novelty_terms),
            "strong_item_rules": len(active_rules) >= 4 or any(term in preference_text for term in strong_rule_terms),
            "evidence_gap_prone": sparsity in ["sparse", "medium"] or len(recent_signals) == 0,
        }

    @staticmethod
    def _communication_advisor_features(communication_evidence=None):
        evidence = dict(communication_evidence or {})

        def as_int(key):
            try:
                return int(evidence.get(key, 0) or 0)
            except Exception:
                return 0

        def as_float(key):
            try:
                return float(evidence.get(key, 0.0) or 0.0)
            except Exception:
                return 0.0

        direct_trust = as_int("direct_trust_count")
        two_hop = as_int("two_hop_count")
        similar = as_int("similar_user_count")
        experienced = max(
            as_int("experienced_user_count"),
            as_int("experienced_count"),
            as_int("candidate_experienced_user_count"),
        )
        trusted_coverage = as_float("trusted_candidate_coverage")
        similar_coverage = as_float("similar_candidate_coverage")
        experienced_coverage = as_float("experienced_candidate_coverage")
        topk_coverage = as_float("topk_candidate_coverage")
        return {
            "direct_trust_count": direct_trust,
            "two_hop_count": two_hop,
            "similar_user_count": similar,
            "experienced_user_count": experienced,
            "candidate_count": as_int("candidate_count"),
            "best_similar_user_score": as_float("best_similar_user_score"),
            "avg_similar_user_score": as_float("avg_similar_user_score"),
            "trusted_candidate_user_count": as_int("trusted_candidate_user_count"),
            "trusted_candidate_item_count": as_int("trusted_candidate_item_count"),
            "trusted_candidate_coverage": trusted_coverage,
            "similar_candidate_user_count": as_int("similar_candidate_user_count"),
            "similar_candidate_item_count": as_int("similar_candidate_item_count"),
            "similar_candidate_coverage": similar_coverage,
            "candidate_experienced_user_count": as_int("candidate_experienced_user_count"),
            "experienced_candidate_item_count": as_int("experienced_candidate_item_count"),
            "experienced_candidate_coverage": experienced_coverage,
            "topk_candidate_user_count": as_int("topk_candidate_user_count"),
            "topk_candidate_item_count": as_int("topk_candidate_item_count"),
            "topk_candidate_coverage": topk_coverage,
            "advisor_candidate_diversity": as_int("advisor_candidate_diversity"),
            "trust_ready": direct_trust >= 1,
            "trust_pool_sufficient": direct_trust >= 2,
            "similar_ready": similar >= 3,
            "experienced_ready": experienced >= 1 or experienced_coverage > 0.0,
            "topk_ready": two_hop >= 1 or topk_coverage > 0.0,
        }

    @staticmethod
    def _bounded_confidence(value, low=0.45, high=0.62):
        try:
            value = float(value)
        except Exception:
            value = float(low)
        return round(max(float(low), min(float(high), value)), 2)

    @staticmethod
    def _ordered_unique(rows):
        out = []
        for row in list(rows or []):
            row = str(row or "").strip()
            if row and row not in out:
                out.append(row)
        return out

    @staticmethod
    def _rerank_with_prior(base_order, score_map):
        base = []
        for node in list(base_order or []):
            node = str(node or "").strip()
            if node and node not in base:
                base.append(node)
        rank = {node: idx for idx, node in enumerate(base)}
        return sorted(
            base,
            key=lambda node: (
                -float((score_map or {}).get(node, 0.0) or 0.0),
                rank.get(node, 999),
                node,
            ),
        )

    def _who_scores_by_how(self, base_who_scores, features=None, advisor_features=None):
        features = dict(features or {})
        advisor_features = dict(advisor_features or {})
        base_who_scores = {
            key: float(base_who_scores.get(key, 0.0) or 0.0)
            for key in ACTIVE_WHO_NODES
        }

        trust_cov = float(advisor_features.get("trusted_candidate_coverage", 0.0) or 0.0)
        similar_cov = float(advisor_features.get("similar_candidate_coverage", 0.0) or 0.0)
        experienced_cov = float(advisor_features.get("experienced_candidate_coverage", 0.0) or 0.0)
        topk_cov = float(advisor_features.get("topk_candidate_coverage", 0.0) or 0.0)
        best_sim = min(0.20, float(advisor_features.get("best_similar_user_score", 0.0) or 0.0))
        avg_sim = min(0.12, float(advisor_features.get("avg_similar_user_score", 0.0) or 0.0))
        diversity = int(advisor_features.get("advisor_candidate_diversity", 0) or 0)
        sparse_social = not advisor_features.get("trust_ready") and not advisor_features.get("similar_ready")

        scores_by_how = {}
        reasons_by_how = {}

        def init_scores():
            return {key: float(value) for key, value in base_who_scores.items()}

        def add(scores, reasons, who, amount, reason):
            if who not in scores:
                return
            amount = float(amount or 0.0)
            if amount == 0.0:
                return
            scores[who] = float(scores.get(who, 0.0) or 0.0) + amount
            reasons.setdefault(who, []).append(f"+{amount:.2f} {reason}")

        single_scores = init_scores()
        single_reasons = {key: [f"base_prior={base_who_scores.get(key, 0.0):.2f}"] for key in ACTIVE_WHO_NODES}
        add(single_scores, single_reasons, "trusted-advisors", 0.06, "single-advisor favors reliable direct trust")
        add(single_scores, single_reasons, "trusted-advisors", 0.10 * trust_cov, "trusted advisors cover current candidates")
        if advisor_features.get("trust_pool_sufficient"):
            add(single_scores, single_reasons, "trusted-advisors", 0.03, "multiple trusted advisors available")
        add(single_scores, single_reasons, "experienced-users", 0.05, "single-advisor can use a strong experienced source")
        add(single_scores, single_reasons, "experienced-users", 0.14 * experienced_cov, "experienced users cover current candidates")
        if features.get("evidence_gap_prone"):
            add(single_scores, single_reasons, "experienced-users", 0.03, "history evidence gap benefits from experience")
        add(single_scores, single_reasons, "similar-users", 0.07 * similar_cov + 0.20 * best_sim, "similar users match history and candidates")
        add(single_scores, single_reasons, "topk-advisors", 0.08 * topk_cov, "two-hop advisors cover current candidates")
        if sparse_social:
            add(single_scores, single_reasons, "topk-advisors", 0.04, "weak direct/similar pool fallback")

        cooperative_scores = init_scores()
        cooperative_reasons = {key: [f"base_prior={base_who_scores.get(key, 0.0):.2f}"] for key in ACTIVE_WHO_NODES}
        add(cooperative_scores, cooperative_reasons, "similar-users", 0.07, "multi-cooperative favors preference-neighbor coverage")
        add(cooperative_scores, cooperative_reasons, "similar-users", 0.16 * similar_cov + 0.18 * avg_sim, "similar users add complementary candidate evidence")
        if features.get("multi_cluster"):
            add(cooperative_scores, cooperative_reasons, "similar-users", 0.03, "multi-cluster preference needs multiple similar views")
        add(cooperative_scores, cooperative_reasons, "experienced-users", 0.12 * experienced_cov, "experienced users broaden candidate coverage")
        if features.get("evidence_gap_prone"):
            add(cooperative_scores, cooperative_reasons, "experienced-users", 0.03, "cooperative check fills evidence gaps")
        add(cooperative_scores, cooperative_reasons, "trusted-advisors", 0.08 * trust_cov, "trusted advisors contribute stable evidence")
        if advisor_features.get("trust_ready"):
            add(cooperative_scores, cooperative_reasons, "trusted-advisors", 0.03, "direct trusted pool available")
        add(cooperative_scores, cooperative_reasons, "topk-advisors", 0.10 * topk_cov, "two-hop pool adds extra coverage")
        if diversity <= 1 and advisor_features.get("topk_ready"):
            add(cooperative_scores, cooperative_reasons, "topk-advisors", 0.03, "low advisor diversity needs fallback expansion")

        competitive_scores = init_scores()
        competitive_reasons = {key: [f"base_prior={base_who_scores.get(key, 0.0):.2f}"] for key in ACTIVE_WHO_NODES}
        add(competitive_scores, competitive_reasons, "experienced-users", 0.08, "multi-competitive favors candidate-experience checks")
        add(competitive_scores, competitive_reasons, "experienced-users", 0.18 * experienced_cov, "experienced users can challenge candidate choices")
        if features.get("strong_item_rules") or features.get("evidence_gap_prone"):
            add(competitive_scores, competitive_reasons, "experienced-users", 0.04, "conflict/evidence-gap case needs experience-based verification")
        add(competitive_scores, competitive_reasons, "trusted-advisors", 0.08 * trust_cov, "trusted advisors provide a stable counterpoint")
        if advisor_features.get("trust_ready"):
            add(competitive_scores, competitive_reasons, "trusted-advisors", 0.04, "trusted source available for debate")
        add(competitive_scores, competitive_reasons, "similar-users", 0.10 * similar_cov + 0.10 * best_sim, "similar users provide preference-side contrast")
        if features.get("multi_cluster"):
            add(competitive_scores, competitive_reasons, "similar-users", 0.03, "multi-cluster preference benefits from contrast")
        add(competitive_scores, competitive_reasons, "topk-advisors", 0.08 * topk_cov, "two-hop advisors can introduce alternative evidence")
        if sparse_social:
            add(competitive_scores, competitive_reasons, "topk-advisors", 0.03, "sparse social graph fallback")

        scores_by_how["single-advisor"] = single_scores
        scores_by_how["multi-cooperative"] = cooperative_scores
        scores_by_how["multi-competitive"] = competitive_scores
        scores_by_how["default"] = {
            key: round(
                (
                    single_scores.get(key, 0.0)
                    + cooperative_scores.get(key, 0.0)
                    + competitive_scores.get(key, 0.0)
                ) / 3.0,
                4,
            )
            for key in ACTIVE_WHO_NODES
        }
        reasons_by_how["single-advisor"] = single_reasons
        reasons_by_how["multi-cooperative"] = cooperative_reasons
        reasons_by_how["multi-competitive"] = competitive_reasons
        reasons_by_how["default"] = {
            key: ["average of how-specific communication route scores"]
            for key in ACTIVE_WHO_NODES
        }
        return scores_by_how, reasons_by_how

    def _communication_template_id(self, features=None, advisor_features=None):
        features = dict(features or {})
        advisor_features = dict(advisor_features or {})
        if features.get("history_sparsity") == "sparse" and not advisor_features.get("trust_ready") and not advisor_features.get("similar_ready"):
            return "sparse-weak-social"
        if advisor_features.get("experienced_ready") and (features.get("evidence_gap_prone") or features.get("history_sparsity") != "rich"):
            return "experience-rich"
        if advisor_features.get("trust_ready") and int(advisor_features.get("direct_trust_count", 0) or 0) >= 2:
            return "trust-rich"
        if features.get("novelty_or_minority"):
            return "novelty-seeking"
        if features.get("strong_item_rules"):
            return "prior-conflict-prone"
        if features.get("history_sparsity") == "rich" and advisor_features.get("similar_ready"):
            return "rich-similar-social"
        return "rich-similar-social" if features.get("history_sparsity") == "rich" else "experience-rich"

    def _initial_communication_route_skill(self, communication_evidence=None, item_skill=None, history_summary=""):
        features = self._communication_history_features(item_skill=item_skill, history_summary=history_summary)
        advisor_features = self._communication_advisor_features(communication_evidence)
        template_id = self._communication_template_id(features, advisor_features)
        who_scores = {
            "trusted-advisors": 0.44 + min(0.12, 0.04 * advisor_features["direct_trust_count"]),
            "similar-users": 0.42
            + (0.07 if features["history_sparsity"] == "rich" else 0.03)
            + min(0.08, 0.01 * advisor_features["similar_user_count"]),
            "experienced-users": 0.45
            + (0.08 if advisor_features["experienced_ready"] else 0.03)
            + (0.04 if features["evidence_gap_prone"] else 0.0),
            "topk-advisors": 0.40
            + (0.06 if advisor_features["topk_ready"] else 0.0)
            + (0.05 if not advisor_features["trust_ready"] else 0.0),
        }
        if features["novelty_or_minority"]:
            who_scores["experienced-users"] += 0.03
            who_scores["topk-advisors"] += 0.03
        if template_id == "trust-rich":
            who_scores["trusted-advisors"] += 0.05
        if template_id == "rich-similar-social":
            who_scores["similar-users"] += 0.05
        if template_id == "experience-rich":
            who_scores["experienced-users"] += 0.05
        if template_id == "sparse-weak-social":
            who_scores["topk-advisors"] += 0.04
            who_scores["experienced-users"] += 0.04
        who_scores = {key: round(float(value), 4) for key, value in who_scores.items()}
        preferred_who = [key for key, _ in sorted(who_scores.items(), key=lambda kv: (-kv[1], kv[0]))]

        what_by_why = {
            "internal-prior-conflict+candidate-conflict": [
                "reasoning_check",
                "compare_remaining_candidates",
                "reduce_hesitation_set",
                "none",
            ],
            "novelty-uncertainty+candidate-conflict": [
                "find_interested_subset",
                "reduce_hesitation_set",
                "compare_remaining_candidates",
                "none",
            ],
            "cold-start+candidate-conflict": [
                "evidence_gap_check",
                "reduce_hesitation_set",
                "compare_remaining_candidates",
                "none",
            ],
            "candidate-conflict": [
                "reduce_hesitation_set",
                "compare_remaining_candidates",
                "reasoning_check",
                "none",
            ],
            "internal-prior-conflict": [
                "reasoning_check",
                "compare_remaining_candidates",
                "none",
            ],
            "novelty-uncertainty": [
                "find_interested_subset",
                "reduce_hesitation_set",
                "none",
            ],
            "cold-start": [
                "evidence_gap_check",
                "reduce_hesitation_set",
                "none",
            ],
            "default": [
                "reduce_hesitation_set",
                "compare_remaining_candidates",
                "none",
            ],
        }
        if template_id == "experience-rich":
            what_by_why["candidate-conflict"] = ["evidence_gap_check", "reduce_hesitation_set", "compare_remaining_candidates", "none"]
        if template_id == "novelty-seeking":
            what_by_why["candidate-conflict"] = ["find_interested_subset", "reduce_hesitation_set", "compare_remaining_candidates", "none"]
        if template_id == "prior-conflict-prone":
            what_by_why["candidate-conflict"] = ["reasoning_check", "compare_remaining_candidates", "reduce_hesitation_set", "none"]

        how_by_what = {}
        for what in ACTIVE_WHAT_NODES:
            if what in ["reasoning_check", "compare_remaining_candidates"]:
                how_by_what[what] = ["multi-competitive", "multi-cooperative", "single-advisor"]
            elif what in ["reduce_hesitation_set", "find_interested_subset", "evidence_gap_check"]:
                how_by_what[what] = ["multi-cooperative", "multi-competitive", "single-advisor"]
            else:
                how_by_what[what] = ["multi-cooperative", "single-advisor", "multi-competitive"]
        how_by_what["default"] = ["multi-cooperative", "multi-competitive", "single-advisor"]
        if template_id in ["sparse-weak-social", "experience-rich"]:
            how_by_what["evidence_gap_check"] = ["multi-cooperative", "single-advisor", "multi-competitive"]
        if template_id == "prior-conflict-prone":
            how_by_what["reasoning_check"] = [
                "multi-competitive",
                "multi-cooperative",
                "single-advisor",
            ]

        who_scores_by_how, who_reasons_by_how = self._who_scores_by_how(
            who_scores,
            features=features,
            advisor_features=advisor_features,
        )
        who_by_how = {
            how: self._rerank_with_prior(preferred_who, who_scores_by_how.get(how, who_scores))
            for how in ACTIVE_HOW_NODES
        }
        who_by_how["default"] = self._rerank_with_prior(
            preferred_who,
            who_scores_by_how.get("default", who_scores),
        )
        normalized_score_payload = {
            how: {
                who: round(float(score), 4)
                for who, score in dict(scores or {}).items()
                if who in ACTIVE_WHO_NODES
            }
            for how, scores in dict(who_scores_by_how or {}).items()
        }
        normalized_reason_payload = {
            how: {
                who: list(reasons or [])[:4]
                for who, reasons in dict(reason_rows or {}).items()
                if who in ACTIVE_WHO_NODES
            }
            for how, reason_rows in dict(who_reasons_by_how or {}).items()
        }

        return {
            "version": 2,
            "template_id": template_id,
            "template_features": {
                "history_richness": features.get("history_sparsity", ""),
                "preference_structure": (
                    "novelty-seeking" if features.get("novelty_or_minority")
                    else ("multi-cluster" if features.get("multi_cluster") else "focused")
                ),
                "social_access": (
                    "trust-rich" if advisor_features.get("trust_ready")
                    else ("similar-rich" if advisor_features.get("similar_ready")
                          else ("experience-rich" if advisor_features.get("experienced_ready") else "weak-social"))
                ),
                "decision_risk": (
                    "prior-conflict-prone" if features.get("strong_item_rules")
                    else ("evidence-gap-prone" if features.get("evidence_gap_prone") else "normal")
                ),
            },
            "signature_order": list(DEFAULT_TRIGGER_SIGNATURES),
            "what_by_why": {
                key: [x for x in self._ordered_unique(value) if x in ACTIVE_WHAT_NODES]
                for key, value in what_by_why.items()
            },
            "how_by_what": {
                key: [x for x in self._ordered_unique(value) if x in ACTIVE_HOW_NODES]
                for key, value in how_by_what.items()
            },
            "who_by_how": {
                key: [x for x in self._ordered_unique(value) if x in ACTIVE_WHO_NODES]
                for key, value in who_by_how.items()
            },
            "initial_route_scores": {
                "global_who_prior": dict(who_scores),
                "who_by_how": normalized_score_payload,
            },
            "initial_route_reasons": {
                "who_by_how": normalized_reason_payload,
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

    def _initial_post_feedback_rules(self, communication_evidence=None, core_preference=None):
        evidence = dict(communication_evidence or {})
        core_preference = dict(core_preference or {})
        direct_trust_count = int(evidence.get("direct_trust_count", 0) or 0)
        stable_summary = ", ".join([str(x) for x in list(core_preference.get("long_term_preference", []) or [])[:3] if str(x).strip()])
        trust_clause = (
            "trusted and history-similar advisors"
            if direct_trust_count > 0
            else "advisor evidence with clear item-level support"
        )
        cluster_clause = stable_summary if stable_summary else (
            "the user's long-term product preference clusters"
            if self._is_product_domain()
            else (
                "the user's long-term book preference clusters"
                if self._is_book_domain()
                else "the user's long-term taste clusters"
            )
        )
        return [
            self._rule(
                f"After advisor feedback, switch only when {trust_clause} provide concrete item-level evidence that another candidate matches {cluster_clause} better than the current favorite.",
                0.66,
            ),
            self._rule("Do not change choices because of vote count, generic praise, or repeated claims; advisor evidence must explain the candidate-to-history bridge.", 0.64),
            self._rule("If feedback reveals that a candidate bridges multiple user preference clusters or a stable minority cluster that the initial choice underweighted, raise that candidate into the hesitation set or switch only with concrete item-level proof.", 0.62),
            self._rule("If advisor evidence remains tied or misses important shortlisted candidates, keep the history-grounded favorite or continue only with a specific missing comparison request.", 0.60),
        ]

    def _initial_post_feedback_skill(self, communication_evidence=None, core_preference=None, active_rules=None):
        active_rules = list(active_rules or self._initial_post_feedback_rules(
            communication_evidence=communication_evidence,
            core_preference=core_preference,
        ))
        return {
            "post_feedback_trust_rules": active_rules,
            "continue_rules": [
                self._rule(
                    "If important focus candidates are silent or the comparison is missing, continue only with a specific feedback-focused repair request.",
                    0.60,
                )
            ],
            "harm_avoidance_rules": [
                self._rule(
                    "If the initial proposal already matches the user skill and advisor evidence is generic or vote-count-only, do not switch away from it.",
                    0.58,
                )
            ],
            "missing_evidence_rules": [
                self._rule(
                    "Silent focus candidates are missing evidence, not negative evidence; ask for direct comparison before finalizing when they remain plausible.",
                    0.62,
                )
            ],
            "active_rules": active_rules,
            "candidate_rules": [],
            "risky_rules": [],
            "inactive_rules": [],
        }

    def _initial_who_preference(self, communication_evidence=None):
        evidence = dict(communication_evidence or {})
        direct_trust_count = int(evidence.get("direct_trust_count", 0) or 0)
        two_hop_count = int(evidence.get("two_hop_count", 0) or 0)
        similar_user_count = int(evidence.get("similar_user_count", 0) or 0)
        history_count = int(evidence.get("history_count", 0) or 0)
        exploratory_signal = bool(evidence.get("exploratory_signal", False))

        selected_who = "trusted-advisors"
        if direct_trust_count > 0:
            selected_reason = "direct trust users are available, so trust is both preferred and executable"
        else:
            selected_reason = (
                "trust is the bootstrap default by design; if direct trusted advisors are unavailable, "
                "the executor must record trust-pool empty and reroute explicitly"
            )
        pref = {
            "trusted-advisors": 0.90,
            "similar-users": 0.58 if history_count >= 20 and similar_user_count > 0 else 0.52,
            "experienced-users": 0.56 if history_count < 20 else 0.52,
            "topk-advisors": 0.54 if exploratory_signal and two_hop_count > 0 else 0.50,
        }

        evidence["selected_who"] = selected_who
        evidence["selected_who_reason"] = selected_reason
        return pref, evidence

    def _default_policy(
        self,
        user_raw,
        history_summary="",
        target_profile="",
        communication_evidence=None,
        core_rules=None,
        core_preference=None,
        core_initial_evidence=None,
    ):
        who_preference, communication_evidence = self._initial_who_preference(communication_evidence)
        core_pref_override = dict(core_preference or {})
        long_term_preference = core_pref_override.get("long_term_preference")
        if not isinstance(long_term_preference, list) or not long_term_preference:
            long_term_preference = [str(target_profile)] if str(target_profile or "").strip() else []
        active_core_rules = list(core_rules or [])
        if not active_core_rules:
            active_core_rules = self._initial_core_rules(history_summary=history_summary, target_profile=target_profile)
        item_skill = self._initial_item_preferences(
            core_preference={
                **core_pref_override,
                "long_term_preference": [str(x) for x in long_term_preference if str(x or "").strip()],
            },
            core_rules=active_core_rules,
            target_profile=target_profile,
        )
        return {
            "user_id": str(user_raw),
            "dataset": self.dataset,
            "version": 1,
            "item_selection_skill": item_skill,
            "communication_route_skill": self._initial_communication_route_skill(
                communication_evidence=communication_evidence,
                item_skill=item_skill,
                history_summary=history_summary,
            ),
        }

    def _normalize_two_skill_schema(self, policy):
        policy = dict(policy or {})
        core_pref = dict(policy.get("core_preference", {}) or {})
        old_core = dict(policy.get("core_decision_reasoning_skill", {}) or {})
        old_comm = dict(policy.get("communication_reasoning_skill", {}) or {})
        old_post = dict(policy.get("post_feedback_reasoning_skill", {}) or {})
        old_comm_pref = dict(policy.get("communication_preference", {}) or {})

        item_skill = dict(policy.get("item_selection_skill", {}) or {})
        legacy_core_rules = []
        for bucket in ["active_rules", "candidate_rules", "risky_rules", "inactive_rules"]:
            legacy_core_rules.extend(list(old_core.get(bucket, []) or []))
        if not item_skill:
            item_skill = self._initial_item_preferences(
                core_preference=core_pref,
                core_rules=legacy_core_rules,
                target_profile=", ".join(core_pref.get("long_term_preference", []) or []),
            )
        for bucket in ["active_rules", "candidate_rules", "risky_rules", "inactive_rules"]:
            rows = []
            for row in list(item_skill.get(bucket, []) or []):
                if isinstance(row, dict) and str(row.get("rule", "")).strip():
                    if not self._is_generic_item_protocol_rule(row.get("rule", "")):
                        rows.append(dict(row))
            item_skill[bucket] = rows
        preferences = []
        for row in list(item_skill.get("preferences", []) or []):
            if isinstance(row, dict):
                pref = self._preference(
                    row.get("attribute", "") or row.get("rule", ""),
                    confidence=row.get("confidence", 0.50),
                    source=row.get("source", "policy"),
                    evidence_artists=row.get("evidence_artists", []),
                    evidence=row.get("evidence", ""),
                    status=row.get("status", "active"),
                )
                pref["reinforce_count"] = int(row.get("reinforce_count", 0) or 0)
                pref["weaken_count"] = int(row.get("weaken_count", 0) or 0)
                if pref["attribute"]:
                    preferences.append(pref)
        seen = set()
        deduped_preferences = []
        weak_meta_count = 0
        for pref in preferences:
            key = " ".join(str(pref.get("attribute", "") or "").lower().split())
            if not key or key in seen:
                continue
            source = str(pref.get("source", "") or "")
            if "rule_migration" in source:
                continue
            if self._is_rule_like_preference_attribute(key):
                continue
            if self._is_generic_item_protocol_rule(key):
                continue
            if self._is_weak_meta_preference_attribute(key):
                weak_meta_count += 1
                if weak_meta_count > 2:
                    continue
            seen.add(key)
            pref["confidence_label"] = self._confidence_label(pref.get("confidence", 0.50))
            deduped_preferences.append(pref)
        item_skill["preferences"] = deduped_preferences[:12]
        item_skill["recent_signals"] = list(item_skill.get("recent_signals", []) or [])[:6]
        item_skill.setdefault("decision_style", "history-cluster grounded selection with minority-cluster preservation")
        if not item_skill.get("active_rules"):
            item_skill["active_rules"] = [
                rule for rule in (self._preference_to_rule(pref) for pref in item_skill["preferences"][:8]) if rule
            ]
        for stale_key in ["path_memory", "advisor_reliability_memory", "communication_round_memory"]:
            item_skill.pop(stale_key, None)

        explicit_comm_skill = dict(policy.get("communication_selection_skill", {}) or {})
        preserve_comm_skill = self._has_communication_selection_skill_content(explicit_comm_skill)
        comm_skill = dict(explicit_comm_skill)
        planning_skill = dict(policy.get("communication_planning_skill", {}) or {})
        route_skill = dict(policy.get("communication_route_skill", {}) or {})
        legacy_comm_rules = []
        for bucket in ["active_rules", "candidate_rules", "risky_rules", "inactive_rules"]:
            legacy_comm_rules.extend(list(old_comm.get(bucket, []) or []))
        if not comm_skill:
            comm_skill = self._initial_communication_selection_skill(
                communication_evidence=policy.get("communication_initial_evidence", {}),
                active_rules=legacy_comm_rules or self._initial_communication_rules(policy.get("communication_initial_evidence", {})),
            )
        trigger_pref = dict(old_comm_pref.get("trigger_preference", {}) or {})
        who_pref = self._normalize_who_mapping(old_comm_pref.get("who_preference", {}))
        how_pref = self._normalize_how_mapping(old_comm_pref.get("how_preference", {}))
        if not comm_skill.get("who_preferences"):
            comm_skill["who_preferences"] = [
                {"attribute": key, "confidence": float(value), "confidence_label": self._confidence_label(value)}
                for key, value in sorted(who_pref.items(), key=lambda kv: kv[1], reverse=True)
            ]
        if not comm_skill.get("how_preferences"):
            comm_skill["how_preferences"] = [
                {"attribute": key, "confidence": float(value), "confidence_label": self._confidence_label(value)}
                for key, value in sorted(how_pref.items(), key=lambda kv: kv[1], reverse=True)
            ]
        if not comm_skill.get("trigger_rules"):
            comm_skill["trigger_rules"] = [
                {"condition": key, "mode": "", "confidence": float(value)}
                for key, value in sorted(trigger_pref.items(), key=lambda kv: kv[1], reverse=True)
            ]
        merged_comm_active = list(comm_skill.get("active_rules", []) or []) + legacy_comm_rules
        comm_skill["active_rules"] = self._dedupe_rule_rows(merged_comm_active)[:10]
        for bucket in ["candidate_rules", "risky_rules", "inactive_rules"]:
            comm_skill.setdefault(bucket, [])
        for memory_key in ["path_memory", "advisor_reliability_memory", "communication_round_memory"]:
            comm_skill.setdefault(memory_key, [])

        if not route_skill:
            route_skill = self._initial_communication_route_skill(
                communication_evidence=policy.get("communication_initial_evidence", {}),
                item_skill=item_skill,
            )
        route_skill = self._normalize_communication_route_skill(route_skill, planning_skill)

        post_skill = dict(policy.get("post_feedback_skill", {}) or {})
        legacy_post_rules = []
        for bucket in ["active_rules", "candidate_rules", "risky_rules", "inactive_rules", "post_feedback_trust_rules"]:
            legacy_post_rules.extend(list(old_post.get(bucket, []) or []))
        if not post_skill:
            post_skill = self._initial_post_feedback_skill(
                communication_evidence=policy.get("communication_initial_evidence", {}),
                core_preference=core_pref,
                active_rules=legacy_post_rules or self._initial_post_feedback_rules(
                    communication_evidence=policy.get("communication_initial_evidence", {}),
                    core_preference=core_pref,
                ),
            )
        active_post = self._dedupe_rule_rows(
            list(post_skill.get("active_rules", []) or [])
            + list(post_skill.get("post_feedback_trust_rules", []) or [])
            + legacy_post_rules
        )[:10]
        post_skill["active_rules"] = active_post
        post_skill["post_feedback_trust_rules"] = self._dedupe_rule_rows(
            list(post_skill.get("post_feedback_trust_rules", []) or []) + active_post
        )[:10]
        for bucket in ["continue_rules", "harm_avoidance_rules", "missing_evidence_rules", "candidate_rules", "risky_rules", "inactive_rules"]:
            post_skill.setdefault(bucket, [])

        absorption_skill = dict(policy.get("communication_absorption_skill", {}) or {})
        legacy_absorption = dict(policy.get("feedback_absorption_policy", {}) or {})
        if not absorption_skill and legacy_absorption:
            absorption_skill = self._initial_communication_absorption_skill()
        if absorption_skill or legacy_absorption:
            absorption_skill["version"] = self._numeric_version(absorption_skill.get("version", 1), default=1)
            inherited_absorption_rules = []
            for bucket in ["active_rules", "candidate_rules", "risky_rules", "inactive_rules"]:
                inherited_absorption_rules.extend(list(legacy_absorption.get(bucket, []) or []))
            absorption_skill["active_rules"] = self._dedupe_rule_rows(
                list(absorption_skill.get("active_rules", []) or []) + inherited_absorption_rules
            )[:3]
            for bucket in ["candidate_rules", "risky_rules", "inactive_rules"]:
                absorption_skill.setdefault(bucket, [])
            absorption_skill["ignored_advisor_signal_cases"] = [
                dict(row)
                for row in list(absorption_skill.get("ignored_advisor_signal_cases", []) or [])
                if isinstance(row, dict)
            ][-6:]
            try:
                absorption_skill["confidence"] = float(absorption_skill.get("confidence", 0.50) or 0.50)
            except Exception:
                absorption_skill["confidence"] = 0.50

        policy["item_selection_skill"] = item_skill
        if preserve_comm_skill and self._has_communication_selection_skill_content(comm_skill):
            policy["communication_selection_skill"] = comm_skill
        else:
            policy.pop("communication_selection_skill", None)
        policy["communication_route_skill"] = route_skill
        if self._has_absorption_skill_content(absorption_skill):
            policy["communication_absorption_skill"] = absorption_skill
        else:
            policy.pop("communication_absorption_skill", None)
        for old_key in [
            "core_decision_reasoning_skill",
            "communication_reasoning_skill",
            "post_feedback_reasoning_skill",
            "core_preference",
            "communication_preference",
            "communication_planning_skill",
            "post_feedback_skill",
            "confidence_calibration",
            "advisor_reliability_memory",
            "policy_evolution_state",
            "core_initial_evidence",
            "evolution_memory",
            "feedback_absorption_policy",
            "decision_patterns",
        ]:
            policy.pop(old_key, None)
        policy["version"] = self._numeric_version(policy.get("version", 1), default=1)
        return self._minimal_policy_schema(policy)

    def _normalize_communication_route_skill(self, route_skill, planning_skill=None):
        route = dict(route_skill or {})
        planning_skill = dict(planning_skill or {})
        if not route:
            route = self._initial_communication_route_skill(item_skill={})
        route["version"] = max(2, self._numeric_version(route.get("version", 1), default=1))
        route.setdefault("template_id", "migrated-route")
        route.setdefault("template_features", {})
        route["signature_order"] = self._ordered_unique(route.get("signature_order", []) or DEFAULT_TRIGGER_SIGNATURES)
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
        if planning_skill:
            for row in list(planning_skill.get("how_policy", []) or []):
                if not isinstance(row, dict):
                    continue
                for avoided in list(row.get("avoid_how", []) or []):
                    avoided = str(avoided or "").strip()
                    if not avoided:
                        continue
                    scope = self._scope_from_condition(row.get("condition", {}))
                    route["demotions"] = self._append_unique_rows(
                        route.get("demotions", []),
                        [{
                            "level": "how",
                            "scope": scope,
                            "node": avoided,
                            "reason": str(row.get("rationale", "") or "migrated avoid_how"),
                            "source": "planning_skill_migration",
                            "count": 1,
                        }],
                        key_fields=["level", "scope", "node"],
                        limit=32,
                    )
            for row in list(planning_skill.get("tree_need_signals", []) or []) + list(planning_skill.get("open_condition_memory", []) or []):
                if not isinstance(row, dict):
                    continue
                if str(row.get("level", "") or "").strip() not in ["what", ""]:
                    continue
                hint = str(row.get("suggested_node_hint", "") or "").strip()
                if not hint:
                    continue
                route["unmapped_task_memory"] = self._append_unique_rows(
                    route.get("unmapped_task_memory", []),
                    [{
                        "task_text_summary": str(row.get("evidence_pattern", "") or row.get("task_intent", "") or hint)[:240],
                        "mapped_what": "none",
                        "suggested_future_what": hint,
                        "count": 1,
                        "source": "planning_skill_migration",
                    }],
                    key_fields=["mapped_what", "suggested_future_what", "task_text_summary"],
                    limit=24,
                )
        route = self._sync_public_tree_active_children_into_route_skill(route)
        return route

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

    @staticmethod
    def _scope_from_condition(condition, preferred=""):
        condition = dict(condition or {}) if isinstance(condition, dict) else {}
        uncertainty = condition.get("uncertainty_shape", "")
        if isinstance(uncertainty, list):
            whens = [str(x) for x in uncertainty if str(x).strip()]
        else:
            whens = [str(uncertainty)] if str(uncertainty or "").strip() else []
        if str(condition.get('why', "") or "").strip():
            whens.append(str(condition.get('why')))
        if str(condition.get("prior_relation", "") or "") == "proposal_differs_from_prior":
            whens.append("internal-prior-conflict")
        if str(condition.get("focus_set_size_min", "") or ""):
            whens.append("candidate-conflict")
        priority = {name: idx for idx, name in enumerate(TRIGGER_SIGNATURE_PRIORITY)}
        whens = sorted({w for w in whens if w and w != "proposal-risk-check"}, key=lambda x: priority.get(x, 99))
        signature = "+".join(whens) if whens else "default"
        preferred = str(preferred or "").strip()
        return f"{signature}|{preferred}" if preferred else signature

    @staticmethod
    def _parent_node_id(node_id):
        node_id = str(node_id or "").strip()
        return "/".join(node_id.split("/")[:-1]) if "/" in node_id else ""

    @staticmethod
    def _node_depth(node_id):
        return len([part for part in str(node_id or "").strip("/").split("/") if part])

    @staticmethod
    def _scope_matches(row_scope, scope):
        row_scope = str(row_scope or "").strip()
        scope = str(scope or "").strip()
        if not row_scope or row_scope == "default":
            return True
        return row_scope == scope or scope.startswith(row_scope) or row_scope.startswith(scope)

    def _route_node_trial_stats(self, route, level, scope, node):
        stats = {"trial_count": 0, "helpful_count": 0, "harmful_count": 0, "ineffective_count": 0}
        for row in list((route or {}).get("exploration_history", []) or []) + list((route or {}).get("exploration_slots", []) or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("level", "") or "") != str(level):
                continue
            if str(row.get("node", "") or "") != str(node):
                continue
            if not self._scope_matches(row.get("scope", ""), scope):
                continue
            for field in list(stats.keys()):
                try:
                    stats[field] = max(int(stats.get(field, 0) or 0), int(row.get(field, 0) or 0))
                except Exception:
                    pass
        return stats

    @staticmethod
    def _place_child_route_order(order, child, parent, stats=None):
        order = [str(x) for x in list(order or []) if str(x).strip() and str(x) != str(child)]
        child = str(child or "").strip()
        parent = str(parent or "").strip()
        if not child:
            return order
        stats = dict(stats or {})
        helpful = int(stats.get("helpful_count", 0) or 0)
        harmful = int(stats.get("harmful_count", 0) or 0)
        ineffective = int(stats.get("ineffective_count", 0) or 0)
        if parent and parent in order:
            idx = order.index(parent) + 1
            return order[:idx] + [child] + order[idx:]
        if helpful > 0 and harmful == 0:
            return [child] + order
        if harmful > 0 or ineffective > 0:
            return order + [child]
        return order + [child]

    def _sync_public_tree_active_children_into_route_skill(self, route):
        route = dict(route or {})
        # Formal COM path selection is driven by each user's persisted
        # communication_route_skill. Public tree children should enter a user
        # route only through explicit tree-route injection, not automatic sync.
        return route

    def _minimal_policy_schema(self, policy):
        policy = dict(policy or {})
        item_skill = dict(policy.get("item_selection_skill", {}) or {})
        comm_skill = dict(policy.get("communication_selection_skill", {}) or {})
        route_skill = dict(policy.get("communication_route_skill", {}) or {})
        absorption_skill = dict(policy.get("communication_absorption_skill", {}) or {})
        for key in ["initial_evidence", "evolution_notes", "diagnostic_counters"]:
            item_skill.pop(key, None)
        return {
            "user_id": str(policy.get("user_id", "unknown")),
            "dataset": str(policy.get("dataset", self.dataset) or self.dataset),
            "version": self._numeric_version(policy.get("version", 1), default=1),
            "item_selection_skill": item_skill,
            **(
                {"communication_selection_skill": comm_skill}
                if self._has_communication_selection_skill_content(comm_skill)
                else {}
            ),
            "communication_route_skill": route_skill,
            **(
                {"communication_initial_evidence": dict(policy.get("communication_initial_evidence", {}) or {})}
                if dict(policy.get("communication_initial_evidence", {}) or {})
                else {}
            ),
            **(
                {"communication_absorption_skill": absorption_skill}
                if self._has_absorption_skill_content(absorption_skill)
                else {}
            ),
        }

    @staticmethod
    def _dedupe_rule_rows(rows):
        out = []
        seen = set()
        for row in rows or []:
            if isinstance(row, str):
                row = {"rule": row, "confidence": 0.50, "status": "active", "reinforce_count": 0, "weaken_count": 0}
            if not isinstance(row, dict):
                continue
            key = " ".join(str(row.get("rule", "") or "").strip().lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(dict(row))
        return out

    def normalize_policy(self, policy):
        policy = dict(policy or {})
        def add_missing_rules(rows, defaults):
            rows = list(rows or [])
            seen = {" ".join(str(row.get("rule", "") or "").strip().lower().split()) for row in rows if isinstance(row, dict)}
            for row in defaults or []:
                key = " ".join(str(row.get("rule", "") or "").strip().lower().split())
                if key and key not in seen:
                    rows.append(dict(row))
                    seen.add(key)
            return rows

        def prioritize_rules(rows, priority_defaults):
            rows = list(rows or [])
            priority = list(priority_defaults or [])
            seen = set()
            ordered = []
            for row in priority + rows:
                if not isinstance(row, dict) or not str(row.get("rule", "")).strip():
                    continue
                key = " ".join(str(row.get("rule", "") or "").strip().lower().split())
                if key in seen:
                    continue
                ordered.append(dict(row))
                seen.add(key)
            return ordered

        def migrate_rule_text(row):
            row = dict(row or {})
            text = " ".join(str(row.get("rule", "") or "").strip().split())
            low = text.lower()
            if "rank all 20 candidates comparatively" in low and "top 5 ranked candidates" in low:
                row["rule"] = ""
            elif "rank candidates listwise" in low:
                row["rule"] = text.replace("rank candidates listwise against every cluster", "judge candidates through the user's evidence-backed taste clusters")
            elif "reconsider the top-5 order" in low:
                row["rule"] = text.replace(
                    "raise that candidate and reconsider the top-5 order",
                    "raise that candidate into the hesitation set or switch only with concrete item-level proof",
                )
            elif "cooperative-inquiry for sparse evidence" in low and "multi-candidate-debate" in low:
                row["rule"] = (
                    "Choose the how communication mode from the current uncertainty shape: warning modes for removing or down-ranking weak HesitationSet candidates, "
                    "promotion modes for retaining a shorter interested set, cooperative modes for shared evidence completion, "
                    "competitive modes for testing rival claims, and feedback repair modes for follow-up user questions."
                )
            return row

        if "core_decision_reasoning_skill" not in policy and "item_selection_skill" not in policy:
            policy["core_decision_reasoning_skill"] = {
                "active_rules": self._initial_core_rules(
                    history_summary=((policy.get("policy_evolution_state", {}) or {}).get("history_seed", "")),
                    target_profile=", ".join((policy.get("core_preference", {}) or {}).get("long_term_preference", []) or []),
                ),
                "candidate_rules": [],
                "risky_rules": [],
                "inactive_rules": [],
            }
        if "communication_reasoning_skill" not in policy and "communication_selection_skill" not in policy:
            policy["communication_reasoning_skill"] = {
                "active_rules": self._initial_communication_rules(),
                "candidate_rules": [],
                "risky_rules": [],
                "inactive_rules": [],
            }
        if (
            "post_feedback_reasoning_skill" not in policy
            and "communication_selection_skill" not in policy
        ):
            policy["post_feedback_reasoning_skill"] = {
                "active_rules": self._initial_post_feedback_rules(
                    communication_evidence=policy.get("communication_initial_evidence", {}),
                    core_preference=policy.get("core_preference", {}),
                ),
                "candidate_rules": [],
                "risky_rules": [],
                "inactive_rules": [],
            }
        policy.pop("execution_instructions", None)
        for layer in ["core_decision_reasoning_skill", "communication_reasoning_skill", "post_feedback_reasoning_skill"]:
            layer_obj = dict(policy.get(layer, {}) or {})
            for bucket in ["active_rules", "candidate_rules", "risky_rules", "inactive_rules"]:
                rows = []
                for row in list(layer_obj.get(bucket, []) or []):
                    if isinstance(row, str):
                        rows.append(self._rule(row, 0.50, status=bucket.replace("_rules", "")))
                    elif isinstance(row, dict) and str(row.get("rule", "")).strip():
                        normalized = dict(row)
                        normalized["rule"] = str(normalized.get("rule", ""))
                        normalized["confidence"] = float(normalized.get("confidence", 0.50) or 0.50)
                        normalized.setdefault("status", bucket.replace("_rules", ""))
                        normalized.setdefault("reinforce_count", 0)
                        normalized.setdefault("weaken_count", 0)
                        migrated = migrate_rule_text(normalized)
                        if layer == "core_decision_reasoning_skill" and self._is_generic_item_protocol_rule(migrated.get("rule", "")):
                            continue
                        if str(migrated.get("rule", "")).strip():
                            rows.append(migrated)
                layer_obj[bucket] = rows
            if layer == "core_decision_reasoning_skill":
                layer_obj["active_rules"] = add_missing_rules(layer_obj.get("active_rules", []), self._initial_core_rules(
                    history_summary=((policy.get("policy_evolution_state", {}) or {}).get("history_seed", "")),
                    target_profile=", ".join((policy.get("core_preference", {}) or {}).get("long_term_preference", []) or []),
                ))
            if layer == "communication_reasoning_skill":
                layer_obj["active_rules"] = prioritize_rules(
                    add_missing_rules(layer_obj.get("active_rules", []), self._initial_communication_rules()),
                    self._initial_communication_rules(),
                )
            if layer == "post_feedback_reasoning_skill":
                layer_obj["active_rules"] = add_missing_rules(
                    layer_obj.get("active_rules", []),
                    self._initial_post_feedback_rules(
                        communication_evidence=policy.get("communication_initial_evidence", {}),
                        core_preference=policy.get("core_preference", {}),
                    ),
                )
            policy[layer] = layer_obj
        policy.setdefault("version", 1)
        policy.setdefault("evolution_memory", {"recent_success_lessons": [], "recent_failure_diagnoses": [], "compressed_summary": ""})
        comm_pref = dict(policy.get("communication_preference", {}) or {})
        comm_pref["who_preference"] = self._normalize_who_mapping(comm_pref.get("who_preference", {}))
        comm_pref["who_preference"]["trusted-advisors"] = max(
            0.90,
            float(comm_pref["who_preference"].get("trusted-advisors", 0.0) or 0.0),
        )
        for fallback_who in ["similar-users", "experienced-users", "topk-advisors"]:
            comm_pref["who_preference"][fallback_who] = min(
                0.74,
                float(comm_pref["who_preference"].get(fallback_who, 0.50) or 0.50),
            )
        comm_pref.setdefault("trigger_preference", {
            "cold-start": 0.5,
            "candidate-conflict": 0.5,
            "novelty-uncertainty": 0.5,
            "internal-prior-conflict": 0.5,
        })
        comm_pref.pop("what_preference", None)
        comm_pref["how_preference"] = self._normalize_how_mapping(comm_pref.get("how_preference", {}))
        policy["communication_preference"] = comm_pref

        reliability_memory = dict(policy.get("advisor_reliability_memory", {}) or {})
        merged_reliability = {key: dict(reliability_memory.get(key, {}) or {}) for key in ACTIVE_WHO_NODES}
        for old_key, new_key in WHO_NODE_ALIASES.items():
            old_val = dict(reliability_memory.get(old_key, {}) or {})
            if old_val:
                merged_reliability.setdefault(new_key, {}).update(old_val)
        policy["advisor_reliability_memory"] = merged_reliability

        state = dict(policy.get("policy_evolution_state", {}) or {})
        state.setdefault("version", 1)
        state.setdefault("num_updates", 0)
        state.setdefault("last_diagnosis_id", None)
        policy["policy_evolution_state"] = state
        return self._normalize_two_skill_schema(policy)

    def _skill_markdown(self, policy):
        policy = self.normalize_policy(policy)
        item_skill = dict(policy.get("item_selection_skill", {}) or {})
        route_skill = dict(policy.get("communication_route_skill", {}) or {})
        absorption_skill = dict(policy.get("communication_absorption_skill", {}) or {})

        def render_rules(title, rules):
            rows = [f"## {title}\n"]
            if not rules:
                rows.append("- none")
            else:
                for rule in rules:
                    rows.append(f"- ({float(rule.get('confidence', 0.0)):.2f}) {rule.get('rule', '')}")
            return "\n".join(rows)

        def render_route_skill():
            rows = ["## Communication Route Skill\n"]
            rows.append(f"- version: {route_skill.get('version', 1)}")
            rows.append(f"- template_id: {route_skill.get('template_id', '')}")
            rows.append(f"- template_features: {route_skill.get('template_features', {})}")
            rows.append(f"- signature_order: {route_skill.get('signature_order', [])}")
            rows.append("- what_by_why:")
            for key, value in list(dict(route_skill.get("what_by_why", {}) or {}).items())[:6]:
                rows.append(f"  - {key}: {value}")
            rows.append("- how_by_what:")
            for key, value in list(dict(route_skill.get("how_by_what", {}) or {}).items())[:6]:
                rows.append(f"  - {key}: {value}")
            rows.append("- who_by_how:")
            for key, value in list(dict(route_skill.get("who_by_how", {}) or {}).items())[:6]:
                rows.append(f"  - {key}: {value}")
            score_payload = dict(route_skill.get("initial_route_scores", {}) or {})
            who_scores_by_how = dict(score_payload.get("who_by_how", {}) or {})
            if who_scores_by_how:
                rows.append("- initial_route_scores.who_by_how:")
                for key, value in list(who_scores_by_how.items())[:4]:
                    rows.append(f"  - {key}: {value}")
            reason_payload = dict(route_skill.get("initial_route_reasons", {}) or {})
            who_reasons_by_how = dict(reason_payload.get("who_by_how", {}) or {})
            if who_reasons_by_how:
                rows.append("- initial_route_reasons.who_by_how:")
                for key, value in list(who_reasons_by_how.items())[:3]:
                    top_order = list(dict(route_skill.get("who_by_how", {}) or {}).get(key, []) or [])[:2]
                    compact = {
                        who: list(dict(value or {}).get(who, []) or [])[:2]
                        for who in top_order
                    }
                    rows.append(f"  - {key}: {compact}")
            rows.append("- demotions:")
            for row in list(route_skill.get("demotions", []) or [])[:6]:
                rows.append(f"  - level={row.get('level', '')}; scope={row.get('scope', '')}; node={row.get('node', '')}; reason={row.get('reason', '')}")
            rows.append(f"- unmapped_task_memory_count: {len(list(route_skill.get('unmapped_task_memory', []) or []))}")
            rows.append(f"- exploration_slots_count: {len(list(route_skill.get('exploration_slots', []) or []))}")
            rows.append(f"- exploration_history_count: {len(list(route_skill.get('exploration_history', []) or []))}")
            return "\n".join(rows)

        def render_preferences(title, preferences):
            rows = [f"## {title}\n"]
            if not preferences:
                rows.append("- none")
            else:
                evidence_label = "evidence_books" if self._is_book_domain() else ("evidence_products" if self._is_product_domain() else "evidence_artists")
                for pref in preferences:
                    artists = ", ".join(list(pref.get("evidence_artists", []) or [])[:5])
                    evidence = str(pref.get("evidence", "") or "").strip()
                    suffix = f"; {evidence_label}: {artists}" if artists else ""
                    suffix += f"; evidence: {evidence}" if evidence else ""
                    rows.append(
                        f"- ({pref.get('confidence_label', self._confidence_label(pref.get('confidence', 0.0)))}) "
                        f"{pref.get('attribute', '')}{suffix}"
                    )
            return "\n".join(rows)

        absorption_section = (
            f"{render_rules('Communication Absorption Skill', absorption_skill.get('active_rules', []))}\n\n"
            if self._has_absorption_skill_content(absorption_skill)
            else ""
        )
        return (
            f"---\n"
            f"name: user-skill-{policy.get('user_id', 'unknown')}\n"
            f"description: User Skill for user {policy.get('user_id', 'unknown')}\n"
            f"---\n\n"
            f"# User Skill\n\n"
            f"User: {policy.get('user_id', 'unknown')}\n\n"
            f"{render_preferences('Item Selection Skill: Preferences', item_skill.get('preferences', []))}\n\n"
            f"{render_preferences('Item Selection Skill: Recent Signals', item_skill.get('recent_signals', []))}\n\n"
            f"- decision_style: {item_skill.get('decision_style', 'none')}\n\n"
            f"{render_route_skill()}\n\n"
            f"{absorption_section}"
        )

    def save_full_policy(self, policy, snapshot_reason="update"):
        policy = self.normalize_policy(policy)
        user_raw = str(policy.get("user_id", "unknown"))
        paths = self._paths(user_raw)
        (paths["dir"] / "references").mkdir(parents=True, exist_ok=True)
        (paths["dir"] / "assets").mkdir(parents=True, exist_ok=True)
        paths["skill_md"].write_text(self._skill_markdown(policy), encoding="utf-8")
        dump_json(paths["policy_json"], policy)
        append_jsonl(
            paths["snapshots_jsonl"],
            {
                "snapshot_reason": str(snapshot_reason or "update"),
                "policy": policy,
            },
        )

    def reset_user_logs(self, user_raw, include_initial=False):
        """Clear append-only artifacts when rebuilding a user's policy from scratch."""
        path_sets = [self._paths(user_raw)]
        if include_initial:
            initial_paths = self._initial_paths(user_raw)
            if initial_paths is not None:
                path_sets.append(initial_paths)
        for paths in path_sets:
            (paths["dir"] / "references").mkdir(parents=True, exist_ok=True)
            (paths["dir"] / "assets").mkdir(parents=True, exist_ok=True)
            for key in [
                "snapshots_jsonl",
                "evolution_log_jsonl",
                "interaction_diagnoses_jsonl",
            ]:
                try:
                    paths[key].write_text("", encoding="utf-8")
                except Exception:
                    pass
            for key in [
                "slim_cache_json",
                "slim_cache_proposal_json",
                "slim_cache_communication_json",
                "slim_cache_post_feedback_json",
            ]:
                try:
                    if paths[key].exists():
                        paths[key].unlink()
                except Exception:
                    pass

    def save_initial_policy_if_missing(self, policy, overwrite=False):
        if self.initial_base_dir is None:
            return False
        policy = self.normalize_policy(policy)
        user_raw = str(policy.get("user_id", "unknown"))
        paths = self._initial_paths(user_raw)
        if paths is None or (paths["policy_json"].exists() and not overwrite):
            return False
        (paths["dir"] / "references").mkdir(parents=True, exist_ok=True)
        (paths["dir"] / "assets").mkdir(parents=True, exist_ok=True)
        paths["skill_md"].write_text(self._skill_markdown(policy), encoding="utf-8")
        dump_json(paths["policy_json"], policy)
        append_jsonl(
            paths["snapshots_jsonl"],
            {
                "snapshot_reason": "initial_bootstrap",
                "policy": policy,
            },
        )
        append_jsonl(
            paths["evolution_log_jsonl"],
            {
                "event": "initial_policy_snapshot_created",
                "stage": "bootstrap",
            },
        )
        return True

    def append_evolution_log(self, user_raw, event):
        paths = self._paths(user_raw)
        (paths["dir"] / "references").mkdir(parents=True, exist_ok=True)
        append_jsonl(paths["evolution_log_jsonl"], event)

    def append_interaction_diagnosis(self, user_raw, diagnosis):
        paths = self._paths(user_raw)
        (paths["dir"] / "references").mkdir(parents=True, exist_ok=True)
        append_jsonl(paths["interaction_diagnoses_jsonl"], diagnosis)

    def load_full_policy(
        self,
        user_raw,
        history_summary="",
        target_profile="",
        stage="test",
        communication_evidence=None,
        core_rules=None,
        core_preference=None,
        core_initial_evidence=None,
        force_bootstrap=False,
    ):
        paths = self._paths(user_raw)
        if force_bootstrap and str(stage or "").lower() == "train":
            self.reset_user_logs(user_raw, include_initial=True)
        read_paths = None if force_bootstrap else self._existing_paths(user_raw)
        policy = None
        if read_paths is not None and read_paths["policy_json"].exists():
            policy = load_json(read_paths["policy_json"], default=None)
        if policy is not None:
            policy = self.normalize_policy(policy)
            if read_paths != paths and str(stage or "").lower() == "train":
                self.save_full_policy(policy, snapshot_reason="migrate_from_model_namespace")
                self.append_evolution_log(
                    user_raw,
                    {
                        "event": "policy_migrated_to_dataset_namespace",
                        "stage": str(stage),
                        "legacy_policy_dir": str(read_paths["dir"]),
                        "dataset_policy_dir": str(paths["dir"]),
                    },
                )
            return policy, "persisted"
        policy = self.normalize_policy(self._default_policy(
            user_raw=user_raw,
            history_summary=history_summary,
            target_profile=target_profile,
            communication_evidence=communication_evidence,
            core_rules=core_rules,
            core_preference=core_preference,
            core_initial_evidence=core_initial_evidence,
        ))
        if str(stage or "").lower() == "train":
            self.save_initial_policy_if_missing(policy, overwrite=force_bootstrap)
            self.save_full_policy(policy, snapshot_reason="bootstrap")
            self.append_evolution_log(
                user_raw,
                {
                    "event": "bootstrap_policy_created",
                    "stage": str(stage),
                    "force_bootstrap": bool(force_bootstrap),
                },
            )
            return policy, "rebootstrapped" if force_bootstrap else "bootstrapped"
        return policy, "ephemeral_default"

    def build_slim_policy(self, full_policy, decision_context):
        full_policy = self.normalize_policy(full_policy)
        decision_context = dict(decision_context or {})
        phase = str(decision_context.get("phase", "proposal") or "proposal").strip().lower()
        primary_trigger = str(decision_context.get("primary_trigger", "") or "")
        uncertainty_points = [str(x) for x in (decision_context.get("uncertainty_points", []) or [])]
        reliability_memory = dict((full_policy or {}).get("advisor_reliability_memory", {}) or {})
        confidence_calibration = dict((full_policy or {}).get("confidence_calibration", {}) or {})
        item_skill = dict((full_policy or {}).get("item_selection_skill", {}) or {})
        comm_skill = dict((full_policy or {}).get("communication_selection_skill", {}) or {})
        route_skill = dict((full_policy or {}).get("communication_route_skill", {}) or {})
        post_skill = dict((full_policy or {}).get("post_feedback_skill", {}) or {})
        absorption_skill = dict((full_policy or {}).get("communication_absorption_skill", {}) or {})

        def pref_weight(row):
            row = dict(row or {})
            base = float(row.get("confidence", 0.0) or 0.0)
            base += 0.035 * int(row.get("learning_priority", 0) or 0)
            base += 0.010 * min(3, int(row.get("reinforce_count", 0) or 0))
            base -= 0.020 * min(3, int(row.get("weaken_count", 0) or 0))
            return base

        def medium_or_high_pref(row):
            try:
                return float((row or {}).get("confidence", 0.0) or 0.0) >= 0.45
            except Exception:
                return False

        item_preferences = self._diverse_preference_rows(
            sorted(
                [
                    row for row in list(item_skill.get("preferences", []) or [])
                    if isinstance(row, dict)
                    and medium_or_high_pref(row)
                    and not self._is_generic_item_protocol_rule(row.get("attribute", ""))
                ],
                key=pref_weight,
                reverse=True,
            ),
            limit=5,
        )
        recent_signals = self._diverse_preference_rows(
            sorted(
                [row for row in list(item_skill.get("recent_signals", []) or []) if isinstance(row, dict) and medium_or_high_pref(row)],
                key=pref_weight,
                reverse=True,
            ),
            limit=3,
        )
        who_pref_rows = sorted(list(comm_skill.get("who_preferences", []) or []), key=pref_weight, reverse=True)
        how_pref_rows = sorted(list(comm_skill.get("how_preferences", []) or []), key=pref_weight, reverse=True)
        trigger_rows = sorted(list(comm_skill.get("trigger_rules", []) or []), key=lambda row: float((row or {}).get("confidence", 0.0) or 0.0), reverse=True)

        planning_condition = dict(decision_context.get("planning_condition", {}) or {})
        if not planning_condition:
            planning_condition = {
                "round_type": str(decision_context.get("round_type", "") or ("repair" if decision_context.get("updated_memory") else "initial")),
                "primary_trigger": primary_trigger,
                "focus_set_size": int(decision_context.get("focus_set_size", len(decision_context.get("shortlist", []) or [])) or 0),
                "confidence_band": (
                    "high" if int(decision_context.get("self_confidence", 0) or 0) >= 75
                    else ("medium" if int(decision_context.get("self_confidence", 0) or 0) >= 50 else "low")
                ),
                "uncertainty_shape": "candidate-conflict" if any("candidate" in str(x) for x in uncertainty_points) else (primary_trigger or "proposal-risk-check"),
                "previous_feedback_exists": bool(decision_context.get("updated_memory")),
            }
        item_comm_features = self._communication_history_features(item_skill=item_skill, history_summary="")
        try:
            context_history_count = int(decision_context.get("history_count", planning_condition.get("history_count", 0)) or 0)
        except Exception:
            context_history_count = 0
        if context_history_count > 0:
            if context_history_count <= 3:
                context_sparsity = "sparse"
            elif context_history_count <= 8:
                context_sparsity = "medium"
            else:
                context_sparsity = "rich"
            planning_condition.setdefault("history_count", context_history_count)
            planning_condition.setdefault("history_sparsity", context_sparsity)
        else:
            planning_condition.setdefault("history_sparsity", item_comm_features.get("history_sparsity", "sparse"))
        planning_condition.setdefault("preference_cluster_count", item_comm_features.get("preference_cluster_count", 0))
        planning_condition.setdefault("strong_item_rules", bool(item_comm_features.get("strong_item_rules", False)))
        planning_condition.setdefault("novelty_preference", bool(item_comm_features.get("novelty_or_minority", False)))

        def row_confidence(row, default=0.50):
            row = dict(row or {})
            value = row.get("confidence", default)
            if isinstance(value, str):
                return {"high": 0.75, "medium": 0.55, "low": 0.35}.get(value.strip().lower(), default)
            try:
                return float(value)
            except Exception:
                return float(default)

        def condition_matches(condition):
            condition = dict(condition or {})
            if not condition:
                return True
            for key, expected in condition.items():
                if key == 'why':
                    actual = planning_condition.get("selected_why") or planning_condition.get("primary_trigger")
                    if expected not in ["", None] and str(actual) != str(expected):
                        return False
                    continue
                if key.endswith("_min"):
                    actual_key = key[:-4]
                    try:
                        if float(planning_condition.get(actual_key, 0) or 0) < float(expected):
                            return False
                    except Exception:
                        return False
                    continue
                actual = planning_condition.get(key)
                if isinstance(expected, list):
                    if str(actual) not in [str(x) for x in expected]:
                        return False
                elif isinstance(expected, bool):
                    if bool(actual) != expected:
                        return False
                elif expected not in ["", None] and str(actual) != str(expected):
                    return False
            return True

        item_rules = [
            {"rule": str(row.get("rule", "")), "confidence": float(row.get("confidence", 0.0) or 0.0)}
            for row in list(item_skill.get("active_rules", []) or [])
            if str(row.get("rule", "") or "").strip() and medium_or_high_pref(row)
        ]
        comm_rules = [
            {"rule": str(row.get("rule", "")), "confidence": float(row.get("confidence", 0.0) or 0.0)}
            for row in list(comm_skill.get("active_rules", []) or [])
            if str(row.get("rule", "") or "").strip() and medium_or_high_pref(row)
        ]
        post_rules = [
            {"rule": str(row.get("rule", "")), "confidence": float(row.get("confidence", 0.0) or 0.0)}
            for row in (
                list(post_skill.get("post_feedback_trust_rules", []) or [])
                + list(post_skill.get("continue_rules", []) or [])
                + list(post_skill.get("harm_avoidance_rules", []) or [])
                + list(post_skill.get("missing_evidence_rules", []) or [])
                + list(post_skill.get("active_rules", []) or [])
            )
            if isinstance(row, dict) and str(row.get("rule", "") or "").strip() and medium_or_high_pref(row)
        ]
        absorption_rules = [
            {"rule": str(row.get("rule", "")), "confidence": float(row.get("confidence", 0.0) or 0.0)}
            for row in list(absorption_skill.get("active_rules", []) or [])
            if isinstance(row, dict) and str(row.get("rule", "") or "").strip() and medium_or_high_pref(row)
        ]
        absorption_cases = [
            dict(row)
            for row in list(absorption_skill.get("ignored_advisor_signal_cases", []) or [])
            if isinstance(row, dict)
        ][-2:]
        absorption_payload = None
        if phase in ["post_feedback", "post-feedback", "feedback", "redecision"] and (absorption_rules or absorption_cases):
            absorption_payload = {
                "version": self._numeric_version(absorption_skill.get("version", 1), default=1),
                "active_rules": absorption_rules[:1],
                "ignored_advisor_signal_cases": absorption_cases,
                "confidence": float(absorption_skill.get("confidence", 0.50) or 0.50),
            }
        risky_rules = [
            {"rule": str(row.get("rule", "")), "confidence": float(row.get("confidence", 0.0) or 0.0)}
            for row in (
                list(item_skill.get("risky_rules", []) or [])[:1]
                + list(comm_skill.get("risky_rules", []) or [])[:1]
            )
            if str(row.get("rule", "") or "").strip() and medium_or_high_pref(row)
        ]

        if phase in ["communication", "comm", "communication_selection"]:
            selected_item = []
            selected_comm = comm_rules[:3]
        elif phase in ["post_feedback", "post-feedback", "feedback", "redecision"]:
            selected_item = item_rules[:2]
            selected_comm = comm_rules[:2]
        else:
            selected_item = item_rules[:5]
            selected_comm = []

        trigger_strength = 0.5
        for row in trigger_rows:
            if str(row.get("condition", "") or "") == primary_trigger or str(row.get("mode", "") or "") == primary_trigger:
                trigger_strength = float(row.get("confidence", 0.5) or 0.5)
                break

        return {
            "user_id": str((full_policy or {}).get("user_id", "")),
            "phase": phase,
            "decision_context": {
                "primary_trigger": primary_trigger,
                "proposal_item": str(decision_context.get("proposal_item", "") or ""),
                "shortlist": [str(x) for x in (decision_context.get("shortlist", []) or [])],
                "uncertainty_points": list(uncertainty_points),
            },
            "item_selection_skill": {
                "preferences": item_preferences,
                "recent_signals": recent_signals,
                "decision_style": str(item_skill.get("decision_style", "") or ""),
            },
            "communication_selection_skill": ({
                "trigger_strength": float(trigger_strength),
                "top_who_preferences": who_pref_rows[:2],
                "top_how_preferences": how_pref_rows[:2],
                "trigger_rules": trigger_rows[:3],
                "path_memory": list(comm_skill.get("path_memory", []) or [])[:3],
                "communication_round_memory": list(comm_skill.get("communication_round_memory", []) or [])[:3],
                "advisor_reliability_memory": list(comm_skill.get("advisor_reliability_memory", []) or [])[:3],
            } if phase in ["communication", "comm", "post_feedback", "post-feedback", "feedback", "redecision", "communication_selection"] else {}),
            "communication_route_skill": ({
                "version": self._numeric_version(route_skill.get("version", 1), default=1),
                "template_id": str(route_skill.get("template_id", "") or ""),
                "template_features": dict(route_skill.get("template_features", {}) or {}),
                "signature_order": list(route_skill.get("signature_order", []) or [])[:12],
                "what_by_why": dict(route_skill.get("what_by_why", {}) or {}),
                "how_by_what": dict(route_skill.get("how_by_what", {}) or {}),
                "who_by_how": dict(route_skill.get("who_by_how", {}) or {}),
                "initial_route_scores": dict(route_skill.get("initial_route_scores", {}) or {}),
                "initial_route_reasons": dict(route_skill.get("initial_route_reasons", {}) or {}),
                "child_order_memory": dict(route_skill.get("child_order_memory", {}) or {}),
                "demotions": list(route_skill.get("demotions", []) or [])[-16:],
                "unmapped_task_memory": list(route_skill.get("unmapped_task_memory", []) or [])[-12:],
                "exploration_slots": list(route_skill.get("exploration_slots", []) or [])[-12:],
                "exploration_history": list(route_skill.get("exploration_history", []) or [])[-12:],
            } if phase in ["communication", "comm", "communication_selection"] else {}),
            "post_feedback_skill": ({
                "post_feedback_trust_rules": post_rules[:4],
                "continue_rules": list(post_skill.get("continue_rules", []) or [])[:3],
                "harm_avoidance_rules": list(post_skill.get("harm_avoidance_rules", []) or [])[:3],
                "missing_evidence_rules": list(post_skill.get("missing_evidence_rules", []) or [])[:3],
            } if phase in ["post_feedback", "post-feedback", "feedback", "redecision"] else {}),
            **({"communication_absorption_skill": absorption_payload} if absorption_payload else {}),
            "retrieved_reasoning_rules": {
                "item_selection": selected_item,
                "communication_selection": selected_comm,
                "post_feedback": post_rules[:4] if phase in ["post_feedback", "post-feedback", "feedback", "redecision"] else [],
                **({"communication_absorption": absorption_rules[:1]} if absorption_payload and absorption_rules else {}),
                "core_decision": selected_item,
                "communication": selected_comm,
                "risky": risky_rules,
            },
            "retrieved_reliability_bias": ({
                key: dict(val or {})
                for key, val in reliability_memory.items()
                if key in ACTIVE_WHO_NODES
            } if phase in ["communication", "comm", "communication_selection", "post_feedback", "post-feedback", "feedback", "redecision"] else {}),
            "confidence_calibration": ({
                "base_confidence_bias": float(confidence_calibration.get("base_confidence_bias", 0.0)),
                "uncertainty_sensitivity": dict(confidence_calibration.get("uncertainty_sensitivity", {}) or {}),
            } if phase in ["proposal", "post_feedback", "post-feedback", "feedback", "redecision"] else {}),
        }

    def cache_slim_policy(self, user_raw, slim_policy, phase="latest", round_info=None, compact_slim_policy=None):
        paths = self._paths(user_raw)
        (paths["dir"] / "assets").mkdir(parents=True, exist_ok=True)
        cache_payload = slim_policy
        if compact_slim_policy is not None:
            cache_payload = {
                "user_id": str(user_raw),
                "phase": str(phase or "latest"),
                "compact_slim_policy": str(compact_slim_policy),
                "source": "llm_distilled_from_current_slim_skill",
            }
            if round_info:
                cache_payload.update(dict(round_info or {}))
        dump_json(paths["slim_cache_json"], cache_payload)
        phase_key = str(phase or "latest").strip().lower().replace("-", "_")
        phase_path_key = f"slim_cache_{phase_key}_json"
        if phase_path_key in paths:
            dump_json(paths[phase_path_key], cache_payload)
        import datetime
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "phase": phase,
            "slim_policy": slim_policy,
        }
        if compact_slim_policy is not None:
            log_entry["compact_slim_policy"] = compact_slim_policy
        if round_info:
            log_entry.update(round_info)
        append_jsonl(paths["slim_policy_log_jsonl"], log_entry)
