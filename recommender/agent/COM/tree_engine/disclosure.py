import json
import re


class UserDisclosureBuilder:
    """Build privacy-filtered requester context for advisors."""

    PRIVATE_FIELDS = {
        "full_history",
        "target_item",
        "target_items",
        "gt_items",
        "private_active_rules",
        "path_memory",
        "post_feedback_rules",
        "post_feedback_skill",
        "training_reflections",
        "evolution_memory",
    }

    def _compact(self, value, max_len=900):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 3)].rstrip() + "..."

    def _extract_preference_summary(self, private_item_slim_skill):
        if isinstance(private_item_slim_skill, str):
            text = self._compact(private_item_slim_skill, max_len=420)
            if not text:
                return []
            parts = [seg.strip() for seg in text.split("|") if seg.strip()]
            public_parts = [
                seg
                for seg in parts
                if seg.lower().startswith("likes:") or seg.lower().startswith("style:")
            ]
            return [" | ".join(public_parts)] if public_parts else [text]
        skill = dict(private_item_slim_skill or {})
        item_skill = dict(skill.get("item_selection_skill", {}) or {})
        rows = []
        for pref in list(item_skill.get("preferences", []) or [])[:4]:
            if isinstance(pref, dict):
                label = str(pref.get("attribute", "") or "").strip()
                if label:
                    rows.append(label)
            elif str(pref or "").strip():
                rows.append(str(pref).strip())
        style = str(item_skill.get("decision_style", "") or "").strip()
        if style:
            rows.append(f"decision_style={style}")
        if not rows and skill:
            rows.append(self._compact(skill, max_len=420))
        return rows[:5]

    def build_shareable_item_brief(
        self,
        private_item_slim_skill,
        stage1_state,
        focus_candidates,
        selected_why=None,
        selected_how=None,
    ):
        state = dict(stage1_state or {})
        focus = [str(x) for x in list(focus_candidates or []) if str(x or "").strip()]
        uncertainty = [str(x) for x in list(state.get("uncertainty_points", []) or []) if str(x or "").strip()]
        brief = {
            "requester_shareable_item_brief": {
                "relevant_preference_summary": self._extract_preference_summary(private_item_slim_skill),
                "current_uncertainty": uncertainty,
                "privacy": "full history, private rules, target labels, and path memory are hidden",
            },
            "privacy_filter_log": {
                "removed_fields": sorted(self.PRIVATE_FIELDS),
                "filter": "rule_based_shareable_brief_v1",
            },
        }
        return brief
