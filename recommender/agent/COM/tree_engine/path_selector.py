from recommender.agent.COM.tree_engine.schemas import build_communication_path
from recommender.agent.COM.tree_engine.public_tree import DEPRECATED_NODE_ALIASES, infer_communication_shape
from recommender.agent.COM.tree_engine.task_planner import retarget_first_round_task
import re

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
    "missing-evidence",
    "default",
]


class PublicTreePathSelector:
    def __init__(self, public_tree_store, args=None):
        self.public_tree_store = public_tree_store
        self.args = args

    def _tree_trial_exploration_enabled(self):
        return bool(getattr(self.args, "com_enable_tree_trial_exploration", False))

    @staticmethod
    def _norm(value):
        return str(value or "").strip()

    @staticmethod
    def _path_key(why, who, how, what=""):
        parts = [str(why or "")]
        if str(what or "").strip():
            parts.append(str(what or ""))
        parts.extend([str(who or ""), str(how or "")])
        return " -> ".join(parts)

    @staticmethod
    def _focus_count(decision_state):
        state = dict(decision_state or {})
        for key in ["candidate_shortlist", "shortlist", "focus_candidates"]:
            rows = [str(x).strip() for x in list(state.get(key, []) or []) if str(x).strip()]
            if rows:
                return len(rows)
        return 0

    def _canonical_node(self, level, node_id, decision_state=None):
        node_id = str(node_id or "").strip()
        return str((DEPRECATED_NODE_ALIASES.get(level, {}) or {}).get(node_id, node_id))

    def _active_options(self):
        stage = str(getattr(self.args, "run_stage", "test") or "test").strip().lower()
        selectable_stage = stage if self._tree_trial_exploration_enabled() else "test"
        options = {
            'why': list(self.public_tree_store.get_selectable_nodes('why', stage=selectable_stage).keys()),
            "what": list(self.public_tree_store.get_selectable_nodes("what", stage=selectable_stage).keys()),
            "who": list(self.public_tree_store.get_selectable_nodes("who", stage=selectable_stage, include_children=False).keys()),
            "who_branches": list(self.public_tree_store.get_selectable_nodes("who", stage=selectable_stage, include_children=True).keys()),
            "how": list(self.public_tree_store.get_selectable_nodes("how", stage=selectable_stage).keys()),
        }
        if not self._tree_trial_exploration_enabled():
            tree = self.public_tree_store.load_tree()
            # There is no per-user route injection path for why triggers, so
            # generated route-injection-only why nodes must not become global
            # triggers through selection_profile matching.
            options['why'] = [
                node_id
                for node_id in options['why']
                if not bool(((tree.get('why', {}) or {}).get(node_id, {}) or {}).get("route_injection_only", False))
            ]
        return options

    @staticmethod
    def _as_float(value, default=0.0):
        if isinstance(value, str):
            value = {"high": 0.75, "medium": 0.55, "low": 0.35}.get(value.strip().lower(), default)
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _parent_node_id(node_id):
        node_id = str(node_id or "").strip()
        return "/".join(node_id.split("/")[:-1]) if "/" in node_id else ""

    @staticmethod
    def _node_depth(node_id):
        return len([part for part in str(node_id or "").strip("/").split("/") if part])

    @staticmethod
    def _text_blob(*values):
        chunks = []
        for value in values:
            if isinstance(value, dict):
                for k, v in value.items():
                    chunks.append(str(k))
                    chunks.append(PublicTreePathSelector._text_blob(v))
            elif isinstance(value, (list, tuple, set)):
                chunks.append(" ".join(PublicTreePathSelector._text_blob(x) for x in value))
            else:
                chunks.append(str(value or ""))
        return " ".join(chunks).lower()

    @staticmethod
    def _node_tokens(node_id, node):
        text = " ".join(
            [
                str(node_id or ""),
                str((node or {}).get("description", "") or ""),
                str((node or {}).get("use_why", "") or ""),
                str((node or {}).get("if_selected", "") or ""),
                str((node or {}).get("evidence_pattern", "") or ""),
                str((node or {}).get("skill_body", "") or ""),
                PublicTreePathSelector._text_blob((node or {}).get("applicability_condition", {})),
                PublicTreePathSelector._text_blob((node or {}).get("execution_hint", {})),
            ]
        ).lower()
        tokens = set()
        for token in re.split(r"[^a-z0-9]+", text):
            if len(token) >= 5 and token not in {"status", "sprout", "active", "selected", "advisor", "candidate"}:
                tokens.add(token)
        return tokens

    def _apply_child_trial_scores(self, level, scores, reasons, planning_payload, condition, extra_context=""):
        tree = self.public_tree_store.load_tree()
        nodes = dict((tree.get(level, {}) or {}))
        context_text = self._text_blob(
            condition,
            extra_context,
            (planning_payload or {}).get("open_condition_memory", []),
            (planning_payload or {}).get("tree_need_signals", []),
            (planning_payload or {}).get("similar_failure_cases", []),
            (planning_payload or {}).get("repair_rules", []),
        )
        for node_id in list(scores.keys()):
            parent = self._parent_node_id(node_id)
            if not parent or parent not in scores:
                continue
            node = dict(nodes.get(node_id, {}) or {})
            parent_score = float(scores.get(parent, 0.0) or 0.0)
            inherited = parent_score * 0.65
            if inherited > float(scores.get(node_id, 0.0) or 0.0):
                scores[node_id] = inherited
                reasons.setdefault(node_id, []).append(f"child inherits parent {parent} score: {inherited:.2f}")
            if str(node.get("status", "") or "") == "sprout":
                scores[node_id] -= 0.08
                reasons.setdefault(node_id, []).append("sprout trial penalty: -0.08")
            tokens = self._node_tokens(node_id, node)
            matched = [tok for tok in tokens if tok in context_text]
            if matched:
                bonus = min(0.25, 0.10 + 0.03 * min(5, len(matched)))
                scores[node_id] += bonus
                reasons.setdefault(node_id, []).append(f"child trial signal match({','.join(matched[:4])}): +{bonus:.2f}")
                slug = str(node_id).split("/")[-1].lower()
                if slug and (slug in context_text or str(node_id).lower() in context_text):
                    direct_score = parent_score + 0.02
                    if direct_score > float(scores.get(node_id, 0.0) or 0.0):
                        scores[node_id] = direct_score
                        reasons.setdefault(node_id, []).append("direct sprout hint can overtake parent: +0.02 over parent")
            prior = self._as_float(node.get("selection_prior", 0.0), 0.0)
            if prior > 0:
                bonus = min(0.03, 0.05 * prior)
                scores[node_id] += bonus
                reasons.setdefault(node_id, []).append(f"child selection_prior: +{bonus:.2f}")

    def _planning_payload(self, slim_user_policy):
        return {}

    def _route_payload(self, slim_user_policy):
        if isinstance(slim_user_policy, str):
            return {}
        route = dict((slim_user_policy or {}).get("communication_route_skill", {}) or {})
        if not route:
            return {}
        old_what = dict(route.get("what_by_signature", {}) or {})
        old_how = dict(route.get("how_by_signature_what", {}) or {})
        old_who = dict(route.get("who_by_signature_what_how", {}) or {})
        if not route.get("what_by_why") and old_what:
            route["what_by_why"] = old_what
        if not route.get("how_by_what") and old_how:
            migrated = {}
            for key, value in old_how.items():
                parts = str(key or "").split("|")
                what_key = parts[-1] if parts else str(key or "")
                if what_key and what_key not in migrated:
                    migrated[what_key] = list(value or [])
            route["how_by_what"] = migrated
        if not route.get("who_by_how") and old_who:
            migrated = {}
            for key, value in old_who.items():
                parts = str(key or "").split("|")
                how_key = parts[-1] if parts else str(key or "")
                if how_key and how_key not in migrated:
                    migrated[how_key] = list(value or [])
            route["who_by_how"] = migrated
        route.setdefault("child_order_memory", {})
        return route

    def _default_route_payload(self, options):
        what_nodes = list(options.get("what", []) or [])
        how_nodes = list(options.get("how", []) or [])
        who_nodes = list(options.get("who", []) or [])
        if not self._tree_trial_exploration_enabled():
            tree = self.public_tree_store.load_tree()

            def non_injection(level, nodes):
                rows = []
                level_nodes = dict((tree.get(level, {}) or {}) if level else {})
                for node_id in list(nodes or []):
                    node_id = str(node_id or "")
                    node = dict(level_nodes.get(node_id, {}) or {})
                    if bool(node.get("route_injection_only", False)):
                        continue
                    rows.append(node_id)
                return rows

            what_nodes = non_injection("what", what_nodes)
            how_nodes = non_injection("how", how_nodes)
            who_nodes = non_injection("who", who_nodes)

        def keep(nodes, preferred):
            rows = [x for x in preferred if x in nodes]
            rows.extend([x for x in nodes if x not in rows])
            return rows

        what_by_why = {
            "internal-prior-conflict+candidate-conflict": keep(what_nodes, ["reasoning_check", "compare_remaining_candidates", "reduce_hesitation_set", "none"]),
            "novelty-uncertainty+candidate-conflict": keep(what_nodes, ["find_interested_subset", "reduce_hesitation_set", "compare_remaining_candidates", "none"]),
            "cold-start+candidate-conflict": keep(what_nodes, ["evidence_gap_check", "reduce_hesitation_set", "compare_remaining_candidates", "none"]),
            "candidate-conflict": keep(what_nodes, ["reduce_hesitation_set", "compare_remaining_candidates", "reasoning_check", "none"]),
            "internal-prior-conflict": keep(what_nodes, ["reasoning_check", "compare_remaining_candidates", "none"]),
            "novelty-uncertainty": keep(what_nodes, ["find_interested_subset", "evidence_gap_check", "none"]),
            "cold-start": keep(what_nodes, ["evidence_gap_check", "none"]),
            "missing-evidence": keep(what_nodes, ["evidence_gap_check", "none"]),
            "default": keep(what_nodes, ["compare_remaining_candidates", "reduce_hesitation_set", "none"]),
        }
        how_by_what = {
            "reasoning_check": keep(how_nodes, ["multi-competitive", "multi-cooperative", "single-advisor"]),
            "compare_remaining_candidates": keep(how_nodes, ["multi-competitive", "multi-cooperative", "single-advisor"]),
            "reduce_hesitation_set": keep(how_nodes, ["multi-cooperative", "multi-competitive", "single-advisor"]),
            "find_interested_subset": keep(how_nodes, ["multi-cooperative", "single-advisor", "multi-competitive"]),
            "evidence_gap_check": keep(how_nodes, ["multi-cooperative", "single-advisor", "multi-competitive"]),
            "none": keep(how_nodes, ["multi-cooperative", "single-advisor", "multi-competitive"]),
            "default": keep(how_nodes, ["multi-cooperative", "multi-competitive", "single-advisor"]),
        }
        default_who = keep(who_nodes, ["similar-users", "experienced-users", "trusted-advisors", "topk-advisors"])
        who_by_how = {
            "multi-competitive": list(default_who),
            "multi-cooperative": list(default_who),
            "single-advisor": list(default_who),
            "default": list(default_who),
        }
        return {
            "version": 2,
            "template_id": "runtime-route-default",
            "template_features": {},
            "signature_order": list(DEFAULT_TRIGGER_SIGNATURES),
            "what_by_why": what_by_why,
            "how_by_what": how_by_what,
            "who_by_how": who_by_how,
            "child_order_memory": {},
            "what_by_signature": {},
            "how_by_signature_what": {},
            "who_by_signature_what_how": {},
            "demotions": [],
            "unmapped_task_memory": [],
            "exploration_slots": [],
            "exploration_history": [],
        }

    def _trigger_signature(self, matched_why):
        priority = {name: idx for idx, name in enumerate(TRIGGER_SIGNATURE_PRIORITY)}
        rows = []
        for row in self._as_list(matched_why):
            row = self._canonical_node('why', row)
            if row and row not in rows and row not in ["skip", "none"]:
                rows.append(row)
        rows.sort(key=lambda x: priority.get(x, 99))
        return "+".join(rows) if rows else "default"

    def _signature_fallbacks(self, signature):
        rows = []
        signature = str(signature or "").strip() or "default"
        if signature:
            rows.append(signature)
        parts = [x for x in signature.split("+") if x]
        priority = {name: idx for idx, name in enumerate(TRIGGER_SIGNATURE_PRIORITY)}
        parts = sorted(parts, key=lambda x: priority.get(x, 99))
        if parts:
            rows.append(parts[0])
        if "candidate-conflict" in parts:
            rows.append("candidate-conflict")
        rows.append("default")
        out = []
        for row in rows:
            if row and row not in out:
                out.append(row)
        return out

    def _route_order(self, route_payload, bucket, keys, active_nodes):
        data = dict((route_payload or {}).get(bucket, {}) or {})
        active = [str(x) for x in list(active_nodes or []) if str(x).strip()]
        for key in self._as_list(keys):
            order = [
                self._canonical_node("how" if bucket.startswith("how_") else ("what" if bucket.startswith("what_") else "who"), x)
                for x in list(data.get(str(key), []) or [])
            ]
            order = [x for x in order if x in active]
            if order:
                return order, str(key)
        return [], ""

    def _route_order_with_legacy(self, route_payload, new_bucket, new_keys, old_bucket, old_keys, active_nodes):
        order, scope_key = self._route_order(route_payload, new_bucket, new_keys, active_nodes)
        if order:
            return order, scope_key, new_bucket
        order, scope_key = self._route_order(route_payload, old_bucket, old_keys, active_nodes)
        return order, scope_key, old_bucket if order else ""

    def _flat_what_order(self, route_payload, active_nodes):
        """Build a generic what order without looking up any explicit why/when key."""
        active = [self._canonical_node("what", node) for node in list(active_nodes or []) if str(node or "").strip()]
        active = list(dict.fromkeys(node for node in active if node))
        buckets = dict((route_payload or {}).get("what_by_why", {}) or {})
        scores = {node: 0.0 for node in active}
        default_order = [self._canonical_node("what", node) for node in list(buckets.get("default", []) or [])]
        for index, node in enumerate(default_order):
            if node in scores:
                scores[node] += 2.0 / float(index + 1)
        for bucket_name in sorted(buckets):
            if bucket_name == "default":
                continue
            for index, node in enumerate([self._canonical_node("what", value) for value in list(buckets.get(bucket_name, []) or [])]):
                if node in scores:
                    scores[node] += 1.0 / float(index + 1)
        return sorted(active, key=lambda node: (-scores.get(node, 0.0), node))

    def _demotion_rows(self, route_payload, level, scope):
        scope = str(scope or "")
        rows = []
        for row in list((route_payload or {}).get("demotions", []) or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("level", "") or "") != str(level):
                continue
            row_scope = str(row.get("scope", "") or "")
            if row_scope and row_scope not in [scope, "default"] and not scope.startswith(row_scope):
                continue
            rows.append(row)
        return rows

    def _apply_route_demotions(self, order, route_payload, level, scope):
        order = [str(x) for x in list(order or []) if str(x).strip()]
        demoted = []
        for row in self._demotion_rows(route_payload, level, scope):
            node = self._canonical_node(level, row.get("node", ""))
            if node in order:
                order = [x for x in order if x != node] + [node]
                demoted.append(dict(row))
        return order, demoted

    def _node_is_demoted(self, route_payload, level, scope, node):
        node = self._canonical_node(level, node)
        if not node:
            return False
        for row in self._demotion_rows(route_payload, level, scope):
            if self._canonical_node(level, row.get("node", "")) == node:
                return True
        return False

    def _route_node_trial_stats(self, route_payload, level, scope, node):
        stats = {
            "trial_count": 0,
            "helpful_count": 0,
            "harmful_count": 0,
            "ineffective_count": 0,
            "last_effect": "",
            "status": "",
            "parent_node": self._parent_node_id(node),
            "reason": "",
        }
        node = self._canonical_node(level, node)
        if not node:
            return stats
        for row in list((route_payload or {}).get("exploration_history", []) or []) + list((route_payload or {}).get("exploration_slots", []) or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("level", "") or "") != str(level):
                continue
            if self._canonical_node(level, row.get("node", "")) != node:
                continue
            if not self._scope_matches(row.get("scope", ""), scope):
                continue
            for field in ["trial_count", "helpful_count", "harmful_count", "ineffective_count"]:
                try:
                    stats[field] = max(int(stats.get(field, 0) or 0), int(row.get(field, 0) or 0))
                except Exception:
                    pass
            for field in ["last_effect", "status", "parent_node", "reason"]:
                if str(row.get(field, "") or "").strip():
                    stats[field] = row.get(field)
        return stats

    def _place_child_in_order(self, order, child, parent, stats=None):
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

    def _sync_public_tree_children_into_route_skill(self, route_payload, options):
        if not self._tree_trial_exploration_enabled():
            return dict(route_payload or {})
        route = dict(route_payload or {})
        stage = str(getattr(self.args, "run_stage", "test") or "test").strip().lower()
        level_to_bucket = {
            "what": "what_by_why",
            "how": "how_by_what",
            "who": "who_by_how",
        }
        active_by_level = {
            "what": set(str(x) for x in list((options or {}).get("what", []) or [])),
            "how": set(str(x) for x in list((options or {}).get("how", []) or [])),
            "who": set(str(x) for x in list((options or {}).get("who_branches", []) or [])),
        }
        tree = self.public_tree_store.load_tree()
        for level, bucket in level_to_bucket.items():
            table = dict(route.get(bucket, {}) or {})
            if not table:
                continue
            nodes = dict((tree.get(level, {}) or {}))
            for node_id, node in sorted(nodes.items(), key=lambda kv: (self._node_depth(kv[0]), str(kv[0]))):
                node_id = str(node_id or "")
                parent = self._parent_node_id(node_id)
                if not parent or node_id not in active_by_level.get(level, set()):
                    continue
                status = str((node or {}).get("status", "") or "")
                if status == "sprout" and stage != "train":
                    continue
                if status == "active" and (
                    not bool((node or {}).get("global_default", False))
                    or bool((node or {}).get("route_injection_only", False))
                ):
                    continue
                for scope, order in list(table.items()):
                    order = [str(x) for x in list(order or []) if str(x).strip()]
                    if parent not in order:
                        continue
                    stats = self._route_node_trial_stats(route, level, scope, node_id)
                    table[scope] = self._place_child_in_order(order, node_id, parent, stats=stats)
            route[bucket] = table
        return route

    @staticmethod
    def _scope_matches(row_scope, scope):
        row_scope = str(row_scope or "").strip()
        scope = str(scope or "").strip()
        if not row_scope or row_scope == "default":
            return True
        return row_scope == scope or scope.startswith(row_scope) or row_scope.startswith(scope)

    def _candidate_exploration_nodes(self, level, parent_node, active_nodes, route_payload, planning_payload, condition, scope, selected_what=""):
        if not self._tree_trial_exploration_enabled():
            return []
        stage = str(getattr(self.args, "run_stage", "test") or "test").strip().lower()
        parent_node = self._canonical_node(level, parent_node)
        if not parent_node or self._node_depth(parent_node) >= 3:
            return []
        tree = self.public_tree_store.load_tree()
        nodes = dict((tree.get(level, {}) or {}))
        active = [str(x) for x in list(active_nodes or []) if str(x).strip()]
        context_text = self._text_blob(
            scope,
            selected_what,
            condition,
            (route_payload or {}).get("unmapped_task_memory", []),
            (route_payload or {}).get("exploration_history", []),
            (planning_payload or {}).get("tree_need_signals", []),
            (planning_payload or {}).get("open_condition_memory", []),
            (planning_payload or {}).get("repair_rules", []),
        )
        candidates = []
        for node_id in active:
            node_id = self._canonical_node(level, node_id)
            if self._parent_node_id(node_id) != parent_node:
                continue
            if self._node_is_demoted(route_payload, level, scope, node_id):
                continue
            node = dict(nodes.get(node_id, {}) or {})
            status = str(node.get("status", "") or "")
            if status == "sprout" and stage != "train":
                continue
            stats = self._route_node_trial_stats(route_payload, level, scope, node_id)
            helpful = int(stats.get("helpful_count", 0) or 0)
            harmful = int(stats.get("harmful_count", 0) or 0)
            ineffective = int(stats.get("ineffective_count", 0) or 0)
            if harmful > 0 or ineffective > 0:
                continue
            if stage != "train" and helpful <= 0:
                continue
            tokens = self._node_tokens(node_id, node)
            matched = [tok for tok in tokens if tok in context_text]
            lifecycle = dict(node.get("lifecycle", {}) or {})
            public_helpful = int(lifecycle.get("helpful_count", 0) or node.get("helpful_count", 0) or 0)
            public_harmful = int(lifecycle.get("harmful_count", 0) or node.get("harmful_count", 0) or 0)
            priority = 0.0
            source = ""
            if helpful > 0:
                priority = 100.0 + helpful
                source = "user_helpful_child_history"
            elif public_helpful > public_harmful:
                priority = 70.0 + public_helpful
                source = "public_lifecycle_good"
            elif matched:
                priority = 50.0 + min(10, len(matched))
                source = "context_matched_child"
            elif status == "sprout" and stage == "train":
                priority = 20.0
                source = "train_sprout_child_trial"
            if priority <= 0:
                continue
            candidates.append({
                "node": node_id,
                "status": status or "active",
                "source": source,
                "reason": source if not matched else f"{source}: {','.join(matched[:4])}",
                "scope": str(scope or ""),
                "priority": priority,
                "parent_node": parent_node,
            })
        candidates = sorted(
            candidates,
            key=lambda row: (
                -float(row.get("priority", 0.0) or 0.0),
                str(row.get("node", "") or ""),
            ),
        )
        return candidates[:1]

    def _select_child_trial_node(self, level, selected_parent, active_nodes, route_payload, planning_payload, condition, scope, selected_what=""):
        if not self._tree_trial_exploration_enabled():
            return selected_parent, []
        if not selected_parent:
            return selected_parent, []
        selected = self._canonical_node(level, selected_parent)
        all_exploration = []
        seen = {selected}
        while selected and self._node_depth(selected) < 3:
            exploration = self._candidate_exploration_nodes(
                level,
                selected,
                active_nodes,
                route_payload,
                planning_payload,
                condition,
                scope,
                selected_what=selected_what,
            )
            if not exploration:
                break
            next_node = str(exploration[0].get("node", "") or "").strip()
            if not next_node or next_node in seen:
                break
            all_exploration.extend(exploration)
            selected = next_node
            seen.add(selected)
        return selected or selected_parent, all_exploration

    def _ensure_child_downstream_order(self, route_payload, bucket, child_key, parent_key):
        route = dict(route_payload or {})
        child_key = str(child_key or "").strip()
        parent_key = str(parent_key or "").strip()
        if not child_key or not parent_key:
            return route, None
        table = dict(route.get(bucket, {}) or {})
        if child_key in table and list(table.get(child_key, []) or []):
            return route, None
        parent_order = [str(x) for x in list(table.get(parent_key, []) or []) if str(x).strip()]
        if not parent_order:
            return route, None
        table[child_key] = list(parent_order)
        route[bucket] = table
        memory = dict(route.get("child_order_memory", {}) or {})
        key = f"{bucket}|{child_key}"
        row = {
            "inherited_from": parent_key,
            "created_by": "child_route_inheritance",
            "order": list(parent_order),
        }
        memory.setdefault(key, row)
        route["child_order_memory"] = memory
        event = {
            "operation": "record_child_order_inheritance",
            "bucket": bucket,
            "child_scope": child_key,
            "parent_scope": parent_key,
            "inherited_order": list(parent_order),
        }
        return route, event

    def _trial_anchor_node(self, level, node_id, node=None):
        level = str(level or "").strip()
        node_id = self._canonical_node(level, node_id)
        node = dict(node or {}) if isinstance(node, dict) else {}
        tree_nodes = (self.public_tree_store.load_tree().get(level, {}) or {}) if level else {}
        anchor = ""
        if hasattr(self.public_tree_store, "infer_trial_anchor_node"):
            anchor = self.public_tree_store.infer_trial_anchor_node(level, node_id, node, tree_nodes=tree_nodes)
        if not anchor:
            anchor = str(node.get("trial_anchor_node", "") or "").strip()
        return self._canonical_node(level, anchor)

    def _candidate_sibling_parent_trials(
        self,
        level,
        order,
        active_nodes,
        route_payload,
        planning_payload,
        condition,
        task_packet,
        scope,
    ):
        if not self._tree_trial_exploration_enabled():
            return []
        level = str(level or "").strip()
        stage = str(getattr(self.args, "run_stage", "test") or "test").strip().lower()
        if stage != "train" or level not in {'why', "what", "how"}:
            return []
        order = [self._canonical_node(level, x) for x in list(order or []) if str(x).strip()]
        active = set(self._canonical_node(level, x) for x in list(active_nodes or []) if str(x).strip())
        tree = self.public_tree_store.load_tree()
        nodes = dict((tree.get(level, {}) or {}))
        context_text = self._text_blob(
            scope,
            condition,
            task_packet,
            (route_payload or {}).get("unmapped_task_memory", []),
            (route_payload or {}).get("exploration_history", []),
            (planning_payload or {}).get("tree_need_signals", []),
            (planning_payload or {}).get("open_condition_memory", []),
            (planning_payload or {}).get("repair_rules", []),
        )
        candidates = []
        for node_id in sorted(active):
            node_id = self._canonical_node(level, node_id)
            if not node_id or "/" in node_id:
                continue
            node = dict(nodes.get(node_id, {}) or {})
            if str(node.get("status", "") or "") != "sprout":
                continue
            anchor = self._trial_anchor_node(level, node_id, node)
            tokens = self._node_tokens(node_id, node)
            matched = [tok for tok in sorted(tokens) if tok in context_text][:8]
            skip_reason = ""
            if not anchor:
                skip_reason = "missing_trial_anchor_node"
            elif anchor not in order:
                skip_reason = "anchor_not_in_order"
            elif self._node_is_demoted(route_payload, level, scope, node_id):
                skip_reason = "node_demoted_for_scope"
            elif not matched:
                skip_reason = "weak_context_match"
            stats = self._route_node_trial_stats(route_payload, level, scope, node_id)
            harmful = int(stats.get("harmful_count", 0) or 0)
            ineffective = int(stats.get("ineffective_count", 0) or 0)
            if harmful > 0 or ineffective > 0:
                skip_reason = "prior_trial_not_helpful"
            priority = 0.0 if skip_reason else 45.0 + min(10, len(matched))
            try:
                priority += min(5.0, float(node.get("selection_prior", 0.0) or 0.0) * 10.0)
            except Exception:
                pass
            candidates.append(
                {
                    "level": level,
                    "node": node_id,
                    "status": "sprout",
                    "trial_anchor_node": anchor,
                    "trial_anchor_source": str(node.get("trial_anchor_source", "") or ("llm" if str(node.get("trial_anchor_node", "") or "").strip() else "fallback_inferred")),
                    "matched_tokens": matched,
                    "insert_after": anchor if not skip_reason else "",
                    "selected": False,
                    "skip_reason": skip_reason,
                    "scope": str(scope or ""),
                    "priority": priority,
                }
            )
        candidates = sorted(
            candidates,
            key=lambda row: (
                bool(row.get("skip_reason")),
                -float(row.get("priority", 0.0) or 0.0),
                str(row.get("node", "") or ""),
            ),
        )
        return candidates

    def _insert_sibling_parent_trials(self, level, order, candidates):
        if not self._tree_trial_exploration_enabled():
            return [self._canonical_node(level, x) for x in list(order or []) if str(x).strip()], []
        level = str(level or "").strip()
        out = [self._canonical_node(level, x) for x in list(order or []) if str(x).strip()]
        for row in list(candidates or []):
            if str(row.get("skip_reason", "") or ""):
                continue
            node = self._canonical_node(level, row.get("node", ""))
            anchor = self._canonical_node(level, row.get("trial_anchor_node", ""))
            if not node or not anchor or anchor not in out:
                continue
            out = [x for x in out if x != node]
            idx = out.index(anchor) + 1
            out = out[:idx] + [node] + out[idx:]
            row["insert_after"] = anchor
        return out, candidates

    def _select_sibling_parent_trial_node(self, level, selected_anchor, candidates):
        if not self._tree_trial_exploration_enabled():
            return self._canonical_node(level, selected_anchor)
        level = str(level or "").strip()
        selected_anchor = self._canonical_node(level, selected_anchor)
        if not selected_anchor:
            return selected_anchor
        eligible = []
        for row in list(candidates or []):
            if str(row.get("skip_reason", "") or ""):
                continue
            if self._canonical_node(level, row.get("trial_anchor_node", "")) != selected_anchor:
                continue
            node = self._canonical_node(level, row.get("node", ""))
            if node:
                eligible.append((float(row.get("priority", 0.0) or 0.0), node))
        if not eligible:
            return selected_anchor
        return sorted(eligible, key=lambda kv: (-kv[0], kv[1]))[0][1]

    def _ensure_sibling_parent_downstream_order(self, route_payload, level, sibling, anchor):
        if not self._tree_trial_exploration_enabled():
            return dict(route_payload or {}), None
        level = str(level or "").strip()
        bucket = {
            'why': "what_by_why",
            "what": "how_by_what",
            "how": "who_by_how",
        }.get(level, "")
        if not bucket:
            return dict(route_payload or {}), None
        route = dict(route_payload or {})
        sibling = self._canonical_node(level, sibling)
        anchor = self._canonical_node(level, anchor)
        if not sibling or not anchor:
            return route, None
        table = dict(route.get(bucket, {}) or {})
        if sibling in table and list(table.get(sibling, []) or []):
            return route, None
        anchor_order = [str(x) for x in list(table.get(anchor, []) or []) if str(x).strip()]
        if not anchor_order:
            return route, None
        table[sibling] = list(anchor_order)
        route[bucket] = table
        event = {
            "operation": "record_sibling_parent_downstream_inheritance",
            "bucket": bucket,
            "sibling": sibling,
            "anchor": anchor,
            "inherited_order": list(anchor_order),
        }
        return route, event

    def _first_available(self, order, active_nodes, default=""):
        active = set(str(x) for x in list(active_nodes or []))
        for row in list(order or []):
            row = str(row or "")
            if row in active:
                return row
        return str(default or "")

    def _condition_matches(self, row_condition, condition):
        row_condition = dict(row_condition or {}) if isinstance(row_condition, dict) else {}
        condition = dict(condition or {}) if isinstance(condition, dict) else {}
        if not row_condition:
            return True
        for key, expected in row_condition.items():
            if expected in [None, "", [], {}]:
                continue
            if str(key).endswith("_min"):
                base_key = str(key)[:-4]
                try:
                    if float(condition.get(base_key, 0) or 0) < float(expected):
                        return False
                except Exception:
                    return False
                continue
            if str(key).endswith("_max"):
                base_key = str(key)[:-4]
                try:
                    if float(condition.get(base_key, 0) or 0) > float(expected):
                        return False
                except Exception:
                    return False
                continue
            actual = condition.get(key)
            if isinstance(expected, (list, tuple, set)):
                if isinstance(actual, (list, tuple, set)):
                    actual_values = {str(x) for x in actual}
                    if not actual_values.intersection({str(x) for x in expected}):
                        return False
                elif str(actual) not in {str(x) for x in expected}:
                    return False
            elif isinstance(actual, (list, tuple, set)):
                if str(expected) not in {str(x) for x in actual}:
                    return False
            elif isinstance(expected, bool):
                if bool(actual) != expected:
                    return False
            else:
                if key not in condition and str(key) in {
                    "candidate_coverage",
                    "evidence_specificity",
                    "protocol_enforced",
                    "advisor_diversity",
                    "advisor_count",
                }:
                    continue
                if str(actual) != str(expected):
                    return False
        return True

    def _condition_similarity(self, case_condition, condition):
        case_condition = dict(case_condition or {}) if isinstance(case_condition, dict) else {}
        condition = dict(condition or {}) if isinstance(condition, dict) else {}
        if not case_condition:
            return 0.35
        comparable = 0
        matched = 0
        for key in [
            'why',
            "primary_trigger",
            "primary_why",
            "uncertainty_shape",
            "round_type",
            "confidence_band",
            "prior_relation",
            "previous_feedback_exists",
        ]:
            if key not in case_condition:
                continue
            comparable += 1
            case_value = case_condition.get(key)
            cur_value = condition.get(key)
            if isinstance(cur_value, (list, tuple, set)):
                if str(case_value) in {str(x) for x in cur_value}:
                    matched += 1
            elif str(case_value) == str(cur_value):
                matched += 1
        if "matched_why" in case_condition:
            comparable += 1
            old = {str(x) for x in self._as_list(case_condition.get("matched_why"))}
            cur = {str(x) for x in self._as_list(condition.get("matched_why"))}
            if old and cur and old.intersection(cur):
                matched += 1
        if "focus_set_size" in case_condition:
            comparable += 1
            try:
                old = int(case_condition.get("focus_set_size", 0) or 0)
                cur = int(condition.get("focus_set_size", 0) or 0)
                if (old <= 2 and cur <= 2) or (old >= 3 and cur >= 3):
                    matched += 1
            except Exception:
                pass
        return float(matched / max(1, comparable))

    def _path_case_effects(self, planning_payload, condition, level):
        effects = []
        for row in list((planning_payload or {}).get("path_case_memory", []) or []):
            if not isinstance(row, dict):
                continue
            path = self._path_from_case(row)
            node = str(path.get(level, "") or "")
            if not node:
                continue
            sim = self._condition_similarity(
                row.get("condition_signature", row.get("case_signature", {})),
                condition,
            )
            if sim < 0.34:
                continue
            effect = str(row.get("effect", "") or "").lower()
            confidence = self._as_float(row.get("confidence", 0.40), 0.40)
            if effect == "helpful":
                delta = 0.16 * confidence * sim
            elif effect == "harmful":
                delta = -0.30 * confidence * sim
            else:
                delta = 0.02 * confidence * sim
            effects.append(
                {
                    "level": level,
                    "node": node,
                    "delta": delta,
                    "effect": effect or "neutral",
                    "similarity": round(sim, 3),
                    "lesson": str(row.get("lesson", "") or "")[:180],
                    "path_key": str(row.get("path_key", "") or ""),
                }
            )
        return effects

    def _repair_rule_effects(self, planning_payload, condition):
        effects = []
        for row in list((planning_payload or {}).get("repair_rules", []) or []):
            if not isinstance(row, dict):
                continue
            text = self._text_blob(row.get("condition", ""), row.get("action", ""))
            confidence = self._as_float(row.get("confidence", 0.45), 0.45)
            if condition.get("previous_feedback_exists") and "feedback" in text:
                effects.append({"target": "how:multi-cooperative", "delta": 0.05 * confidence, "rule": row})
            if "initial_hit_to_final_miss" in text or "concrete" in text:
                effects.append({"target": "how:multi-competitive", "delta": -0.04 * confidence, "rule": row})
            if "advisor_pool_empty" in text or "fallback" in text or "reroute" in text:
                effects.append({"target": "who:topk-advisors", "delta": 0.03 * confidence, "rule": row})
        return effects

    def _condition(self, decision_state, planning_payload):
        state = dict(decision_state or {})
        payload_condition = dict((planning_payload or {}).get("planning_condition", {}) or {})
        if payload_condition:
            return payload_condition
        confidence = int(state.get("self_confidence", 0) or 0)
        uncertainty = [str(x) for x in self._as_list(state.get("uncertainty_points")) if str(x).strip()]
        primary = str(state.get("primary_trigger", "") or "").strip()
        if any("candidate" in x for x in uncertainty) or self._focus_count(state) >= 2:
            shape = "candidate-conflict"
        elif any("novelty" in x for x in uncertainty) or primary == "novelty-uncertainty":
            shape = "novelty-check"
        elif any("prior" in x for x in uncertainty) or primary == "internal-prior-conflict":
            shape = "internal-prior-conflict"
        else:
            shape = "proposal-risk-check"
        try:
            history_count = int(state.get("history_count", 0) or 0)
        except Exception:
            history_count = 0
        if history_count <= 0:
            history_sparsity = str(state.get("history_sparsity", "") or "")
        elif history_count <= 3:
            history_sparsity = "sparse"
        elif history_count <= 8:
            history_sparsity = "medium"
        else:
            history_sparsity = "rich"
        return {
            "round_type": str(state.get("round_type", "") or "initial"),
            "primary_trigger": primary,
            "uncertainty_shape": shape,
            "confidence_band": "high" if confidence >= 75 else ("medium" if confidence >= 50 else "low"),
            "focus_set_size": self._focus_count(state),
            "prior_relation": "proposal_differs_from_prior" if str(state.get("prior_item", "") or "").strip() and str(state.get("prior_item", "") or "").strip() != str(state.get("proposal_item", "") or "").strip() else "proposal_equals_prior",
            "previous_feedback_exists": bool(state.get("previous_user_feedback")),
            "history_count": history_count,
            "history_sparsity": history_sparsity or "sparse",
        }

    def _first_round_facts(self, decision_state, condition):
        state = dict(decision_state or {})
        condition = dict(condition or {})
        focus_count = int(condition.get("focus_set_size", 0) or 0)
        confidence_band = str(condition.get("confidence_band", "") or "")
        uncertainty_text = self._text_blob(state.get("uncertainty_points", []), condition.get("primary_trigger", ""))
        prior_relation = str(condition.get("prior_relation", "") or "")
        return {
            "round_type": str(condition.get("round_type", "") or "initial"),
            "focus_candidate_count": focus_count,
            "has_comparable_candidates": focus_count >= 2,
            "score_gap_small": confidence_band in {"low", "medium"},
            "history_count": int(condition.get("history_count", 0) or 0),
            "history_sparsity": str(condition.get("history_sparsity", "") or "sparse"),
            "proposal_conflicts_with_prior": prior_relation == "proposal_differs_from_prior",
            "proposal_is_novel": "novelty" in uncertainty_text,
            "candidate_evidence_missing": any(token in uncertainty_text for token in ["missing", "evidence", "coverage", "unanswered"]),
            "legacy_primary_trigger": str(condition.get("primary_trigger", "") or ""),
            "legacy_uncertainty_shape": str(condition.get("uncertainty_shape", "") or ""),
        }

    def _followup_facts(self, task_packet, decision_state):
        packet = dict(task_packet or {})
        state = dict(decision_state or {})
        text = str(packet.get("user_task", "") or state.get("user_task", "") or "").lower()
        compact = re.sub(r"\s+", " ", text)
        shortlist = [str(x).strip() for x in self._as_list(state.get("shortlist", [])) if str(x).strip()]
        explicit_targets = [str(x).strip() for x in self._as_list(packet.get("task_targets", [])) if str(x).strip()]
        mentioned = set(explicit_targets)
        for item in shortlist:
            if item and item.lower() in compact:
                mentioned.add(item)

        def has_any(tokens):
            return any(token in compact for token in tokens)

        asks_compare = has_any([
            "compare",
            "comparison",
            "direct comparison",
            "versus",
            " vs ",
            "better",
            "worse",
            "stronger",
            "weaker",
            "rank",
            "trade-off",
            "which one",
            "challenge",
            "debate",
        ])
        asks_reduce = has_any([
            "reduce",
            "shrink",
            "remove",
            "exclude",
            "eliminate",
            "down-rank",
            "rule out",
            "not fit",
            "risk",
            "warning",
        ])
        asks_verify = has_any([
            "reason",
            "reasoning",
            "assumption",
            "verify",
            "check my",
            "actually fit",
            "surface match",
            "is that true",
        ])
        asks_missing = has_any([
            "missing",
            "evidence gap",
            "silent",
            "coverage",
            "unanswered",
            "explain",
            "details",
            "specific",
            "not covered",
        ])
        asks_positive = has_any([
            "retain",
            "keep",
            "interested",
            "promising",
            "worth considering",
            "positive",
            "support",
            "recommend",
            "still worth",
        ])
        return {
            "has_followup_feedback": bool(str(packet.get("user_task", "") or "").strip()),
            "asks_compare_candidates": asks_compare,
            "asks_reduce_options": asks_reduce,
            "asks_verify_reasoning": asks_verify,
            "asks_missing_evidence": asks_missing,
            "asks_positive_subset": asks_positive,
            "mentioned_candidate_count": len(mentioned),
            "requires_candidate_contrast": bool(asks_compare or len(mentioned) >= 2),
            "requires_evidence_gap_check": bool(asks_missing),
            "requires_reason_validation": bool(asks_verify),
            "feedback_text": str(packet.get("user_task", "") or "")[:500],
            "legacy_task_type_hint": str(packet.get("task_type_hint", "") or ""),
            "legacy_mapped_what": str(packet.get("legacy_mapped_what", "") or packet.get("what", "") or ""),
        }

    @staticmethod
    def _coerce_fact_value(value):
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "false"}:
                return low == "true"
            try:
                if re.fullmatch(r"[-+]?\d+", low):
                    return int(low)
                if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", low):
                    return float(low)
            except Exception:
                pass
            return value.strip()
        return value

    def _evaluate_selection_predicate(self, predicate, facts):
        facts = dict(facts or {})
        original = predicate
        if isinstance(predicate, dict):
            key = str(predicate.get("fact", "") or predicate.get("key", "") or "").strip()
            op = str(predicate.get("op", "") or predicate.get("operator", "") or "==").strip()
            expected = predicate.get("value", predicate.get("expected", True))
        else:
            text = str(predicate or "").strip()
            match = re.match(r"^([A-Za-z0-9_./-]+)\s*(>=|<=|!=|==|=|>|<)\s*(.+)$", text)
            if match:
                key, op, expected = match.group(1), match.group(2), match.group(3)
            else:
                key, op, expected = text, "==", True
        if not key:
            return False, str(original)
        actual = facts.get(key)
        expected = self._coerce_fact_value(expected)
        if op == "=":
            op = "=="
        try:
            if op in {">", ">=", "<", "<="}:
                actual_num = float(actual)
                expected_num = float(expected)
                if op == ">":
                    return actual_num > expected_num, str(original)
                if op == ">=":
                    return actual_num >= expected_num, str(original)
                if op == "<":
                    return actual_num < expected_num, str(original)
                return actual_num <= expected_num, str(original)
            if op == "!=":
                return str(actual).lower() != str(expected).lower(), str(original)
            if isinstance(expected, bool):
                return bool(actual) == expected, str(original)
            return str(actual).lower() == str(expected).lower(), str(original)
        except Exception:
            return False, str(original)

    def _selection_profile(self, node):
        node = dict(node or {})
        profile = dict(node.get("selection_profile", {}) or {})
        source = "skill_json" if profile else "fallback"
        if not profile:
            profile = {
                "requires": [],
                "prefers": [],
                "do_not_use_why": [],
                "selection_prior": node.get("selection_prior", 0.0),
            }
        profile.setdefault("requires", [])
        profile.setdefault("prefers", [])
        profile.setdefault("do_not_use_why", [])
        profile.setdefault("selection_prior", node.get("selection_prior", 0.0))
        return profile, source

    def _score_nodes_by_selection_profile(self, level, facts, candidate_nodes, stage=None):
        stage = str(stage or getattr(self.args, "run_stage", "test") or "test").strip().lower()
        tree_nodes = dict((self.public_tree_store.load_tree().get(level, {}) or {}))
        allowed_status = {"active", "sprout"} if stage == "train" else {"active"}
        rows = []
        for node_id in [self._canonical_node(level, x) for x in list(candidate_nodes or []) if str(x or "").strip()]:
            node = dict(tree_nodes.get(node_id, {}) or {})
            status = str(node.get("status", "") or "")
            if status not in allowed_status:
                continue
            profile, source = self._selection_profile(node)
            failed_requires = []
            matched_prefers = []
            do_not_hits = []
            for pred in self._as_list(profile.get("requires")):
                ok, label = self._evaluate_selection_predicate(pred, facts)
                if not ok:
                    failed_requires.append(label)
            for pred in self._as_list(profile.get("do_not_use_why")):
                ok, label = self._evaluate_selection_predicate(pred, facts)
                if ok:
                    do_not_hits.append(label)
            score = 0.0
            for pred in self._as_list(profile.get("prefers")):
                ok, label = self._evaluate_selection_predicate(pred, facts)
                if ok:
                    matched_prefers.append(label)
                    score += 1.0
            requires_list = self._as_list(profile.get("requires"))
            prefers_list = self._as_list(profile.get("prefers"))
            has_positive_match = bool(requires_list) or bool(matched_prefers) or node_id in {"none", "skip"}
            eligible = not failed_requires and not do_not_hits and has_positive_match
            score += min(0.25, 0.08 * self._as_float(profile.get("selection_prior", node.get("selection_prior", 0.0)), 0.0))
            if status == "sprout":
                score -= 0.05
            rows.append(
                {
                    "level": level,
                    "node": node_id,
                    "status": status,
                    "eligible": bool(eligible),
                    "matched_prefers": matched_prefers,
                    "failed_requires": failed_requires,
                    "do_not_use_hits": do_not_hits,
                    "score": round(float(score), 4),
                    "selected": False,
                    "selection_profile_source": source,
                }
            )
        rows = sorted(rows, key=lambda row: (-float(row.get("score", 0.0) or 0.0), str(row.get("node", "") or "")))
        eligible_rows = [row for row in rows if row.get("eligible")]
        selected = ""
        if eligible_rows:
            selected = str(eligible_rows[0].get("node", "") or "")
            for row in rows:
                row["selected"] = str(row.get("node", "") or "") == selected
        return selected, rows

    def _add_why_match(self, matches, active_why, why, reason, score=0.0):
        why = self._canonical_node('why', why)
        if not why or why not in active_why:
            return
        row = matches.setdefault(why, {"score": 0.0, "reasons": []})
        row["score"] += float(score or 0.0)
        if reason and reason not in row["reasons"]:
            row["reasons"].append(str(reason))

    def _match_whys(self, options, condition, planning_payload, decision_state=None):
        self._last_why_sibling_parent_candidates = []
        self._last_first_round_facts = {}
        self._last_why_selection_candidates = []
        active_why = set(options.get('why', []) or [])
        if not active_why:
            return "", [], []
        facts = self._first_round_facts(decision_state or {}, condition)
        self._last_first_round_facts = dict(facts)
        primary, candidates = self._score_nodes_by_selection_profile('why', facts, list(active_why))
        self._last_why_selection_candidates = list(candidates)
        matches = {}
        for row in candidates:
            if not row.get("eligible"):
                continue
            node = str(row.get("node", "") or "")
            reason = f"selection_profile_v1 score={float(row.get('score', 0.0) or 0.0):.2f}"
            matched = list(row.get("matched_prefers", []) or [])
            if matched:
                reason += f" matched={','.join(matched[:4])}"
            self._add_why_match(matches, active_why, node, reason, float(row.get("score", 0.0) or 0.0))
        if matches:
            base_order = [
                key for key, _ in sorted(matches.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
            ]
            why_sibling_candidates = self._candidate_sibling_parent_trials(
                'why',
                base_order,
                active_why,
                {},
                planning_payload,
                condition,
                {},
                "why_trigger",
            )
            for row in why_sibling_candidates:
                if str(row.get("skip_reason", "") or ""):
                    continue
                node = self._canonical_node('why', row.get("node", ""))
                anchor = self._canonical_node('why', row.get("trial_anchor_node", ""))
                anchor_score = float((matches.get(anchor, {}) or {}).get("score", 0.5) or 0.5)
                self._add_why_match(
                    matches,
                    active_why,
                    node,
                    f"sibling_parent_trial:{node} after {anchor}",
                    max(0.15, anchor_score - 0.05 + 0.02 * len(row.get("matched_tokens", []) or [])),
                )
            self._last_why_sibling_parent_candidates = why_sibling_candidates
        if not matches:
            return "", [], ["no when selection_profile matched first-round facts"]
        primary_why = primary if primary in matches else sorted(matches.items(), key=lambda kv: (-kv[1]["score"], kv[0]))[0][0]
        primary_why = self._select_sibling_parent_trial_node(
            'why',
            primary_why,
            getattr(self, "_last_why_sibling_parent_candidates", []) or [],
        )
        for row in list(getattr(self, "_last_why_selection_candidates", []) or []):
            row["selected"] = str(row.get("node", "") or "") == str(primary_why or "")
        ordered = [primary_why] + [
            key for key, _ in sorted(matches.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
            if key != primary_why
        ]
        reasons = []
        for key in ordered:
            reasons.extend(list(matches.get(key, {}).get("reasons", []) or []))
        return primary_why, ordered, reasons

    def _match_why(self, options, condition, planning_payload):
        primary, _matched, reasons = self._match_whys(options, condition, planning_payload)
        return primary, reasons

    def _path_from_case(self, row):
        row = dict(row or {})
        for key in ["path", "selected_path"]:
            value = row.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def _task_packet(self, decision_state):
        state = dict(decision_state or {})
        packet = dict(state.get("task_packet", {}) or {})
        if not packet:
            packet = {
                "user_task": str(state.get("user_task", "") or ""),
                "task_type_hint": str(state.get("task_type_hint", "") or ""),
                "what": str(state.get("what", "") or ""),
                "task_source": str(state.get("task_source", "") or ""),
                "unmapped_task": bool(state.get("unmapped_task", False)),
            }
        what = str(packet.get("what", "") or "")
        unmapped = bool(packet.get("unmapped_task", False))
        reasons = list(packet.get("mapping_reasons", []) or [])
        if str(packet.get("task_source", "") or "") == "feedback_to_advisors":
            packet.setdefault("legacy_mapped_what", what)
            packet.setdefault("legacy_mapping_reasons", list(reasons))
            packet["what"] = str(packet.get("what", "") or "")
            packet["how"] = str(packet.get("how", "") or "")
            packet["mapping_reasons"] = reasons + ["follow-up what/how will be selected by selection_profile_v1"]
            return packet
        packet["what"] = what
        packet["unmapped_task"] = bool(unmapped)
        packet["mapping_reasons"] = reasons
        return packet

    def _select_who_branch(self, selected_who, options, planning_payload, condition=None, task_packet=None):
        if not self._tree_trial_exploration_enabled():
            return "", [], []
        selected_who = str(selected_who or "")
        tree = self.public_tree_store.load_tree()
        who_nodes = dict((tree.get("who", {}) or {}))
        branches = [
            str(x) for x in list(options.get("who_branches", []) or [])
            if str(x).startswith(selected_who + "/")
        ]
        if not branches:
            return "", [], []
        scores = {branch: 0.02 for branch in branches}
        reasons = {branch: ["available explicit who branch"] for branch in branches}
        context_text = self._text_blob(
            condition,
            task_packet,
            (planning_payload or {}).get("open_condition_memory", []),
            (planning_payload or {}).get("advisor_reliability_candidates", []),
            (planning_payload or {}).get("tree_need_signals", []),
            (planning_payload or {}).get("repair_rules", []),
        )
        branch_debug = {}
        for branch in branches:
            node = dict(who_nodes.get(branch, {}) or {})
            tokens = self._node_tokens(branch, node)
            matched = sorted(tok for tok in tokens if tok in context_text)
            constraints = dict(node.get("retrieval_constraints", {}) or {})
            if not constraints:
                try:
                    constraints = dict(self.public_tree_store.infer_who_subgroup_metadata(branch, node).get("retrieval_constraints", {}) or {})
                except Exception:
                    constraints = {}
            if matched:
                bonus = min(0.12, 0.03 + 0.02 * min(4, len(matched)))
                scores[branch] += bonus
                reasons[branch].append(f"who subgroup context match({','.join(matched[:4])}): +{bonus:.2f}")
            prior = self._as_float(node.get("selection_prior", 0.0), 0.0)
            if prior > 0:
                bonus = min(0.03, 0.05 * prior)
                scores[branch] += bonus
                reasons[branch].append(f"who subgroup selection_prior: +{bonus:.2f}")
            branch_debug[branch] = {
                "level": "who",
                "node": branch,
                "status": str(node.get("status", "") or ""),
                "source": "who_branch_subgroup_trial",
                "matched_tokens": matched[:8],
                "retrieval_constraints": constraints,
                "priority": float(scores.get(branch, 0.0) or 0.0),
                "selected": False,
            }
        for row in list((planning_payload or {}).get("open_condition_memory", []) or []):
            if not isinstance(row, dict):
                continue
            hint = str(row.get("suggested_node_hint", "") or row.get("preferred_who_branch", "") or "")
            if hint and "/" not in hint:
                hint = f"{selected_who}/{hint}"
            if hint in scores:
                bonus = 0.04 * self._as_float(row.get("confidence", 0.35), 0.35)
                scores[hint] += bonus
                reasons[hint].append(f"open condition memory branch hint: +{bonus:.2f}")
        for row in list((planning_payload or {}).get("advisor_reliability_candidates", []) or []):
            hint = str(row.get("who_branch", "") or row.get("trust_subbranch", "") or "")
            if hint and "/" not in hint:
                hint = f"{selected_who}/{hint.replace('|', '-')}"
            if hint in scores:
                helpful = int(row.get("helpful_count", 0) or 0)
                harmful = int(row.get("harmful_count", 0) or 0)
                delta = min(0.12, 0.03 * helpful) - min(0.12, 0.05 * harmful)
                scores[hint] += delta
                reasons[hint].append(f"advisor branch reliability memory: {delta:+.2f}")
        selected = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        rows = []
        for branch in branches:
            row = dict(branch_debug.get(branch, {}) or {})
            row["priority"] = float(scores.get(branch, 0.0) or 0.0)
            row["selected"] = branch == selected
            rows.append(row)
        rows = sorted(rows, key=lambda row: (-float(row.get("priority", 0.0) or 0.0), str(row.get("node", "") or "")))
        return selected, reasons.get(selected, []), rows

    def _plan_path_with_route_skill(
        self,
        decision_state,
        task_packet,
        selected_why,
        matched_why,
        why_reasons,
        condition,
        options,
        slim_user_policy,
        planning_payload,
    ):
        route_payload = self._route_payload(slim_user_policy)
        if not route_payload:
            route_payload = self._default_route_payload(options)
        route_payload = self._sync_public_tree_children_into_route_skill(route_payload, options)
        sibling_parent_candidates = list(getattr(self, "_last_why_sibling_parent_candidates", []) or [])
        sibling_downstream_inheritance = []
        signature = self._trigger_signature(matched_why)
        signature_keys = self._signature_fallbacks(signature)
        why_anchor = ""
        if selected_why and "/" not in str(selected_why):
            why_node = dict((self.public_tree_store.load_tree().get('why', {}) or {}).get(selected_why, {}) or {})
            if str(why_node.get("status", "") or "") == "sprout":
                why_anchor = self._trial_anchor_node('why', selected_why, why_node)
                if why_anchor:
                    route_payload, inherit_event = self._ensure_sibling_parent_downstream_order(
                        route_payload,
                        'why',
                        selected_why,
                        why_anchor,
                    )
                    if inherit_event:
                        sibling_downstream_inheritance.append(inherit_event)
                    signature_keys = [selected_why, why_anchor] + [x for x in signature_keys if x not in {selected_why, why_anchor}]
        what_order, what_scope_key, what_bucket = self._route_order_with_legacy(
            route_payload,
            "what_by_why",
            signature_keys,
            "what_by_signature",
            signature_keys,
            options.get("what", []),
        )
        has_real_user_task = bool(str(task_packet.get("user_task", "") or "").strip())
        is_followup_entry = str(task_packet.get("task_source", "") or "") == "feedback_to_advisors" or bool((decision_state or {}).get("followup_base_path"))
        followup_facts = {}
        what_selection_candidates = []
        what_demotions = []
        what_exploration = []
        child_order_inheritance = []
        if is_followup_entry and has_real_user_task:
            followup_facts = self._followup_facts(task_packet, decision_state)
            route_what_candidates = list(what_order or [])
            selected_what, what_selection_candidates = self._score_nodes_by_selection_profile(
                "what",
                followup_facts,
                route_what_candidates,
            )
            if selected_what:
                what_order = [selected_what] + [x for x in route_what_candidates if x != selected_what]
                what_reason = f"selection_profile_v1 follow-up selected what:{selected_what}"
            else:
                selected_what = "none" if "none" in options.get("what", []) else self._first_available(what_order, options.get("what", []), default="none")
                what_order = [selected_what] + [x for x in route_what_candidates if x != selected_what]
                what_reason = "selection_profile_v1 found no eligible follow-up what; used fallback"
        else:
            what_order, what_demotions = self._apply_route_demotions(what_order, route_payload, "what", signature)
            what_sibling_candidates = self._candidate_sibling_parent_trials(
                "what",
                what_order,
                options.get("what", []),
                route_payload,
                planning_payload,
                condition,
                task_packet,
                signature,
            )
            what_order, what_sibling_candidates = self._insert_sibling_parent_trials("what", what_order, what_sibling_candidates)
            sibling_parent_candidates.extend(what_sibling_candidates)
            selected_what = self._first_available(what_order, options.get("what", []), default="none")
            selected_what = self._select_sibling_parent_trial_node("what", selected_what, what_sibling_candidates)
            selected_what, what_exploration = self._select_child_trial_node(
                "what",
                selected_what,
                options.get("what", []),
                route_payload,
                planning_payload,
                condition,
                signature,
                selected_what=selected_what,
            )
            what_reason = f"route what order:{what_scope_key or 'fallback'} via {what_bucket or 'none'}"
        if not selected_what:
            return None
        if not is_followup_entry:
            task_packet = retarget_first_round_task(
                task_packet,
                selected_what,
                hesitation_set=(decision_state or {}).get("shortlist", []),
                hesitation_evidence=(decision_state or {}).get("hesitation_evidence", []),
            )

        parent_what = self._parent_node_id(selected_what)
        if parent_what:
            route_payload, inherit_event = self._ensure_child_downstream_order(
                route_payload,
                "how_by_what",
                selected_what,
                parent_what,
            )
            if inherit_event:
                child_order_inheritance.append(inherit_event)
        elif selected_what and selected_what not in {"none", "skip"}:
            what_node = dict((self.public_tree_store.load_tree().get("what", {}) or {}).get(selected_what, {}) or {})
            if str(what_node.get("status", "") or "") == "sprout":
                what_anchor = self._trial_anchor_node("what", selected_what, what_node)
                if what_anchor:
                    route_payload, inherit_event = self._ensure_sibling_parent_downstream_order(
                        route_payload,
                        "what",
                        selected_what,
                        what_anchor,
                    )
                    if inherit_event:
                        sibling_downstream_inheritance.append(inherit_event)
        how_scope = parent_what or selected_what or "default"
        how_new_keys = [selected_what]
        if parent_what:
            how_new_keys.append(parent_what)
        how_new_keys.append("default")
        how_old_keys = []
        for sig in signature_keys:
            how_old_keys.append(f"{sig}|{selected_what}")
            if parent_what:
                how_old_keys.append(f"{sig}|{parent_what}")
        how_order, how_scope_key, how_bucket = self._route_order_with_legacy(
            route_payload,
            "how_by_what",
            how_new_keys,
            "how_by_signature_what",
            how_old_keys,
            options.get("how", []),
        )
        if not how_order:
            return None
        how_order, how_demotions = self._apply_route_demotions(how_order, route_payload, "how", how_scope)
        how_sibling_candidates = self._candidate_sibling_parent_trials(
            "how",
            how_order,
            options.get("how", []),
            route_payload,
            planning_payload,
            condition,
            task_packet,
            how_scope,
        )
        how_order, how_sibling_candidates = self._insert_sibling_parent_trials("how", how_order, how_sibling_candidates)
        sibling_parent_candidates.extend(how_sibling_candidates)
        selected_how = self._first_available(how_order, options.get("how", []), default="")
        selected_how = self._select_sibling_parent_trial_node("how", selected_how, how_sibling_candidates)
        selected_how, how_exploration = self._select_child_trial_node(
            "how",
            selected_how,
            options.get("how", []),
            route_payload,
            planning_payload,
            condition,
            how_scope,
            selected_what=selected_what,
        )
        if not selected_how:
            return None

        parent_how = self._parent_node_id(selected_how)
        if parent_how:
            route_payload, inherit_event = self._ensure_child_downstream_order(
                route_payload,
                "who_by_how",
                selected_how,
                parent_how,
            )
            if inherit_event:
                child_order_inheritance.append(inherit_event)
        elif selected_how:
            how_node = dict((self.public_tree_store.load_tree().get("how", {}) or {}).get(selected_how, {}) or {})
            if str(how_node.get("status", "") or "") == "sprout":
                how_anchor = self._trial_anchor_node("how", selected_how, how_node)
                if how_anchor:
                    route_payload, inherit_event = self._ensure_sibling_parent_downstream_order(
                        route_payload,
                        "how",
                        selected_how,
                        how_anchor,
                    )
                    if inherit_event:
                        sibling_downstream_inheritance.append(inherit_event)
        who_scope = parent_how or selected_how or "default"
        who_new_keys = [selected_how]
        if parent_how:
            who_new_keys.append(parent_how)
        who_new_keys.append("default")
        who_old_keys = []
        parent_what_for_who = self._parent_node_id(selected_what)
        for sig in signature_keys:
            who_old_keys.append(f"{sig}|{selected_what}|{selected_how}")
            if parent_how:
                who_old_keys.append(f"{sig}|{selected_what}|{parent_how}")
            if parent_what_for_who:
                who_old_keys.append(f"{sig}|{parent_what_for_who}|{selected_how}")
                if parent_how:
                    who_old_keys.append(f"{sig}|{parent_what_for_who}|{parent_how}")
        who_order, who_scope_key, who_bucket = self._route_order_with_legacy(
            route_payload,
            "who_by_how",
            who_new_keys,
            "who_by_signature_what_how",
            who_old_keys,
            options.get("who_branches", []) if not self._tree_trial_exploration_enabled() else options.get("who", []),
        )
        if not who_order:
            return None
        who_order, who_demotions = self._apply_route_demotions(who_order, route_payload, "who", who_scope)
        selected_who_branch = ""
        if not self._tree_trial_exploration_enabled():
            roots = set(str(x) for x in list(options.get("who", []) or []))
            branches = set(str(x) for x in list(options.get("who_branches", []) or []))
            selected_who = ""
            for row in list(who_order or []):
                row = str(row or "")
                if row in branches and "/" in row:
                    selected_who_branch = row
                    selected_who = self._parent_node_id(row)
                    break
                if row in roots:
                    selected_who = row
                    break
        else:
            selected_who = self._first_available(who_order, options.get("who", []), default="")
        if not selected_who:
            return None
        if selected_who_branch:
            who_branch_reasons = [f"who_branch selected explicitly from communication_route_skill:{selected_who_branch}"]
            who_branch_candidates = [
                {
                    "level": "who",
                    "node": selected_who_branch,
                    "status": "active",
                    "source": "communication_route_skill_explicit_who_branch",
                    "selected": True,
                    "priority": 1.0,
                }
            ]
        else:
            selected_who_branch, who_branch_reasons, who_branch_candidates = self._select_who_branch(
                selected_who,
                options,
                planning_payload,
                condition=condition,
                task_packet=task_packet,
            )
        selected_by_level = {
            'why': selected_why,
            "what": selected_what,
            "how": selected_how,
        }
        for row in sibling_parent_candidates:
            level = str(row.get("level", "") or "")
            row["selected"] = bool(str(row.get("node", "") or "") == str(selected_by_level.get(level, "") or ""))
        route_reason = [
            "planner_source=communication_route_skill",
            f"trigger_signature={signature}",
            what_reason,
            f"route how order:{how_scope_key or 'fallback'} via {how_bucket or 'none'}",
            f"route who order:{who_scope_key or 'fallback'} via {who_bucket or 'none'}",
            *why_reasons,
            *who_branch_reasons[:3],
        ]
        if how_demotions or who_demotions:
            route_reason.append("route demotions applied")
        if 'what_exploration' in locals() and what_exploration:
            route_reason.append("route what exploration node considered")
        if how_exploration:
            route_reason.append("route exploration node considered")
        if child_order_inheritance:
            route_reason.append("child downstream route order inherited from parent")
        if sibling_parent_candidates:
            route_reason.append("sibling parent sprout trials considered")
        if sibling_downstream_inheritance:
            route_reason.append("sibling parent downstream route order inherited from trial anchor")
        path = build_communication_path(
            why=selected_why,
            what=selected_what,
            who=selected_who,
            how=selected_how,
            user_task=str(task_packet.get("user_task", "") or ""),
            task_type_hint=str(task_packet.get("task_type_hint", "") or ""),
            task_targets=task_packet.get("task_targets", []),
            criteria=task_packet.get("criteria", []),
            secondary_what=task_packet.get("secondary_what", []),
            mapping_confidence=str(task_packet.get("mapping_confidence", "") or ""),
            unmapped_parts=task_packet.get("unmapped_parts", []),
            expected_output=str(task_packet.get("expected_output", "") or ""),
            task_source=str(task_packet.get("task_source", "") or ""),
            unmapped_task=bool(task_packet.get("unmapped_task", False)),
            previous_what=str(((decision_state or {}).get("followup_base_path", {}) or {}).get("what", "") or task_packet.get("previous_what", "") or ""),
            previous_how=str(((decision_state or {}).get("followup_base_path", {}) or {}).get("how", "") or task_packet.get("previous_how", "") or ""),
            followup_of_round=int((decision_state or {}).get("followup_of_round", 0) or task_packet.get("followup_of_round", 0) or 0),
            advisor_group_source="reuse_previous_round" if is_followup_entry else "",
            path_reason=route_reason,
            path_score=1.0,
            risk_marks=[],
            trial_flag=False,
            pattern_source="communication_route_skill",
            primary_why=selected_why,
            matched_why=matched_why,
            why_reasons=why_reasons,
        )
        if selected_who_branch:
            path["who_branch"] = selected_who_branch
        if task_packet.get("tree_need_signals"):
            path["tree_need_signals"] = list(task_packet.get("tree_need_signals", []) or [])
        path["trigger_signature"] = signature
        if child_order_inheritance:
            path["child_order_inheritance"] = list(child_order_inheritance)
        if sibling_downstream_inheritance:
            path["sibling_downstream_inheritance"] = list(sibling_downstream_inheritance)
        path["path_skill_payload"] = self.public_tree_store.build_path_skill_payload(path)
        sprout_nodes = []
        for level, payload in dict(path.get("path_skill_payload", {}) or {}).items():
            if isinstance(payload, dict) and str(payload.get("status", "") or "") == "sprout":
                sprout_nodes.append(f"{level}/{payload.get('node_id', '')}")
        if sprout_nodes:
            path["trial_flag"] = True
            path["sprout_nodes"] = sprout_nodes
        path["planner_log"] = {
            "state_signature": dict(condition),
            "route_skill_used": True,
            "template_id": str(route_payload.get("template_id", "") or ""),
            "trigger_signature": signature,
            "first_round_facts": dict(getattr(self, "_last_first_round_facts", {}) or {}),
            "why_selection_mode": "selection_profile_v1",
            "why_selection_candidates": list(getattr(self, "_last_why_selection_candidates", []) or []),
            "followup_facts": dict(followup_facts or {}),
            "what_selection_mode": "selection_profile_v1" if is_followup_entry else "route_skill",
            "what_selection_candidates": list(what_selection_candidates or []),
            "primary_why": selected_why,
            "matched_why": list(matched_why or []),
            "why_reasons": list(why_reasons or []),
            "selected_action": {
                'why': selected_why,
                "matched_why": list(matched_why or []),
                "what": selected_what,
                "how": selected_how,
                "who": selected_who,
            },
            "selected_from_order": {
                "what": list(what_order or []),
                "how": list(how_order or []),
                "who": list(who_order or []),
            },
            "route_scope_keys": {
                "what": what_scope_key,
                "how": how_scope,
                "how_lookup": how_scope_key,
                "who": who_scope,
                "who_lookup": who_scope_key,
            },
            "selected_who_branch": selected_who_branch,
            "who_branch_candidates": list(who_branch_candidates or []),
            "sibling_parent_candidates": list(sibling_parent_candidates or []),
            "sibling_downstream_inheritance": list(sibling_downstream_inheritance or []),
            "demotions_applied": list(locals().get("what_demotions", []) or []) + list(how_demotions or []) + list(who_demotions or []),
            "exploration_nodes_considered": list(what_exploration or []) + list(how_exploration or []),
            "child_order_inheritance": list(child_order_inheritance),
            "fallback_used": False,
            "used_llm_for_path_selection": False,
            "tree_need_signals": list(path.get("tree_need_signals", []) or []),
        }
        return path

    def _plan_path(self, decision_state, slim_user_policy):
        options = self._active_options()
        planning_payload = self._planning_payload(slim_user_policy)
        condition = self._condition(decision_state, planning_payload)
        task_packet = self._task_packet(decision_state)
        base_path = dict((decision_state or {}).get("followup_base_path", {}) or {})
        if str(task_packet.get("task_source", "") or "") == "feedback_to_advisors" and base_path:
            self._last_first_round_facts = {}
            self._last_why_selection_candidates = []
            original_selected_why = str(
                base_path.get("original_selected_why", "")
                or base_path.get('why', "")
                or base_path.get("primary_why", "")
                or ""
            )
            original_matched_why = list(base_path.get("original_matched_why", []) or base_path.get("matched_why", []) or ([original_selected_why] if original_selected_why else []))
            original_why_reasons = list(base_path.get("original_why_reasons", []) or base_path.get("why_reasons", []) or [])
            selected_why = original_selected_why
            matched_why = list(original_matched_why)
            why_reasons = original_why_reasons + ["follow-up reuses previous communication entry when"]
            condition = dict(condition)
            condition["primary_why"] = selected_why
            condition["matched_why"] = list(matched_why or [])
            route_path = self._plan_path_with_route_skill(
                decision_state,
                task_packet,
                selected_why,
                matched_why,
                why_reasons,
                condition,
                options,
                slim_user_policy,
                planning_payload,
            )
            if route_path:
                return route_path
        original_selected_why, original_matched_why, original_why_reasons = self._match_whys(
            options,
            condition,
            planning_payload,
            decision_state=decision_state,
        )
        selected_why = original_selected_why
        matched_why = list(original_matched_why)
        why_reasons = list(original_why_reasons)
        condition = dict(condition)
        condition["primary_why"] = selected_why
        condition["matched_why"] = list(matched_why or [])
        if not selected_why:
            path = build_communication_path(
                why="skip",
                what=str(task_packet.get("what", "none") or "none"),
                who="none",
                how="none",
                user_task=str(task_packet.get("user_task", "") or ""),
                task_type_hint=str(task_packet.get("task_type_hint", "") or ""),
                task_targets=task_packet.get("task_targets", []),
                criteria=task_packet.get("criteria", []),
                secondary_what=task_packet.get("secondary_what", []),
                mapping_confidence=str(task_packet.get("mapping_confidence", "") or ""),
                unmapped_parts=task_packet.get("unmapped_parts", []),
                expected_output=str(task_packet.get("expected_output", "") or ""),
                task_source=str(task_packet.get("task_source", "") or ""),
                unmapped_task=bool(task_packet.get("unmapped_task", False)),
                path_reason=[
                    "planner_source=memory_guided_constrained_planning",
                    *why_reasons,
                ],
                path_score=0.0,
                risk_marks=[],
                trial_flag=False,
                pattern_source="memory_guided_planner",
                primary_why="skip",
                matched_why=[],
                why_reasons=why_reasons,
            )
            path["planner_log"] = {
                "state_signature": dict(condition),
                "matched_why": [],
                "primary_why": "skip",
                "selected_action": {'why': "skip", "matched_why": [], "how": "none", "who": "none"},
                "no_communication_reason": "; ".join(why_reasons),
                "matched_user_skill_policy": [x for x in why_reasons if "matched_user" in str(x)],
                "matched_bootstrap_policy": [x for x in why_reasons if "bootstrap" in str(x)],
                "code_fallback": [x for x in why_reasons if str(x).startswith("derived") or str(x).startswith("matched primary")],
                "path_case_memory_effect": [],
                "repair_rule_effect": [],
                "used_llm_for_path_selection": False,
            }
            return path

        route_path = self._plan_path_with_route_skill(
            decision_state,
            task_packet,
            selected_why,
            matched_why,
            why_reasons,
            condition,
            options,
            slim_user_policy,
            planning_payload,
        )
        if route_path:
            return route_path
        path = build_communication_path(
            why="skip",
            what="none",
            who="none",
            how="none",
            user_task=str(task_packet.get("user_task", "") or ""),
            task_type_hint=str(task_packet.get("task_type_hint", "") or ""),
            task_targets=task_packet.get("task_targets", []),
            criteria=task_packet.get("criteria", []),
            secondary_what=task_packet.get("secondary_what", []),
            mapping_confidence=str(task_packet.get("mapping_confidence", "") or ""),
            unmapped_parts=task_packet.get("unmapped_parts", []),
            expected_output=str(task_packet.get("expected_output", "") or ""),
            task_source=str(task_packet.get("task_source", "") or ""),
            unmapped_task=bool(task_packet.get("unmapped_task", False)),
            path_reason=["planner_source=communication_route_skill", "route resolver failed; skip communication instead of using old score selector"],
            path_score=0.0,
            risk_marks=["route_resolver_failed"],
            trial_flag=False,
            pattern_source="communication_route_skill",
            primary_why="skip",
            matched_why=matched_why,
            why_reasons=why_reasons,
        )
        if task_packet.get("tree_need_signals"):
            path["tree_need_signals"] = list(task_packet.get("tree_need_signals", []) or [])
            path["path_reason"] = list(path.get("path_reason", []) or []) + [
                "unmapped task recorded as what-tree growth signal"
            ]
        path["path_skill_payload"] = self.public_tree_store.build_path_skill_payload(path)
        path["planner_log"] = {
            "state_signature": dict(condition),
            "route_skill_used": True,
            "primary_why": "skip",
            "matched_why": list(matched_why or []),
            "why_reasons": list(why_reasons or []),
            "selected_action": {'why': "skip", "matched_why": list(matched_why or []), "what": "none", "how": "none", "who": "none"},
            "fallback_used": False,
            "old_score_selector_used": False,
            "route_failure": True,
            "tree_need_signals": list(path.get("tree_need_signals", []) or []),
            "used_llm_for_path_selection": False,
        }
        return path

    def _validate_choice(self, choice, decision_state, options):
        choice = dict(choice or {})

        why = self._norm(choice.get('why'))
        what = self._canonical_node("what", choice.get("what"), decision_state=decision_state)
        if not what:
            what = str(self._task_packet(decision_state).get("what", "") or "none")
        who = self._canonical_node("who", choice.get("who"), decision_state=decision_state)
        how = self._canonical_node("how", choice.get("how"), decision_state=decision_state)
        reasons = [str(x) for x in (choice.get("reasons", []) or []) if str(x).strip()]
        validation_notes = []

        if why not in options['why']:
            validation_notes.append(f"invalid why={why}")
        if what not in options["what"]:
            validation_notes.append(f"invalid what={what}")
        if who not in options["who"]:
            validation_notes.append(f"invalid who={who}")
        if how not in options["how"]:
            validation_notes.append(f"invalid how={how}")

        return {
            'why': why,
            "what": what,
            "who": who,
            "who_branch": str(choice.get("who_branch", "") or ""),
            "how": how,
            "legacy_what": what,
            "confidence": float(choice.get("confidence", 0.0) or 0.0),
            "reasons": reasons + validation_notes,
            "valid": len(validation_notes) == 0,
        }

    def _apply_risk_filter(self, choice):
        tree = self.public_tree_store.load_tree()
        risky_paths = dict((tree.get("indexes", {}) or {}).get("risky_paths", {}) or {})
        inactive_branches = {
            key: row
            for key, row in dict((tree.get("indexes", {}) or {}).get("branch_stats", {}) or {}).items()
            if str((row or {}).get("status", "") or "") == "inactive"
        }
        key = self._path_key(choice['why'], choice["who"], choice["how"], choice.get("what", ""))
        risk_marks = []
        if key in risky_paths:
            risk_marks.append(f"risky_path:{key}")
        path_parts = [choice['why'], choice.get("what", ""), choice["who"], choice["how"]]
        for idx, part in enumerate(path_parts, start=1):
            prefix = " -> ".join(path_parts[:idx])
            if prefix in inactive_branches:
                risk_marks.append(f"inactive_branch:{prefix}")
        return risk_marks

    def select(self, decision_state, slim_user_policy, path_choice=None):
        options = self._active_options()
        if not path_choice:
            return self._plan_path(decision_state, slim_user_policy)
        selected_why = self._norm((path_choice or {}).get('why'))
        if selected_why.lower() in ["", "none", "skip", "no", "null", "n/a"]:
            return build_communication_path(
                why="skip",
                who="none",
                how="none",
                what="none",
                path_reason=[
                    "no public-tree why condition matched",
                    'communication action derived as skip from SelectedWhy=none',
                ],
                path_score=0.0,
                risk_marks=[],
                trial_flag=False,
                pattern_source="no_why_condition_matched",
                primary_why="skip",
                matched_why=[],
                why_reasons=["no public-tree why condition matched"],
            )
        choice = self._validate_choice(path_choice, decision_state, options)
        if not choice.get("valid", False):
            return build_communication_path(
                why="unavailable",
                who="none",
                how="none",
                what="none",
                path_reason=[
                    "User Communication Reasoning Skill returned an invalid path",
                    "no hard-coded communication mode was substituted",
                    *choice.get("reasons", []),
                ],
                path_score=0.0,
                risk_marks=["invalid_user_skill_path_choice"],
                trial_flag=False,
                pattern_source="invalid_user_skill",
                primary_why="unavailable",
                matched_why=[],
                why_reasons=["invalid user skill path choice"],
            )
        risk_marks = self._apply_risk_filter(choice)
        reasons = [
            "path_source=user_skill",
            "chosen by User Communication Reasoning Skill and current decision state",
            *choice.get("reasons", []),
        ]
        if risk_marks:
            reasons.append(f"risk marks={risk_marks}")

        path = build_communication_path(
            why=choice['why'],
            what=choice["what"],
            who=choice["who"],
            how=choice["how"],
            user_task=str((path_choice or {}).get("user_task", "") or self._task_packet(decision_state).get("user_task", "")),
            task_type_hint=str((path_choice or {}).get("task_type_hint", "") or self._task_packet(decision_state).get("task_type_hint", "")),
            task_targets=(path_choice or {}).get("task_targets", self._task_packet(decision_state).get("task_targets", [])),
            criteria=(path_choice or {}).get("criteria", self._task_packet(decision_state).get("criteria", [])),
            secondary_what=(path_choice or {}).get("secondary_what", self._task_packet(decision_state).get("secondary_what", [])),
            mapping_confidence=str((path_choice or {}).get("mapping_confidence", "") or self._task_packet(decision_state).get("mapping_confidence", "")),
            unmapped_parts=(path_choice or {}).get("unmapped_parts", self._task_packet(decision_state).get("unmapped_parts", [])),
            expected_output=str((path_choice or {}).get("expected_output", "") or self._task_packet(decision_state).get("expected_output", "")),
            task_source=str((path_choice or {}).get("task_source", "") or self._task_packet(decision_state).get("task_source", "")),
            unmapped_task=bool((path_choice or {}).get("unmapped_task", self._task_packet(decision_state).get("unmapped_task", False))),
            path_reason=reasons,
            path_score=float(choice.get("confidence", 0.0) or 0.0),
            risk_marks=risk_marks,
            trial_flag=False,
            pattern_source="user_skill",
            primary_why=choice['why'],
            matched_why=(path_choice or {}).get("matched_why", [choice['why']]),
            why_reasons=(path_choice or {}).get("why_reasons", []),
        )
        path["path_skill_payload"] = self.public_tree_store.build_path_skill_payload(path)
        return path

    def match_why_set(self, decision_state, slim_user_policy):
        options = self._active_options()
        planning_payload = self._planning_payload(slim_user_policy)
        condition = self._condition(decision_state, planning_payload)
        primary, matched, reasons = self._match_whys(options, condition, planning_payload, decision_state=decision_state)
        condition = dict(condition)
        condition["primary_why"] = primary
        condition["matched_why"] = list(matched or [])
        return {
            "primary_why": primary,
            "matched_why": list(matched or []),
            "why_reasons": list(reasons or []),
            "condition": condition,
        }

