import random
from collections import defaultdict

from recommender.agent.COM.tree_engine.public_tree import infer_communication_shape


class AdvisorPoolManager:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def _advisor_count_for_shape(host, shape, path=None):
        if shape == "single":
            count = 1
            policy = "single_fixed_1"
        else:
            configured_max = max(2, int(getattr(getattr(host, "args", None), "com_max_advisors", 2) or 2))
            upper = max(2, min(int(getattr(host, "advisor_topk", 3) or 3), configured_max))
            count = random.randint(2, upper) if upper > 2 else 2
            policy = f"multi_random_2_to_{upper}"
        if isinstance(path, dict):
            path["advisor_count_target"] = int(count)
            path["advisor_count_policy"] = policy
        return int(count)

    def _profile_from_user(
        self,
        host,
        target_u_int,
        u_int,
        advisor_type,
        target_items,
        focus_item_ids=None,
        hop=1,
        mutual_count=0,
        trust_relation="none",
        trust_group_size=0,
    ):
        items = dict(getattr(host.data, "training_set_u", {}).get(int(u_int), {}) or {})
        item_ids = list(items.keys())
        item_set = set(item_ids)
        sim = host._sasrec_user_similarity(target_u_int, u_int)
        focus_item_ids = [int(x) for x in (focus_item_ids or [])]
        experienced_items = [int(iid) for iid in focus_item_ids if int(iid) in items]
        experience_score = 0.0
        if focus_item_ids:
            experience_score = len(experienced_items) / max(1, len(focus_item_ids))
        reliability = 0.15
        reliability += 0.45 * sim
        reliability += 0.30 * experience_score
        history_similarity_bucket = "sasrec-similar" if sim >= 0.08 else "sasrec-dissimilar"
        if advisor_type == "trusted-advisors":
            reliability += 0.18
            if trust_relation == "mutual-trust":
                reliability += 0.08
            if history_similarity_bucket == "sasrec-similar":
                reliability += 0.06
        if advisor_type == "topk-advisors":
            reliability += 0.05 * max(0, min(3, int(mutual_count)))
        if advisor_type == "topk-advisors":
            reliability -= 0.05 * max(0, int(hop) - 1)
        reliability = max(0.05, min(1.0, reliability))

        top_items = sorted(items.items(), key=lambda kv: float(kv[1]), reverse=True)[:20]
        top_item_names = [host._get_item_name(int(iid)) for iid, _ in top_items]
        history_clues = host._extract_history_clues(", ".join(top_item_names[:12]), max_n=6)
        profile_summary = (
            f"Top items: {', '.join(top_item_names) if top_item_names else 'none'}. "
            f"Preference clues: {', '.join(history_clues) if history_clues else 'mixed interests'}."
        )
        trust_scope = "multi-trust" if int(trust_group_size or 0) >= 2 else "single-trust" if int(trust_group_size or 0) == 1 else "non-trust"
        trust_subbranch = f"{trust_relation}|{trust_scope}|{history_similarity_bucket}" if advisor_type == "trusted-advisors" else ""

        return {
            "u_int": int(u_int),
            "u_raw": str(host._to_raw_id(int(u_int), is_user=True) or ""),
            "advisor_type": str(advisor_type),
            "items": items,
            "item_set": item_set,
            "sim": float(sim),
            "experience_items": list(experienced_items),
            "experience_item_names": [host._get_item_name(int(iid)) for iid in experienced_items],
            "experience_score": float(experience_score),
            "hop": int(hop),
            "mutual_count": int(mutual_count),
            "trust_relation": str(trust_relation or "none"),
            "trust_scope": str(trust_scope),
            "trust_subbranch": str(trust_subbranch),
            "history_similarity_bucket": str(history_similarity_bucket),
            "reliability": float(reliability),
            "top_item_names": list(top_item_names),
            "profile_summary": profile_summary,
            "selection_reason": f"type={advisor_type}; sasrec_cosine={sim:.3f}; experience_score={experience_score:.3f}; hop={hop}; trust_subbranch={trust_subbranch or 'none'}",
        }

    def _direct_friends(self, host, u_raw, u_int):
        out = set()
        if hasattr(host.data, "friend_map_int") and getattr(host.data, "friend_map_int", None):
            out.update(int(x) for x in (host.data.friend_map_int.get(int(u_int), set()) or set()))
        for fr_raw in (host._social_graph_raw.get(str(u_raw), set()) or set()):
            fr_int = host._to_internal_id(fr_raw, is_user=True)
            if fr_int is not None:
                out.add(int(fr_int))
        out.discard(int(u_int))
        return sorted(out)

    def _two_hop_social(self, host, u_raw, u_int):
        direct_raw = set(host._social_graph_raw.get(str(u_raw), set()) or set())
        two_hop_counts = defaultdict(int)
        for mid_raw in direct_raw:
            for cand_raw in (host._social_graph_raw.get(str(mid_raw), set()) or set()):
                if str(cand_raw) == str(u_raw) or cand_raw in direct_raw:
                    continue
                cand_int = host._to_internal_id(cand_raw, is_user=True)
                if cand_int is None:
                    continue
                two_hop_counts[int(cand_int)] += 1
        if hasattr(host.data, "friend_map_int") and getattr(host.data, "friend_map_int", None):
            direct_int = set(int(x) for x in (host.data.friend_map_int.get(int(u_int), set()) or set()))
        else:
            direct_int = set(self._direct_friends(host, u_raw, u_int))
        two_hop_counts.pop(int(u_int), None)
        for fr_int in list(direct_int):
            two_hop_counts.pop(int(fr_int), None)
        return dict(two_hop_counts)

    @staticmethod
    def _infer_branch_constraints(who_branch):
        parts = [part for part in str(who_branch or "").strip().split("/") if part]
        if len(parts) < 2:
            return {}
        tokens = set()
        for part in parts[1:]:
            tokens.update(tok for tok in str(part).lower().replace("_", "-").split("-") if tok)
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

    def _who_branch_metadata(self, path, who):
        path = dict(path or {})
        who_branch = str(path.get("who_branch", "") or "").strip()
        if not who_branch or "/" not in who_branch:
            return {}
        payload = dict((path.get("path_skill_payload", {}) or {}).get("who_branch", {}) or {})
        parts = [part for part in who_branch.split("/") if part]
        advisor_source = str(payload.get("advisor_source", "") or (parts[0] if parts else who) or "")
        if advisor_source != str(who or ""):
            return {}
        constraints = dict(payload.get("retrieval_constraints", {}) or {})
        if not constraints:
            constraints = self._infer_branch_constraints(who_branch)
        return {
            "node_id": who_branch,
            "advisor_source": advisor_source,
            "who_node_kind": str(payload.get("who_node_kind", "") or "source_subgroup"),
            "retrieval_constraints": constraints,
        }

    @staticmethod
    def _profile_matches_constraints(profile, constraints):
        constraints = dict(constraints or {})
        if not constraints:
            return True
        trust_relation = str(constraints.get("trust_relation", "") or "any")
        if trust_relation and trust_relation != "any" and str(profile.get("trust_relation", "") or "") != trust_relation:
            return False
        trust_scope = str(constraints.get("trust_scope", "") or "any")
        if trust_scope and trust_scope != "any" and str(profile.get("trust_scope", "") or "") != trust_scope:
            return False
        history_similarity = str(constraints.get("history_similarity", "") or "any")
        if history_similarity and history_similarity != "any":
            expected = f"history-{history_similarity}" if history_similarity in {"similar", "dissimilar"} else history_similarity
            if str(profile.get("history_similarity_bucket", "") or "") != expected:
                return False
        if bool(constraints.get("requires_item_experience", False)) and float(profile.get("experience_score", 0.0) or 0.0) <= 0:
            return False
        hop = constraints.get("hop", None)
        if hop not in [None, "", "any"] and int(profile.get("hop", 0) or 0) != int(hop):
            return False
        return True

    def _apply_who_branch_constraints(self, profiles, branch_meta, advisor_count, path=None):
        profiles = [dict(row) for row in list(profiles or []) if isinstance(row, dict)]
        branch_meta = dict(branch_meta or {})
        constraints = dict(branch_meta.get("retrieval_constraints", {}) or {})
        branch_id = str(branch_meta.get("node_id", "") or "")
        if not branch_id or not constraints:
            return profiles
        matched = []
        fallback = []
        for prof in profiles:
            if self._profile_matches_constraints(prof, constraints):
                prof = dict(prof)
                prof["who_branch"] = branch_id
                prof["selection_reason"] = f"who_branch={branch_id}; {prof.get('selection_reason', '')}"
                matched.append(prof)
            else:
                fallback.append(prof)
        status = {
            "who_branch": branch_id,
            "advisor_source": str(branch_meta.get("advisor_source", "") or ""),
            "retrieval_constraints": constraints,
            "matched_count": int(len(matched)),
            "fallback_count": int(len(fallback)),
            "fallback_reason": "",
        }
        if len(matched) < int(advisor_count or 1):
            status["fallback_reason"] = "subgroup_pool_empty_or_too_small"
        if isinstance(path, dict):
            path["who_branch_selection_status"] = status
        return matched + fallback

    def _supplement_generic_advisors(
        self,
        host,
        selected,
        selected_users,
        u_int,
        target_items,
        focus_item_ids,
        min_count=2,
        max_count=3,
        path=None,
    ):
        selected = [dict(row) for row in list(selected or []) if isinstance(row, dict)]
        selected_users = set(int(x) for x in (selected_users or set()))
        min_count = max(0, int(min_count or 0))
        max_count = max(min_count, int(max_count or min_count or 0))
        if len(selected) >= min_count:
            return selected
        candidates = []
        for other_u in sorted(getattr(host, "_user_item_sets", {}).keys()):
            other_u = int(other_u)
            if other_u == int(u_int) or other_u in selected_users:
                continue
            prof = self._profile_from_user(
                host,
                u_int,
                other_u,
                "generic-fallback",
                target_items,
                focus_item_ids=focus_item_ids,
                hop=99,
            )
            candidates.append(prof)
        candidates = sorted(
            candidates,
            key=lambda row: (
                -float(row.get("experience_score", 0.0)),
                -float(row.get("reliability", 0.0)),
                -float(row.get("sim", 0.0)),
                int(row.get("u_int", 0)),
            ),
        )
        added = []
        for prof in candidates:
            if len(selected) >= min_count or len(selected) >= max_count:
                break
            uid = int(prof.get("u_int", -1))
            if uid in selected_users:
                continue
            prof = dict(prof)
            prof["advisor_mix_source"] = "generic-fallback"
            prof["advisor_type"] = str(prof.get("advisor_type", "") or "generic-fallback")
            prof["selection_reason"] = f"generic fallback to satisfy multi-advisor minimum; {prof.get('selection_reason', '')}"
            selected.append(prof)
            selected_users.add(uid)
            added.append(str(prof.get("u_raw", "") or uid))
        if added and isinstance(path, dict):
            path["advisor_pool_supplemented"] = True
            path["advisor_pool_supplement_reason"] = "multi_advisor_minimum"
            path["advisor_pool_supplemented_ids"] = list(added)
            path["path_reason"] = list(path.get("path_reason", []) or []) + [
                f"supplemented advisor pool to {len(selected)} speakers for multi-advisor protocol"
            ]
        return selected

    def _trust_relation(self, host, u_raw, u_int, v_int):
        v_raw = str(host._to_raw_id(int(v_int), is_user=True) or "")
        target_raw = str(u_raw or "")
        raw_mutual = bool(target_raw and v_raw and target_raw in set(host._social_graph_raw.get(v_raw, set()) or set()))
        int_mutual = False
        if hasattr(host.data, "friend_map_int") and getattr(host.data, "friend_map_int", None):
            int_mutual = int(u_int) in set(int(x) for x in (host.data.friend_map_int.get(int(v_int), set()) or set()))
        return "mutual-trust" if raw_mutual or int_mutual else "one-way-trust"

    def retrieve(self, host, path, u_raw, u_int, cands_int, proposal_iid, shortlist_names):
        who = str((path or {}).get("who", "") or "")
        how = str((path or {}).get("how", "") or "")
        shape = infer_communication_shape(how)
        what = str((path or {}).get("what", "") or "")
        target_items = set(host._user_item_sets.get(int(u_int), set()) or set())
        shortlist_iids = []
        seen = set()
        for name in shortlist_names or []:
            iid = host._match_name_to_iid(name, cands_int)
            if iid is None or int(iid) in seen:
                continue
            seen.add(int(iid))
            shortlist_iids.append(int(iid))
        focus_item_ids = shortlist_iids[:]
        if proposal_iid is not None and int(proposal_iid) not in seen:
            focus_item_ids = [int(proposal_iid)] + focus_item_ids

        advisor_count = self._advisor_count_for_shape(host, shape, path=path)
        branch_meta = self._who_branch_metadata(path, who)

        def collect_for_who(who_id):
            who_id = str(who_id or "").strip()
            profiles = []
            if who_id == "trusted-advisors":
                direct_ids = self._direct_friends(host, u_raw, u_int)
                for v_int in direct_ids:
                    profiles.append(
                        self._profile_from_user(
                            host,
                            u_int,
                            v_int,
                            who_id,
                            target_items,
                            focus_item_ids=focus_item_ids,
                            hop=1,
                            trust_relation=self._trust_relation(host, u_raw, u_int, v_int),
                            trust_group_size=len(direct_ids),
                        )
                    )

            elif who_id == "similar-users":
                for other_u in sorted(host._user_item_sets.keys()):
                    if int(other_u) == int(u_int):
                        continue
                    prof = self._profile_from_user(host, u_int, other_u, who_id, target_items, focus_item_ids=focus_item_ids, hop=99)
                    if prof["sim"] > 0:
                        profiles.append(prof)

            elif who_id == "experienced-users":
                candidate_users = set()
                training_set_i = getattr(host.data, "training_set_i", {}) or {}
                for iid in focus_item_ids:
                    candidate_users.update(int(x) for x in (training_set_i.get(int(iid), {}) or {}).keys())
                candidate_users.discard(int(u_int))
                for other_u in sorted(candidate_users):
                    profiles.append(self._profile_from_user(host, u_int, other_u, who_id, target_items, focus_item_ids=focus_item_ids, hop=1))

            elif who_id == "topk-advisors":
                two_hop_counts = self._two_hop_social(host, u_raw, u_int)
                for other_u, mutual_count in sorted(two_hop_counts.items()):
                    profiles.append(
                        self._profile_from_user(
                            host,
                            u_int,
                            other_u,
                            who_id,
                            target_items,
                            focus_item_ids=focus_item_ids,
                            hop=2,
                            mutual_count=mutual_count,
                        )
                    )
            profiles = sorted(
                profiles,
                key=lambda row: (-float(row.get("reliability", 0.0)), -float(row.get("experience_score", 0.0)), -float(row.get("sim", 0.0)), int(row.get("u_int", 0))),
            )
            if who_id == "experienced-users" and focus_item_ids:
                selected = []
                selected_users = set()
                for iid in focus_item_ids:
                    for prof in profiles:
                        if int(prof.get("u_int", -1)) in selected_users:
                            continue
                        if int(iid) not in set(int(x) for x in prof.get("experience_items", []) or []):
                            continue
                        prof = dict(prof)
                        prof["selection_reason"] = (
                            f"coverage-first experienced advisor for {host._get_item_name(int(iid))}; "
                            f"{prof.get('selection_reason', '')}"
                        )
                        selected.append(prof)
                        selected_users.add(int(prof.get("u_int", -1)))
                        break
                    if len(selected) >= advisor_count:
                        break
                for prof in profiles:
                    if len(selected) >= advisor_count:
                        break
                    if int(prof.get("u_int", -1)) in selected_users:
                        continue
                    selected.append(prof)
                    selected_users.add(int(prof.get("u_int", -1)))
                profiles = selected + [prof for prof in profiles if int(prof.get("u_int", -1)) not in selected_users]
            if branch_meta and str(branch_meta.get("advisor_source", "") or "") == who_id:
                profiles = self._apply_who_branch_constraints(profiles, branch_meta, advisor_count, path=path)
            return profiles

        if shape == "multi":
            explicit_mix = [str(x) for x in list((path or {}).get("who_mix", []) or []) if str(x).strip()]
            if explicit_mix:
                who_order = explicit_mix
            elif what in {"compare_remaining_candidates", "reduce_hesitation_set", "evidence_gap_check", "reasoning_check"}:
                who_order = [who, "experienced-users", "similar-users", "trusted-advisors", "topk-advisors"]
            else:
                who_order = [who, "similar-users", "experienced-users", "trusted-advisors", "topk-advisors"]
            dedup_order = []
            for source in who_order:
                source = str(source or "").strip()
                if source and source not in dedup_order:
                    dedup_order.append(source)

            source_profiles = {source: collect_for_who(source) for source in dedup_order}
            selected = []
            selected_users = set()
            used_sources = []
            for source in dedup_order:
                for prof in source_profiles.get(source, []):
                    uid = int(prof.get("u_int", -1))
                    if uid in selected_users:
                        continue
                    prof = dict(prof)
                    prof["advisor_mix_source"] = source
                    prof["selection_reason"] = f"mixed-who source={source}; {prof.get('selection_reason', '')}"
                    selected.append(prof)
                    selected_users.add(uid)
                    used_sources.append(source)
                    break
                if len(selected) >= advisor_count:
                    break

            combined = []
            for source in dedup_order:
                for prof in source_profiles.get(source, []):
                    uid = int(prof.get("u_int", -1))
                    if uid in selected_users:
                        continue
                    prof = dict(prof)
                    prof["advisor_mix_source"] = source
                    prof["selection_reason"] = f"mixed-who fill source={source}; {prof.get('selection_reason', '')}"
                    combined.append(prof)
            combined = sorted(
                combined,
                key=lambda row: (-float(row.get("experience_score", 0.0)), -float(row.get("reliability", 0.0)), -float(row.get("sim", 0.0)), int(row.get("u_int", 0))),
            )
            for prof in combined:
                if len(selected) >= advisor_count:
                    break
                selected.append(prof)
                selected_users.add(int(prof.get("u_int", -1)))
            if len(selected) < min(2, advisor_count):
                selected = self._supplement_generic_advisors(
                    host=host,
                    selected=selected,
                    selected_users=selected_users,
                    u_int=u_int,
                    target_items=target_items,
                    focus_item_ids=focus_item_ids,
                    min_count=min(2, advisor_count),
                    max_count=advisor_count,
                    path=path,
                )
            if isinstance(path, dict):
                path["who_mix_used"] = list(used_sources)
                path["who_mix_order"] = list(dedup_order)
            return selected[:advisor_count]

        profiles = []

        if who == "trusted-advisors":
            direct_ids = self._direct_friends(host, u_raw, u_int)
            for v_int in direct_ids:
                profiles.append(
                    self._profile_from_user(
                        host,
                        u_int,
                        v_int,
                        who,
                        target_items,
                        focus_item_ids=focus_item_ids,
                        hop=1,
                        trust_relation=self._trust_relation(host, u_raw, u_int, v_int),
                        trust_group_size=len(direct_ids),
                    )
                )

        elif who == "similar-users":
            for other_u in sorted(host._user_item_sets.keys()):
                if int(other_u) == int(u_int):
                    continue
                prof = self._profile_from_user(host, u_int, other_u, who, target_items, focus_item_ids=focus_item_ids, hop=99)
                if prof["sim"] > 0:
                    profiles.append(prof)

        elif who == "experienced-users":
            candidate_users = set()
            training_set_i = getattr(host.data, "training_set_i", {}) or {}
            for iid in focus_item_ids:
                candidate_users.update(int(x) for x in (training_set_i.get(int(iid), {}) or {}).keys())
            candidate_users.discard(int(u_int))
            for other_u in sorted(candidate_users):
                profiles.append(self._profile_from_user(host, u_int, other_u, who, target_items, focus_item_ids=focus_item_ids, hop=1))

        elif who == "topk-advisors":
            two_hop_counts = self._two_hop_social(host, u_raw, u_int)
            for other_u, mutual_count in sorted(two_hop_counts.items()):
                profiles.append(
                    self._profile_from_user(
                        host,
                        u_int,
                        other_u,
                        who,
                        target_items,
                        focus_item_ids=focus_item_ids,
                        hop=2,
                        mutual_count=mutual_count,
                    )
                )

        profiles = sorted(
            profiles,
            key=lambda row: (-float(row.get("reliability", 0.0)), -float(row.get("experience_score", 0.0)), -float(row.get("sim", 0.0)), int(row.get("u_int", 0))),
        )
        if who == "experienced-users" and focus_item_ids:
            selected = []
            selected_users = set()
            for iid in focus_item_ids:
                for prof in profiles:
                    if int(prof.get("u_int", -1)) in selected_users:
                        continue
                    if int(iid) not in set(int(x) for x in prof.get("experience_items", []) or []):
                        continue
                    prof = dict(prof)
                    prof["selection_reason"] = (
                        f"coverage-first experienced advisor for {host._get_item_name(int(iid))}; "
                        f"{prof.get('selection_reason', '')}"
                    )
                    selected.append(prof)
                    selected_users.add(int(prof.get("u_int", -1)))
                    break
                if len(selected) >= advisor_count:
                    break
            for prof in profiles:
                if len(selected) >= advisor_count:
                    break
                if int(prof.get("u_int", -1)) in selected_users:
                    continue
                selected.append(prof)
                selected_users.add(int(prof.get("u_int", -1)))
            profiles = selected + [prof for prof in profiles if int(prof.get("u_int", -1)) not in selected_users]
        if branch_meta and str(branch_meta.get("advisor_source", "") or "") == who:
            profiles = self._apply_who_branch_constraints(profiles, branch_meta, advisor_count, path=path)
        return profiles[:advisor_count]
