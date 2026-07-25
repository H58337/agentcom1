import json
import math
import os
import random
import re
import threading
import time
import traceback
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

from recommender.agent.agent_sequence_split import candidate_suffix_for_agent_stage, source_for_agent_stage
from recommender.agent.COM.dataset.com_dataset import COMDataset
from recommender.agent.COM.tree_engine import ComTreeEngine
from recommender.agent.COM.tree_engine.utils import load_json
from recommender.agent.COM.utils.com_agent import (
    ComAdvisorAgent,
    ComUserAgent,
    build_com_args,
    get_llm_phase_usage_stats,
    llm_request,
    reset_llm_phase_usage_stats,
)
from recommender.agent.COM.utils.api_request import get_llm_usage_stats, reset_llm_usage_stats
BOT = None
BOT_LOCK = threading.Lock()
BOT_INIT_LOCK = threading.Lock()


def _get_qwen_local_cls():
    from recommender.agent.COM.utils.local_request import QwenLocal

    return QwenLocal

def _reset_llm_usage_tracker():
    try:
        reset_llm_usage_stats()
    except Exception:
        pass
    try:
        reset_llm_phase_usage_stats()
    except Exception:
        pass


def _get_llm_usage_snapshot():
    try:
        stats = get_llm_usage_stats(reset=False)
        if isinstance(stats, dict):
            stats["per_phase"] = get_llm_phase_usage_stats(reset=False)
        return stats
    except Exception:
        return None


class com:
    def __init__(self, args, data):
        print("Recommender: com (Communication-first Agent Paradigm)")
        self.args = args
        self.data = data

        self.tool_model = None
        self.model = None

        self.item_num = int(getattr(self.data, "item_num", len(getattr(self.data, "item", {}))))
        self.state_size = int(getattr(self.args, "MAX_ITEM_LIST_LENGTH", 50))

        top_raw = str(getattr(self.args, "topK", "1"))
        top_list = [int(x) for x in top_raw.split(",") if x.strip().isdigit()]
        self.max_N = max(top_list) if top_list else 10

        self.advisor_topk = int(getattr(self.args, "com_advisor_topk", 3))
        self.advisor_policy = str(getattr(self.args, "com_advisor_policy", "single")).strip().lower()
        if self.advisor_policy not in ["single", "topk"]:
            print(f"[com] Unknown friend selection policy={self.advisor_policy}, fallback to single")
            self.advisor_policy = "single"
        self.debug_round = bool(getattr(self.args, "com_debug_round", False))
        self.debug_user_limit = int(getattr(self.args, "com_debug_user_limit", 5))
        self.save_diagnostics = bool(getattr(self.args, "com_save_diagnostics", False))
        self.save_dialogue = self.save_diagnostics and bool(getattr(self.args, "com_save_dialogue", False))
        self.dialogue_user_limit = int(getattr(self.args, "com_dialogue_user_limit", 0))
        self.dialogue_include_history = bool(getattr(self.args, "com_dialogue_include_history", False))

        self._profile_cache = {}
        self._profile_cache_lock = threading.Lock()
        self._profile_disk_cache = {}
        self._profile_cache_path = str(getattr(self.args, "com_profile_cache_path", "") or "").strip()
        self._profile_cache_write_lock = threading.Lock()
        self._load_profile_cache()

        self._social_graph_raw = defaultdict(set)
        self._user_item_sets = {}
        self._user_history_sequences = defaultdict(list)
        self._sasrec_user_embedding_cache = {}
        self._sasrec_embedding_lock = threading.Lock()
        self._init_friend_resources()
        self.tree_engine = ComTreeEngine(self.args, self.data)
        self.tree_engine.bootstrap(stage="runtime")

        self.agent_results = {}
        self.worker_init(self.args)

    def _is_product_domain(self):
        dataset = str(getattr(self.args, "dataset", "") or "").lower()
        return "epinions" in dataset

    def _is_book_domain(self):
        dataset = str(getattr(self.args, "dataset", "") or "").lower()
        return "librarything" in dataset

    def _domain_terms(self):
        if self._is_product_domain():
            return {
                "dataset_label": "Epinions product recommendation",
                "item_word": "product/item",
                "item_plural": "products/items",
                "history_label": "LongTermProductInteractionHistory",
                "summary_key": "product_preference_summary",
                "evidence_field_meaning": "product/item names copied from history",
                "stat_weak_signal": "product-name weak signals",
                "transferable_categories": (
                    "product category, use case, brand/manufacturer family, feature/function, "
                    "price/value tier, quality/durability, reliability, design/form factor, "
                    "compatibility/accessory relationship, audience/household need, review/rating sentiment, "
                    "popularity/niche level, or substitute/complement bridge"
                ),
                "raw_stat_labels": (
                    "mid-popularity, popular/mainstream item affinity, long-term anchors, recent anchors, "
                    "regional-language naming marker, long-form item/project-name signal, model-number naming signal, "
                    "or statistical history cluster"
                ),
                "minority_clause": "Stable minority product categories/use-cases must be preserved because future target products may come from them.",
                "rules_evidence": (
                    "product category, use case, brand/model family, feature/function, price/value tier, "
                    "quality/durability, reliability, design/form factor, compatibility/accessory bridge, "
                    "review/rating sentiment, recent need drift, and popularity/niche level"
                ),
                "name_signal_examples": "brand/model/series, technical-spec, edition/bundle, numeric model, regional-name, or household-use naming signals",
                "fallback_sparse": (
                    "Long-term product history is sparse; infer product preferences from available items, "
                    "candidate category/use-case/brand/feature evidence, and cautious substitute/complement bridges."
                ),
                "fallback_anchor_intro": "Long-term product anchors",
                "fallback_anchor_tail": (
                    "Use these anchors to infer multiple product preference clusters by category, use case, "
                    "brand/model family, feature/function, price/value, quality/durability, design/form factor, "
                    "and substitute/complement relationships."
                ),
                "default_decision_style": "product-history grounded selection with minority-use-case preservation",
            }
        if self._is_book_domain():
            return {
                "dataset_label": "LibraryThing book recommendation",
                "item_word": "book",
                "item_plural": "books",
                "history_label": "LongTermBookHistory",
                "summary_key": "book_preference_summary",
                "evidence_field_meaning": "book titles/authors copied from reading history",
                "stat_weak_signal": "book-title/author weak signals",
                "transferable_categories": (
                    "fiction/non-fiction genre, literary form, topic/subject, author style, narrative tone, "
                    "era/setting, cultural-language signal, audience/age category, series/franchise relation, "
                    "award/canon/niche level, or adjacent theme bridge"
                ),
                "raw_stat_labels": (
                    "mid-popularity, popular/mainstream book affinity, long-term anchors, recent anchors, "
                    "title-shape/author-name signals, or statistical history cluster"
                ),
                "minority_clause": "Stable minority book genres/topics/author-style clusters must be preserved because future target books may come from them.",
                "rules_evidence": (
                    "book genre, fiction/non-fiction form, topic/subject, author style, narrative tone, "
                    "era/setting, cultural-language signal, audience/age category, series/franchise relation, "
                    "canonical/award/niche level, and recent reading drift"
                ),
                "name_signal_examples": "author/title language, edition/volume/series marker, subtitle/topic phrase, year/number marker, translated-title, or regional/cultural title signal",
                "fallback_sparse": (
                    "Long-term book history is sparse; infer reading preferences from available books, "
                    "candidate genre/topic/author-style evidence, and cautious adjacent-theme bridges."
                ),
                "fallback_anchor_intro": "Long-term book anchors",
                "fallback_anchor_tail": (
                    "Use these anchors to infer multiple reading preference clusters by genre, literary form, "
                    "topic, author style, narrative tone, era/setting, culture/language, and adjacent themes."
                ),
                "default_decision_style": "book-history grounded selection with minority-genre/topic preservation",
            }
        return {
            "dataset_label": "LastFM artist recommendation",
            "item_word": "artist",
            "item_plural": "artists",
            "history_label": "LongTermArtistHistory",
            "summary_key": "music_taste_summary",
            "evidence_field_meaning": "artist copied from history",
            "stat_weak_signal": "artist-name weak signals",
            "transferable_categories": (
                "genre/subgenre, scene, era, mood, vocal or instrumental style, energy, "
                "cultural-language music style, or co-listening style bridge"
            ),
            "raw_stat_labels": (
                "mid-popularity, popular/mainstream artist affinity, long-term anchors, recent anchors, "
                "regional-language naming marker, long-form band/project-name signal, or statistical history cluster"
            ),
            "minority_clause": "Stable minority clusters must be preserved because future target artists may come from them.",
            "rules_evidence": (
                "long-term clusters, recent interaction drift, genre, scene, era, mood, vocal/instrumental style, "
                "energy, cultural/region signals, popularity level, and likely co-listening bridges inferred from history"
            ),
            "name_signal_examples": "regional-script, collaboration/DJ/MC, duo, stylized, or long-form band/project naming signals",
            "fallback_sparse": (
                "Long-term artist history is sparse; infer music taste from available artists, "
                "candidate genre/scene/mood evidence, and cautious novelty bridges."
            ),
            "fallback_anchor_intro": "Long-term artist anchors",
            "fallback_anchor_tail": (
                "Use these anchors to infer multiple music taste clusters by genre, scene, era, mood, "
                "vocal/instrumental style, energy, and likely fan co-listening."
            ),
            "default_decision_style": "history-cluster grounded selection with minority-cluster preservation",
        }

    def _fixed_user_ids(self):
        raw = str(getattr(self.args, "com_fixed_user_ids", "") or "").strip()
        if not raw:
            return []
        if raw.startswith("@"):
            path = raw[1:].strip().strip('"').strip("'")
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = f.read()
                except Exception:
                    return []
        users = []
        for token in re.split(r"[\s,;]+", raw):
            token = token.strip()
            if token and token not in users:
                users.append(token)
        return users

    @staticmethod
    def _user_sort_key(u):
        su = str(u)
        return (0, int(su)) if su.isdigit() else (1, su)

    def _com_train_sample_offset(self):
        try:
            return int(getattr(self.args, "com_train_sample_offset", -1))
        except Exception:
            return -1

    def _com_sample_offset(self, stage_key):
        attr = "com_test_sample_offset" if str(stage_key or "").lower() == "test" else "com_train_sample_offset"
        try:
            return int(getattr(self.args, attr, -1))
        except Exception:
            return -1

    def _com_sample_order(self, stage_key):
        attr = "com_test_sample_order" if str(stage_key or "").lower() == "test" else "com_train_sample_order"
        sample_order = str(getattr(self.args, attr, "random") or "random").strip().lower()
        return "sorted" if sample_order == "sorted" else "random"

    def _select_com_stage_sample(self, entries, sample_num, sample_seed, label, user_key_fn=lambda x: x, stage_key="train"):
        entries = list(entries)
        if sample_num <= 0:
            return entries
        offset = self._com_sample_offset(stage_key)
        total = len(entries)
        if offset >= 0:
            ordered = sorted(entries, key=lambda entry: self._user_sort_key(user_key_fn(entry)))
            sample_order = self._com_sample_order(stage_key)
            if sample_order != "sorted":
                sample_order = "random"
                rng = random.Random(sample_seed)
                rng.shuffle(ordered)
            start = min(offset, total)
            end = min(start + sample_num, total)
            selected = ordered[start:end]
            print(
                f"[com] sampled {label} users: {len(selected)} / {total} "
                f"(order={sample_order}, seed={sample_seed}, offset={offset}, window={start}:{end})"
            )
            return selected
        if sample_num < total:
            rng = random.Random(sample_seed)
            selected = rng.sample(entries, sample_num)
            print(f"[com] sampled {label} users: {len(selected)} / {total} (seed={sample_seed})")
            return selected
        return entries

    def _select_com_train_sample(self, entries, sample_num, sample_seed, label, user_key_fn=lambda x: x):
        return self._select_com_stage_sample(
            entries,
            sample_num=sample_num,
            sample_seed=sample_seed,
            label=label,
            user_key_fn=user_key_fn,
            stage_key="train",
        )

    def _load_failed_user_queue_ids(self):
        path = str(getattr(self.args, "com_failed_user_queue_path", "") or "").strip()
        if not path:
            return []
        if not os.path.exists(path):
            print(f"[com] failed-user queue not found: {path}")
            return []
        users = []
        seen = set()
        read_count = 0
        skip_counts = defaultdict(int)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    read_count += 1
                    try:
                        row = json.loads(line)
                    except Exception:
                        skip_counts["invalid_json"] += 1
                        continue
                    outcome = str(row.get("outcome_signal", "") or "").strip()
                    if outcome not in ["TW", "WW"]:
                        skip_counts["not_final_miss"] += 1
                        continue
                    if not bool(row.get("should_replay", True)):
                        skip_counts["should_replay_false"] += 1
                        continue
                    if not bool(row.get("communication_train_eligible", False)):
                        # Replay is intentionally strict: this queue is for
                        # communication-failed users, not all train failures.
                        skip_counts["not_communication_train_eligible"] += 1
                        continue
                    user_id = str(row.get("user_id", "") or row.get("raw_user_id", "") or "").strip()
                    if not user_id:
                        skip_counts["missing_user_id"] += 1
                        continue
                    if user_id and user_id not in seen:
                        users.append(user_id)
                        seen.add(user_id)
                    elif user_id:
                        skip_counts["duplicate_user_id"] += 1
        except Exception as exc:
            print(f"[com] failed-user queue load failed path={path}: {exc}")
            return []
        print(
            f"[com] loaded communication-failed replay queue: users={len(users)} "
            f"read_rows={read_count} skipped={dict(skip_counts)} path={path}"
        )
        return users

    def _filter_replay_failed_users(self, users, stage_key):
        if str(stage_key or "").lower() != "train":
            return list(users)
        if not bool(getattr(self.args, "com_replay_failed_users_only", False)):
            return list(users)
        replay_users = self._load_failed_user_queue_ids()
        if not replay_users:
            print("[com] replay failed users requested, but queue is empty; no train users selected")
            return []
        available = {str(u) for u in users}
        selected = [u for u in replay_users if str(u) in available]
        missing = [u for u in replay_users if str(u) not in available]
        if missing:
            print(f"[com] replay users missing from train dataset: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        print(f"[com] replay failed train users: {len(selected)} / {len(users)}")
        return selected

    def _filter_fixed_users(self, available_users, stage_key):
        fixed_users = self._fixed_user_ids()
        if not fixed_users:
            return list(available_users)
        available_set = {str(u) for u in available_users}
        selected = [u for u in fixed_users if str(u) in available_set]
        missing = [u for u in fixed_users if str(u) not in available_set]
        if missing:
            print(f"[com] fixed {stage_key} users missing from dataset: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        print(f"[com] fixed {stage_key} users: {len(selected)} / {len(available_users)}")
        return selected

    def _init_friend_resources(self):
        # Build history caches. Similar-user retrieval uses SASRec sequence embeddings;
        # item sets remain available for advisor evidence summaries.
        if hasattr(self.data, "training_set_u") and self.data.training_set_u:
            for u_int, i_dict in self.data.training_set_u.items():
                self._user_item_sets[int(u_int)] = set(i_dict.keys())
        for entry in list(getattr(self.data, "training_data", []) or []):
            if len(entry) < 2:
                continue
            try:
                self._user_history_sequences[int(entry[0])].append(int(entry[1]))
            except (TypeError, ValueError):
                continue

        # Always load the social graph — it is foundational metadata used for
        # communication initial evidence (direct_trust_count, two_hop_count) and
        # user policy bootstrapping.
        if hasattr(self.data, "friend_map_raw") and self.data.friend_map_raw:
            for u_raw, f_set in self.data.friend_map_raw.items():
                for v_raw in f_set:
                    self._social_graph_raw[str(u_raw)].add(str(v_raw))
            print(f"[com] social graph: loaded from friend_id_list users={len(self._social_graph_raw)}")
        else:
            social_path = getattr(self.args, "com_advisor_social_file", None)
            if not social_path:
                dataset_dir = os.path.join(getattr(self.args, "data_path", "data/clean/"), getattr(self.args, "dataset", ""))
                social_name = f"{getattr(self.args, 'dataset', '')}.social"
                social_path = os.path.join(dataset_dir, social_name)

            if not social_path or not os.path.exists(social_path):
                print(f"[com] social graph: social file not found -> {social_path}")
            else:
                loaded = 0
                with open(social_path, "r", encoding="utf-8") as f:
                    header_checked = False
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if not header_checked:
                            header_checked = True
                            low = line.lower()
                            if "user_id" in low and "friend_id" in low:
                                continue

                        parts = line.split("\t")
                        if len(parts) < 2:
                            parts = line.split()
                        if len(parts) < 2:
                            continue

                        u_raw = str(parts[0]).strip()
                        v_raw = str(parts[1]).strip()
                        if not u_raw or not v_raw:
                            continue
                        self._social_graph_raw[u_raw].add(v_raw)
                        loaded += 1
                print(f"[com] social graph: loaded directed social edges={loaded}")

    def _sasrec_user_embedding(self, u_int):
        u_int = int(u_int)
        with self._sasrec_embedding_lock:
            cached = self._sasrec_user_embedding_cache.get(u_int)
        if cached is not None:
            return cached
        if self.tool_model is None or not hasattr(self.tool_model, "encode_user_sequence"):
            raise RuntimeError("SASRec user embeddings are unavailable; initialize the SASRec backbone first")
        sequence = list(self._user_history_sequences.get(u_int, []) or [])[-self.state_size :]
        embedding = self.tool_model.encode_user_sequence(sequence)
        if embedding is None:
            return None
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0:
            return None
        embedding = embedding / norm
        with self._sasrec_embedding_lock:
            self._sasrec_user_embedding_cache[u_int] = embedding
        return embedding

    def _sasrec_user_similarity(self, target_u_int, advisor_u_int):
        target = self._sasrec_user_embedding(target_u_int)
        advisor = self._sasrec_user_embedding(advisor_u_int)
        if target is None or advisor is None:
            return 0.0
        return float(np.clip(np.dot(target, advisor), -1.0, 1.0))

    @staticmethod
    def worker_init(args):
        global BOT
        with BOT_INIT_LOCK:
            if BOT is not None:
                return
            if "Qwen3-8B" in str(getattr(args, "model", "")):
                model_path = getattr(args, "local_model_path", None)
                if model_path:
                    try:
                        BOT = _get_qwen_local_cls()(model_path=model_path)
                        BOT.set_generation_config(max_new_tokens=512, do_sample=False)
                    except Exception as e:
                        print(f"[com] local bot init failed: {e}")
                        BOT = None

    def _to_internal_id(self, raw_id, is_user=True):
        mapping = self.data.user if is_user else self.data.item
        if raw_id is None:
            return None
        val = mapping.get(str(raw_id))
        return int(val) if val is not None else None

    def _to_raw_id(self, internal_id, is_user=True):
        mapping = self.data.id2user if is_user else self.data.id2item
        if internal_id is None:
            return None
        val = mapping.get(int(internal_id))
        return str(val) if val is not None else None

    def _get_item_name(self, iid):
        if iid is None:
            return "Unknown"
        if hasattr(self.data, "get_item_meta"):
            try:
                meta = self.data.get_item_meta(iid, fields=None)
                if isinstance(meta, dict):
                    if meta.get("title"):
                        return str(meta["title"])
                    if meta.get("movie_title"):
                        return str(meta["movie_title"])
            except Exception:
                pass
        return self._to_raw_id(iid, is_user=False) or "Unknown"

    def _name_key(self, s):
        if s is None:
            return ""
        text = unicodedata.normalize("NFKC", str(s)).strip()
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
        text = text.replace('""', '"')
        text = text.replace("_by_", " by ")
        text = text.replace("_", " ")
        text = re.sub(r"[\[\]{}()<>:;,.!?\"'`~|\\/]+", " ", text)
        text = re.sub(r"[-–—]+", " ", text)
        text = re.sub(r"&", " and ", text)
        return " ".join(text.lower().split())

    def _match_name_to_iid(self, name, cands_int):
        key = self._name_key(name)
        if not key:
            return None
        for iid in cands_int:
            nk = self._name_key(self._get_item_name(iid))
            if key == nk or key in nk or nk in key:
                return iid
        return None

    def _load_candidate_json(self, save_dir, suffix=""):
        if suffix == "_val":
            ext_path = str(getattr(self.args, "com_candidates_val_json_path", "") or "").strip()
        else:
            ext_path = str(getattr(self.args, "com_candidates_json_path", "") or "").strip()

        if ext_path:
            path = ext_path
        else:
            file_name = f"candidates{suffix}.json"
            path = os.path.join(str(self.args.save_dir), "SASRec", str(self.args.dataset), file_name)
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        if ext_path:
            raise FileNotFoundError(f"External COM candidates json not found: {path}")
        return self._generate_sasrec_candidates(path=path, suffix=suffix)

    def _generate_sasrec_candidates(self, path, suffix=""):
        if self.tool_model is None or not hasattr(self.tool_model, "recommend_topk_all_items"):
            raise RuntimeError("SASRec backbone is required to generate COM candidate sets")
        stage = "train" if suffix == "_val" else "test"
        source, _ = source_for_agent_stage(self.data, stage)
        candidate_num = max(2, int(getattr(self.args, "sasrec_candidate_num", 20) or 20))
        candidates = {}
        for entry in source:
            u_int, target_iid = int(entry[0]), int(entry[1])
            u_raw = self._to_raw_id(u_int, is_user=True)
            target_raw = self._to_raw_id(target_iid, is_user=False)
            if not u_raw or target_raw is None:
                continue
            history = list(entry[4])[-self.state_size :]
            rows = self.tool_model.recommend_topk_all_items(
                item_sequence=history,
                k=max(candidate_num + 1, candidate_num * 2),
                filter_seen=True,
                exclude_items=None,
                with_scores=False,
                with_names=False,
            )
            negatives = []
            for row in rows or []:
                iid = row.get("item_int") if isinstance(row, dict) else None
                if iid is None:
                    continue
                raw_iid = self._to_raw_id(int(iid), is_user=False)
                if raw_iid is None or str(raw_iid) == str(target_raw) or str(raw_iid) in negatives:
                    continue
                negatives.append(str(raw_iid))
                if len(negatives) >= candidate_num - 1:
                    break
            if len(negatives) != candidate_num - 1:
                raise RuntimeError(
                    f"SASRec returned only {len(negatives)} non-target candidates for user={u_raw}; "
                    f"expected {candidate_num - 1}."
                )
            candidates[str(u_raw)] = [str(target_raw)] + negatives

        if not candidates:
            raise RuntimeError(f"SASRec generated no candidates for stage={stage}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False)
        print(f"[com] generated SASRec candidates -> {path} users={len(candidates)} size={candidate_num}")
        return candidates

    def _load_agent_dataset(self, save_dir, stage="test", suffix=""):
        candidates_map = self._load_candidate_json(save_dir, suffix=suffix)
        ds = COMDataset(
            stage=stage,
            arlib_dataset=self.data,
            candidates_map=candidates_map,
            state_size=self.state_size,
        )
        out = {}
        for sample in ds:
            u_int = sample.get("user_id")
            if u_int is None:
                continue
            u_raw = self._to_raw_id(u_int, is_user=True)
            if u_raw is not None:
                out[str(u_raw)] = sample
        return out

    def _score_candidates(self, seq_int, cands_int):
        if self.tool_model is None:
            raise RuntimeError("SASRec backbone is required to score COM candidates")
        scores = self.tool_model.predict_with_sequence(seq_int, cands_int)
        return {int(iid): float(score) for iid, score in dict(scores or {}).items()}

    @staticmethod
    def _trim_for_memory(text, limit=1000):
        body = " ".join(str(text or "").split())
        if len(body) <= int(limit):
            return body
        return body[: max(0, int(limit) - 3)] + "..."

    def _render_shared_memory(self, memory_window, max_turns=8):
        if not memory_window:
            return "none"
        recent = memory_window[-max(1, int(max_turns)) :]
        lines = []
        for idx, row in enumerate(recent, start=1):
            role = str(row.get("role", "unknown"))
            text = self._trim_for_memory(row.get("text", ""), limit=800)
            lines.append(f"{idx}. [{role}] {text}")
        return "\n".join(lines) if lines else "none"

    def _load_profile_cache(self):
        path = self._profile_cache_path
        if not path:
            return
        if not os.path.exists(path):
            print(f"[com] profile cache not found: {path}")
            return

        loaded = 0
        cache = {}
        try:
            if path.lower().endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        u_raw = str(row.get("user_raw", "")).strip()
                        profile = str(row.get("profile", "")).strip()
                        if u_raw and profile:
                            cache[u_raw] = profile
                            loaded += 1
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for u_raw, profile in data.items():
                        u_raw = str(u_raw).strip()
                        profile = str(profile or "").strip()
                        if u_raw and profile:
                            cache[u_raw] = profile
                            loaded += 1
        except Exception as e:
            print(f"[com] profile cache load failed: {e}")
            return

        self._profile_disk_cache = cache
        print(f"[com] profile cache loaded: {loaded} users")

    def _save_profile_cache(self, u_raw, profile):
        path = self._profile_cache_path
        if not path or not u_raw or not profile:
            return

        payload = {"user_raw": str(u_raw), "profile": str(profile)}

        with self._profile_cache_write_lock:
            if path.lower().endswith(".jsonl"):
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=True) + "\n")
                return

            data = dict(self._profile_disk_cache)
            data[str(u_raw)] = str(profile)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=True)
            os.replace(tmp_path, path)

    def _get_profile_cached(self, cache_key, u_raw, history_str, candidate_names, prior_hint, profile_agent):
        key = str(cache_key or u_raw or "").strip()

        if u_raw:
            cached = self._profile_disk_cache.get(str(u_raw))
            if cached:
                with self._profile_cache_lock:
                    if key:
                        self._profile_cache[key] = cached
                return cached

        if key:
            with self._profile_cache_lock:
                cached = self._profile_cache.get(key)
            if cached:
                return cached

        profile = profile_agent.summarize_profile(history_str, candidate_names, prior_hint)
        profile = str(profile or "").strip()
        if profile:
            with self._profile_cache_lock:
                if key:
                    self._profile_cache[key] = profile
            if u_raw:
                self._profile_disk_cache[str(u_raw)] = profile
                self._save_profile_cache(u_raw, profile)
        return profile

    def _fallback_target_profile(self, history_str):
        terms = self._domain_terms()
        clues = self._extract_history_clues(history_str, max_n=16)
        if not clues:
            return terms["fallback_sparse"]
        return (
            f"{terms['fallback_anchor_intro']}: "
            f"{', '.join(clues[:16])}. "
            f"{terms['fallback_anchor_tail']}"
        )

    def _fallback_candidates_from_tool(self, seq_int, k=20):
        if self.tool_model is None or not seq_int:
            return []
        try:
            rows = self.tool_model.recommend_topk_all_items(
                item_sequence=seq_int,
                k=max(2, int(k)),
                filter_seen=True,
                exclude_items=None,
                with_scores=False,
                with_names=False,
            )
        except Exception as e:
            print(f"[com] fallback candidates failed: {e}")
            return []

        cands = []
        for row in rows or []:
            iid = row.get("item_int") if isinstance(row, dict) else None
            if iid is None:
                continue
            iid = int(iid)
            if iid not in cands:
                cands.append(iid)
        return cands

    def _generate_prior_csv(self, candidates_map, output_dir=None, suffix=""):
        out_dir = os.path.join(output_dir, "prior_csv")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"com_prior{suffix}.csv")

        source = getattr(self.data, "val_data", []) if suffix == "_val" else getattr(self.data, "test_data", [])
        rows = []

        for x in source:
            u_int = int(x[0])
            u_raw = self._to_raw_id(u_int, is_user=True)
            if u_raw is None:
                continue

            seq_int = list(x[4])[-self.state_size :]

            cands_raw = candidates_map.get(str(u_raw), [])
            cands_int = []
            for cr in cands_raw:
                iid = self._to_internal_id(cr, is_user=False)
                if iid is not None:
                    cands_int.append(iid)

            if not cands_int:
                cands_int = self._fallback_candidates_from_tool(seq_int, k=max(20, self.max_N))
            if not cands_int:
                continue

            scores = self._score_candidates(seq_int, cands_int)
            best = max(cands_int, key=lambda i: scores.get(i, -1e18))
            rows.append(
                {
                    "id": str(u_raw),
                    "user_id": str(u_raw),
                    "generate": self._get_item_name(best),
                    "generate_raw_item": self._to_raw_id(best, is_user=False),
                }
            )

        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"[com] saved prior csv {csv_path} rows={len(rows)}")
        else:
            print(f"[com] Warning: empty prior rows for {csv_path}")

    def _resolve_prior_hint_text(self, item_val):
        """Normalize external prior item to a prompt-friendly item name.

        External CSV may store raw item id tokens (e.g., "6217") or numeric-like
        strings (e.g., "6217.0"). This resolves them to item names when possible.
        """
        text = str(item_val or "").strip()
        if not text:
            return ""

        # Keep non-numeric prior text as-is (already likely a readable item name).
        try:
            num = float(text)
            if not np.isfinite(num):
                return text
            if abs(num - round(num)) > 1e-9:
                return text
            token = str(int(round(num)))
        except Exception:
            return text

        # 1) raw item token -> internal id
        iid = self._to_internal_id(token, is_user=False)
        if iid is not None:
            name = self._get_item_name(int(iid))
            return str(name or token)

        # 2) token may already be internal id
        try:
            iid_int = int(token)
            if hasattr(self.data, "id2item") and iid_int in self.data.id2item:
                name = self._get_item_name(iid_int)
                return str(name or token)
        except Exception:
            pass

        return token

    @staticmethod
    def _normalize_csv_id_token(value):
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            num = float(text)
            if np.isfinite(num) and abs(num - round(num)) <= 1e-9:
                return str(int(round(num)))
        except Exception:
            pass
        return text

    def _load_prior_recs(self, save_dir, suffix=""):
        if suffix == "_val":
            ext_path = str(getattr(self.args, "com_prior_val_csv_path", "") or "").strip()
        else:
            ext_path = str(getattr(self.args, "com_prior_csv_path", "") or "").strip()

        path = ext_path or self._infer_external_prior_csv_path(suffix=suffix)
        if not path:
            path = os.path.join(save_dir, "prior_csv", f"com_prior{suffix}.csv") if save_dir else None
        recs = {}
        if not path or not os.path.exists(path):
            return recs
        print(f"[com] loading prior csv for suffix='{suffix}' -> {path}")
        df = pd.read_csv(path)
        user_col = str(getattr(self.args, "com_prior_user_col", "user_id") or "user_id")
        item_col = str(getattr(self.args, "com_prior_item_col", "generate") or "generate")
        converted = 0
        numeric_hint = 0
        raw_item_hint = 0
        for _, row in df.iterrows():
            uid = None
            if user_col in row and pd.notna(row[user_col]):
                uid = self._normalize_csv_id_token(row[user_col])
            elif "user_id" in row and pd.notna(row["user_id"]):
                uid = self._normalize_csv_id_token(row["user_id"])
            elif "id" in row and pd.notna(row["id"]):
                uid = self._normalize_csv_id_token(row["id"])
            if not uid:
                continue

            item_val = ""
            # External priors can provide either raw item ids or readable titles.
            # When the caller uses the default column, prefer the stable raw id
            # if present, then resolve it to the local display name.
            if (
                item_col == "generate"
                and "generate_raw_item" in row
                and pd.notna(row["generate_raw_item"])
                and str(row["generate_raw_item"]).strip()
            ):
                item_val = self._normalize_csv_id_token(row["generate_raw_item"])
                raw_item_hint += 1
            elif item_col in row and pd.notna(row[item_col]):
                item_val = self._normalize_csv_id_token(row[item_col])
            elif "generate" in row and pd.notna(row["generate"]):
                item_val = self._normalize_csv_id_token(row["generate"])
            elif "generate_raw_item" in row and pd.notna(row["generate_raw_item"]):
                item_val = self._normalize_csv_id_token(row["generate_raw_item"])
                raw_item_hint += 1

            raw_text = str(item_val)
            if raw_text:
                try:
                    num = float(raw_text)
                    if np.isfinite(num) and abs(num - round(num)) <= 1e-9:
                        numeric_hint += 1
                except Exception:
                    pass

            resolved = self._resolve_prior_hint_text(raw_text)
            if resolved != raw_text:
                converted += 1
            recs[uid] = resolved

        if numeric_hint > 0 or raw_item_hint > 0:
            print(
                f"[com] prior hint normalization: numeric={numeric_hint}, "
                f"converted_to_name={converted}, raw_item_hints={raw_item_hint}, total={len(recs)}"
            )
        return recs

    def _infer_external_prior_csv_path(self, suffix=""):
        cand_attr = "com_candidates_val_json_path" if suffix == "_val" else "com_candidates_json_path"
        cand_path = str(getattr(self.args, cand_attr, "") or "").strip()
        if not cand_path and suffix == "_val":
            test_cand = str(getattr(self.args, "com_candidates_json_path", "") or "").strip()
            cand_path = os.path.join(os.path.dirname(test_cand), "candidates_val.json") if test_cand else ""
        if not cand_path:
            return ""

        base_dir = os.path.dirname(cand_path)
        names = [f"com_prior{suffix}.csv", f"prior{suffix}.csv", f"prior_top1{suffix}.csv"]
        roots = [
            os.path.join(base_dir, "clean", "prior_csv"),
            os.path.join(base_dir, "clean", "tool", "priorcsv"),
            os.path.join(base_dir, "prior_csv"),
        ]
        for root in roots:
            for name in names:
                path = os.path.join(root, name)
                if os.path.exists(path):
                    return path
        return ""

    @staticmethod
    def _extract_history_clues(history_str, max_n=3):
        text = str(history_str or "")
        # Prefer full item names from history list over generic word tokens.
        raw_parts = re.split(r"[,\n]+", text)
        out = []
        seen = set()
        stop = {"history", "user", "candidates", "none", "true", "false"}

        def is_item_like(val):
            if not val:
                return False
            low = val.lower()
            if low in stop:
                return False
            if len(val) <= 2:
                return False
            # LastFM history is comma-separated artist names; single-token artists
            # such as Muse, Blur, Rihanna, or Oasis are valid item anchors.
            word_count = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", val))
            has_item_marker = ("_" in val) or any(ch.isdigit() for ch in val)
            has_artist_punctuation = any(ch in val for ch in ["'", "&", "-", ".", "!", " "])
            has_upper = any(ch.isupper() for ch in val)
            return word_count >= 1 or has_item_marker or has_artist_punctuation or has_upper

        for part in raw_parts:
            cleaned = " ".join(part.strip().split())
            if not cleaned:
                continue
            if not is_item_like(cleaned):
                continue
            low = cleaned.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(cleaned)
            if len(out) >= int(max_n):
                return out

        # Fallback: extract longer tokens if item-like parts were not found.
        toks = re.findall(r"[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff'&\-]{2,}", text)
        for t in toks:
            low = t.lower()
            if low in seen or low in stop:
                continue
            if len(t) <= 2:
                continue
            seen.add(low)
            out.append(t)
            if len(out) >= int(max_n):
                break
        return out

    def _direct_trust_user_ids_for_policy(self, u_raw, u_int):
        out = set()
        if u_int is not None and hasattr(self.data, "friend_map_int") and getattr(self.data, "friend_map_int", None):
            out.update(int(x) for x in (self.data.friend_map_int.get(int(u_int), set()) or set()))
        for fr_raw in (self._social_graph_raw.get(str(u_raw), set()) or set()):
            fr_int = self._to_internal_id(fr_raw, is_user=True)
            if fr_int is not None:
                out.add(int(fr_int))
        if u_int is not None:
            out.discard(int(u_int))
        return sorted(out)

    def _two_hop_trust_count_for_policy(self, u_raw, u_int, direct_ids=None):
        direct_raw = set(self._social_graph_raw.get(str(u_raw), set()) or set())
        direct_ids = set(int(x) for x in (direct_ids or []))
        two_hop = set()
        for mid_raw in direct_raw:
            for cand_raw in (self._social_graph_raw.get(str(mid_raw), set()) or set()):
                if str(cand_raw) == str(u_raw) or cand_raw in direct_raw:
                    continue
                cand_int = self._to_internal_id(cand_raw, is_user=True)
                if cand_int is None:
                    continue
                if u_int is not None and int(cand_int) == int(u_int):
                    continue
                if int(cand_int) in direct_ids:
                    continue
                two_hop.add(int(cand_int))
        return len(two_hop)

    def _similar_user_stats_for_policy(self, u_int, candidate_items=None, min_sim=0.01):
        if u_int is None:
            return {
                "count": 0,
                "best_sim": 0.0,
                "avg_sim": 0.0,
                "candidate_user_count": 0,
                "candidate_item_count": 0,
                "candidate_coverage": 0.0,
            }
        target = set(self._user_item_sets.get(int(u_int), set()) or set())
        if not target:
            return {
                "count": 0,
                "best_sim": 0.0,
                "avg_sim": 0.0,
                "candidate_user_count": 0,
                "candidate_item_count": 0,
                "candidate_coverage": 0.0,
            }
        candidate_set = set()
        for iid in list(candidate_items or []):
            try:
                candidate_set.add(int(iid))
            except Exception:
                continue
        candidate_items = candidate_set
        count = 0
        best_sim = 0.0
        sim_sum = 0.0
        candidate_users = 0
        covered_candidate_items = set()
        for other_u, other_items in self._user_item_sets.items():
            if int(other_u) == int(u_int):
                continue
            other = set(other_items or set())
            sim = self._sasrec_user_similarity(u_int, other_u)
            if sim <= 0.0:
                continue
            best_sim = max(best_sim, sim)
            if sim >= float(min_sim):
                count += 1
                sim_sum += sim
                if candidate_items:
                    overlap = candidate_items & other
                    if overlap:
                        candidate_users += 1
                        covered_candidate_items.update(overlap)
        avg_sim = float(sim_sum / count) if count > 0 else 0.0
        return {
            "count": int(count),
            "best_sim": float(best_sim),
            "avg_sim": float(avg_sim),
            "candidate_user_count": int(candidate_users),
            "candidate_item_count": int(len(covered_candidate_items)),
            "candidate_coverage": float(len(covered_candidate_items) / max(1, len(candidate_items))) if candidate_items else 0.0,
        }

    def _similar_user_count_for_policy(self, u_int, min_sim=0.01):
        stats = self._similar_user_stats_for_policy(u_int, min_sim=min_sim)
        return int(stats.get("count", 0) or 0), float(stats.get("best_sim", 0.0) or 0.0)

    def _two_hop_trust_user_ids_for_policy(self, u_raw, u_int, direct_ids=None):
        direct_raw = set(self._social_graph_raw.get(str(u_raw), set()) or set())
        direct_ids = set(int(x) for x in (direct_ids or []))
        two_hop = set()
        for mid_raw in direct_raw:
            for cand_raw in (self._social_graph_raw.get(str(mid_raw), set()) or set()):
                if str(cand_raw) == str(u_raw) or cand_raw in direct_raw:
                    continue
                cand_int = self._to_internal_id(cand_raw, is_user=True)
                if cand_int is None:
                    continue
                if u_int is not None and int(cand_int) == int(u_int):
                    continue
                if int(cand_int) in direct_ids:
                    continue
                two_hop.add(int(cand_int))
        return sorted(two_hop)

    def _advisor_candidate_coverage_for_policy(self, advisor_ids, candidate_items):
        candidate_set = set()
        for iid in list(candidate_items or []):
            try:
                candidate_set.add(int(iid))
            except Exception:
                continue
        advisor_id_rows = []
        for advisor_id in list(advisor_ids or []):
            try:
                advisor_id_rows.append(int(advisor_id))
            except Exception:
                continue
        candidate_items = candidate_set
        advisor_ids = advisor_id_rows
        if not candidate_items or not advisor_ids:
            return 0, 0, 0.0
        covered_items = set()
        covered_users = 0
        for advisor_id in advisor_ids:
            items = set(self._user_item_sets.get(int(advisor_id), set()) or set())
            overlap = candidate_items & items
            if overlap:
                covered_users += 1
                covered_items.update(overlap)
        return int(covered_users), int(len(covered_items)), float(len(covered_items) / max(1, len(candidate_items)))

    def _experienced_candidate_coverage_for_policy(self, u_int, candidate_items):
        candidate_rows = []
        for iid in list(candidate_items or []):
            try:
                candidate_rows.append(int(iid))
            except Exception:
                continue
        candidate_items = candidate_rows
        if not candidate_items:
            return 0, 0, 0.0
        training_set_i = getattr(self.data, "training_set_i", {}) or {}
        experienced_users = set()
        covered_items = set()
        for iid in candidate_items:
            users = set(int(x) for x in (training_set_i.get(int(iid), {}) or {}).keys())
            if u_int is not None:
                users.discard(int(u_int))
            if users:
                experienced_users.update(users)
                covered_items.add(int(iid))
        return (
            int(len(experienced_users)),
            int(len(covered_items)),
            float(len(covered_items) / max(1, len(set(candidate_items)))),
        )

    def _build_communication_initial_evidence(self, u_raw, u_int, sample, history_str):
        seq = list((sample or {}).get("seq", []) or [])
        candidate_items = []
        for iid in list((sample or {}).get("cans", []) or []):
            try:
                candidate_items.append(int(iid))
            except Exception:
                continue
        candidate_items = list(dict.fromkeys(candidate_items))
        target_items = set(self._user_item_sets.get(int(u_int), set()) or set()) if u_int is not None else set()
        history_count = int(max(len(seq), len(target_items)))
        direct_ids = self._direct_trust_user_ids_for_policy(u_raw, u_int)
        two_hop_ids = self._two_hop_trust_user_ids_for_policy(u_raw, u_int, direct_ids=direct_ids)
        two_hop_count = len(two_hop_ids)
        similar_stats = self._similar_user_stats_for_policy(u_int, candidate_items=candidate_items)
        similar_count = int(similar_stats.get("count", 0) or 0)
        best_sim = float(similar_stats.get("best_sim", 0.0) or 0.0)
        trusted_user_cover, trusted_item_cover, trusted_coverage = self._advisor_candidate_coverage_for_policy(
            direct_ids,
            candidate_items,
        )
        topk_user_cover, topk_item_cover, topk_coverage = self._advisor_candidate_coverage_for_policy(
            two_hop_ids,
            candidate_items,
        )
        experienced_users, experienced_item_cover, experienced_coverage = self._experienced_candidate_coverage_for_policy(
            u_int,
            candidate_items,
        )
        coverage_rates = [
            trusted_coverage,
            float(similar_stats.get("candidate_coverage", 0.0) or 0.0),
            experienced_coverage,
            topk_coverage,
        ]
        advisor_candidate_diversity = sum(1 for value in coverage_rates if float(value or 0.0) > 0.0)

        # Conservative by design: exploration should not dominate bootstrap unless
        # there is enough history and a usable friend-of-friend pool.
        exploratory_signal = bool(history_count >= 50 and two_hop_count > 0 and similar_count <= 3)
        return {
            "direct_trust_count": int(len(direct_ids)),
            "two_hop_count": int(two_hop_count),
            "similar_user_count": int(similar_count),
            "experienced_user_count": int(experienced_users),
            "candidate_count": int(len(candidate_items)),
            "best_similar_user_score": float(best_sim),
            "avg_similar_user_score": float(similar_stats.get("avg_sim", 0.0) or 0.0),
            "trusted_candidate_user_count": int(trusted_user_cover),
            "trusted_candidate_item_count": int(trusted_item_cover),
            "trusted_candidate_coverage": float(trusted_coverage),
            "similar_candidate_user_count": int(similar_stats.get("candidate_user_count", 0) or 0),
            "similar_candidate_item_count": int(similar_stats.get("candidate_item_count", 0) or 0),
            "similar_candidate_coverage": float(similar_stats.get("candidate_coverage", 0.0) or 0.0),
            "candidate_experienced_user_count": int(experienced_users),
            "experienced_candidate_item_count": int(experienced_item_cover),
            "experienced_candidate_coverage": float(experienced_coverage),
            "topk_candidate_user_count": int(topk_user_cover),
            "topk_candidate_item_count": int(topk_item_cover),
            "topk_candidate_coverage": float(topk_coverage),
            "advisor_candidate_diversity": int(advisor_candidate_diversity),
            "history_count": int(history_count),
            "history_rich": bool(history_count >= 20),
            "history_sparse": bool(history_count < 20),
            "social_sparse": bool(len(direct_ids) <= 0),
            "exploratory_signal": bool(exploratory_signal),
            "evidence_source": "social_graph_history_and_candidate_coverage",
        }

    @staticmethod
    def _extract_json_object(text):
        raw = str(text or "").strip()
        if not raw:
            return None
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return None

    def _item_popularity_counts(self):
        counts = defaultdict(int)
        for items in self._user_item_sets.values():
            for iid in set(items or []):
                counts[int(iid)] += 1
        return counts

    def _item_name_signal(self, name):
        name = str(name or "").strip()
        low = name.lower()
        signals = []
        if self._is_product_domain():
            if re.search(r"[^\x00-\x7f]", name):
                signals.append("regional or non-English product-name signal")
            if re.search(r"\d", name):
                signals.append("model-number or technical-spec product-name signal")
            if any(token in low for token in [" pro", "mini", "plus", "max", "deluxe", "premium", "kit", "set", "bundle", "edition"]):
                signals.append("edition/bundle/value-tier product-name signal")
            if "&" in name or "+" in name or "/" in name:
                signals.append("bundle or multi-function product-name signal")
            if len(name.split()) >= 4:
                signals.append("long-form descriptive product-name signal")
        elif self._is_book_domain():
            if re.search(r"[^\x00-\x7f]", name):
                signals.append("translated or non-English book-title signal")
            if re.search(r"\d", name):
                signals.append("year/volume/edition book-title signal")
            if any(token in low for token in ["volume", "vol.", " book ", "series", "edition", "novel", "memoir", "biography", "history", "guide", "letters", "diary"]):
                signals.append("form/topic book-title signal")
            if ":" in name:
                signals.append("subtitle/topic-descriptor book-title signal")
            if "_by_" in low or " by " in low:
                signals.append("author-attributed book-title signal")
            if len(name.split()) >= 5:
                signals.append("long-form descriptive book-title signal")
        else:
            if re.search(r"[^\x00-\x7f]", name):
                signals.append("non-English or regional-script artist-name signal")
            if any(token in low for token in [" dj ", "dj ", "mc ", " feat", " ft.", " vs "]):
                signals.append("collaboration/DJ/MC electronic or hip-hop naming signal")
            if "&" in name or "+" in name:
                signals.append("duo/collaboration artist signal")
            if re.search(r"\b(the|les|los|la|el|de|del|da|die|der)\b", low):
                signals.append("regional-language naming marker")
            if re.search(r"\d", name):
                signals.append("stylized modern artist-name signal")
            if len(name.split()) >= 3:
                signals.append("long-form band or project-name signal")
        return signals

    def _build_stat_initial_core_skill(self, u_raw, sample, history_str, target_profile):
        terms = self._domain_terms()
        history_artists = self._extract_history_clues(history_str, max_n=120)
        seq_ids = [int(x) for x in list((sample or {}).get("seq", []) or []) if x is not None]
        recent_ids = seq_ids[-12:]
        recent_artists = [self._get_item_name(iid) for iid in recent_ids if self._get_item_name(iid) != "Unknown"]
        pop_counts = self._item_popularity_counts()
        if self._is_product_domain():
            long_anchor_cluster = "long-term repeated product anchors"
            recent_anchor_cluster = "recent product-need drift anchors"
            popularity_affinity = "product affinity"
        elif self._is_book_domain():
            long_anchor_cluster = "long-term repeated book anchors"
            recent_anchor_cluster = "recent reading drift anchors"
            popularity_affinity = "book affinity"
        else:
            long_anchor_cluster = "long-term repeated listening anchors"
            recent_anchor_cluster = "recent listening drift anchors"
            popularity_affinity = "artist affinity"

        def pop_bucket(name, iid=None):
            count = int(pop_counts.get(int(iid), 0) or 0) if iid is not None else 0
            if count >= 50:
                return "popular/mainstream"
            if count <= 5:
                return "niche/low-popularity"
            return "mid-popularity"

        named_ids = []
        for iid in seq_ids[-80:]:
            name = self._get_item_name(iid)
            if name and name != "Unknown":
                named_ids.append((iid, name))
        if not named_ids:
            named_ids = [(None, name) for name in history_artists[:80]]

        signal_groups = defaultdict(list)
        popularity_groups = defaultdict(list)
        for iid, name in named_ids:
            popularity_groups[pop_bucket(name, iid)].append(name)
            for sig in self._item_name_signal(name):
                signal_groups[sig].append(name)

        clusters = []
        if history_artists:
            clusters.append(
                {
                    "cluster": long_anchor_cluster,
                    "evidence_artists": history_artists[:10],
                    "ranking_rule": f"Treat candidate {terms['item_plural']} connected to these recurring history anchors as positive evidence for this user.",
                    "confidence": 0.58,
                    "source": "stat_init",
                }
            )
        if recent_artists:
            clusters.append(
                {
                    "cluster": recent_anchor_cluster,
                    "evidence_artists": recent_artists[-8:],
                    "ranking_rule": "Use recent anchors as short-term positive signals when they bridge back to stable history clusters.",
                    "confidence": 0.48,
                    "source": "stat_init_recent",
                }
            )
        for bucket, artists in sorted(popularity_groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:2]:
            unique = []
            seen = set()
            for artist in artists:
                key = artist.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(artist)
            if len(unique) >= 3:
                clusters.append(
                    {
                        "cluster": f"{bucket} {popularity_affinity}",
                        "evidence_artists": unique[:8],
                        "ranking_rule": f"Preserve candidates with a {bucket} profile when they connect to user history rather than using popularity alone.",
                        "confidence": 0.50,
                        "source": "stat_init_popularity",
                    }
                )
        for sig, artists in sorted(signal_groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]:
            unique = []
            seen = set()
            for artist in artists:
                key = artist.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(artist)
            if len(unique) >= 2:
                clusters.append(
                    {
                        "cluster": sig,
                        "evidence_artists": unique[:8],
                        "ranking_rule": "Keep this weak name/region/style signal as low-confidence positive evidence when it recurs in history.",
                        "confidence": 0.42,
                        "source": "stat_init_weak_name_signal",
                    }
                )

        clusters = clusters[:6]
        preferences = [
            {
                "attribute": row["cluster"],
                "confidence": row.get("confidence", 0.50),
                "source": row.get("source", "stat_init"),
                "evidence_artists": row.get("evidence_artists", []),
                "evidence": row.get("ranking_rule", ""),
            }
            for row in clusters
        ]
        recent_signals = [
            {
                "attribute": recent_anchor_cluster,
                "confidence": 0.46,
                "source": "stat_init_recent",
                "evidence_artists": recent_artists[-8:],
                "evidence": "Recent history should be treated as a positive but bridged signal.",
            }
        ] if recent_artists else []
        item_rules = [
            self.tree_engine.user_policy_store._rule(
                f"Prefer candidates matching the user's transferable preference signal: {row['cluster']}. Evidence {terms['item_plural']}: {', '.join(row.get('evidence_artists', [])[:5])}.",
                float(row.get("confidence", 0.50) or 0.50),
            )
            for row in clusters[:5]
        ]
        if not item_rules:
            item_rules = self.tree_engine.user_policy_store._initial_core_rules(
                history_summary=history_str,
                target_profile=target_profile,
            )
        long_term = [row["cluster"] for row in clusters[:5]]
        return {
            "core_rules": item_rules,
            "core_preference": {
                "long_term_preference": long_term,
                "taste_clusters": clusters,
                "preferences": preferences,
                "recent_signals": recent_signals,
                "decision_style": "statistical history clusters first, with stable minority and recent signals preserved",
                "novelty_tolerance": 0.45,
                "stability_preference": 0.62 if len(history_artists) >= 30 else 0.50,
            },
            "core_initial_evidence": {
                "source": "stat_init_history_cooccurrence_popularity_name_signals",
                "history_artist_count": int(len(history_artists)),
                "recent_artist_count": int(len(recent_artists)),
                "history_item_count": int(len(history_artists)),
                "recent_item_count": int(len(recent_artists)),
                "taste_clusters": clusters,
            },
        }

    def _merge_llm_label_payload(self, stat_payload, llm_payload):
        if not llm_payload:
            evidence = dict((stat_payload or {}).get("core_initial_evidence", {}) or {})
            return {
                "core_rules": [],
                "core_preference": {
                    "decision_style": "LLM initialization unavailable; rely on direct history evidence at decision time",
                    "novelty_tolerance": 0.50,
                    "stability_preference": 0.50,
                },
                "core_initial_evidence": {
                    "source": "llm_initial_core_skill_failed_no_stat_skill_written",
                    "stat_init_was_prompt_only": True,
                    "history_artist_count": int(evidence.get("history_artist_count", 0) or 0),
                    "recent_artist_count": int(evidence.get("recent_artist_count", 0) or 0),
                    "history_item_count": int(evidence.get("history_item_count", evidence.get("history_artist_count", 0)) or 0),
                    "recent_item_count": int(evidence.get("recent_item_count", evidence.get("recent_artist_count", 0)) or 0),
                },
            }
        merged = dict(llm_payload or {})
        llm_pref = dict(merged.get("core_preference", {}) or {})
        filtered_pref = {}
        for key in [
            "taste_clusters",
            "preferences",
            "long_term_preference",
            "recent_signals",
            "decision_style",
            "novelty_tolerance",
            "stability_preference",
        ]:
            if key in llm_pref:
                filtered_pref[key] = llm_pref.get(key)
        merged["core_preference"] = filtered_pref
        evidence = dict(merged.get("core_initial_evidence", {}) or {})
        stat_evidence = dict((stat_payload or {}).get("core_initial_evidence", {}) or {})
        evidence["stat_init_was_prompt_only"] = True
        evidence["history_artist_count"] = int(stat_evidence.get("history_artist_count", 0) or 0)
        evidence["recent_artist_count"] = int(stat_evidence.get("recent_artist_count", 0) or 0)
        evidence["history_item_count"] = int(stat_evidence.get("history_item_count", stat_evidence.get("history_artist_count", 0)) or 0)
        evidence["recent_item_count"] = int(stat_evidence.get("recent_item_count", stat_evidence.get("recent_artist_count", 0)) or 0)
        evidence["source"] = "llm_initial_core_skill_from_stat_evidence"
        merged["core_initial_evidence"] = evidence
        return merged

    def _sanitize_llm_core_skill_payload(self, payload, target_profile):
        obj = dict(payload or {})

        def clean_text(value):
            text = unicodedata.normalize("NFKC", str(value or "")).strip()
            text = text.replace(".ェ", "; ").replace("。ェ", "; ").replace("｡ｪ", "; ").replace("｡", ".").replace("\u2014", "; ")
            text = re.sub(r"\s+", " ", text)
            return text

        raw_rules = obj.get("item_reasoning_rules")
        if not isinstance(raw_rules, list):
            raw_rules = obj.get("core_rules")
        if not isinstance(raw_rules, list):
            raw_rules = obj.get("rules")
        if not isinstance(raw_rules, list):
            raw_rules = []
        rules = []
        seen = set()
        for row in raw_rules:
            if isinstance(row, dict):
                rule_text = clean_text(row.get("rule", ""))
                conf = row.get("confidence", 0.0)
            else:
                rule_text = clean_text(row)
                conf = 0.62
            if not rule_text:
                continue
            if self.tree_engine.user_policy_store._is_generic_item_protocol_rule(rule_text):
                continue
            key = " ".join(rule_text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.62
            if conf > 1.0:
                conf = conf / 100.0
            conf = max(0.30, min(0.85, conf))
            rules.append(self.tree_engine.user_policy_store._rule(rule_text, conf))
            if len(rules) >= 6:
                break
        pref_obj = dict(obj.get("core_preference", {}) or {})
        terms = self._domain_terms()
        summary = clean_text(obj.get(terms["summary_key"], "") or obj.get("summary", "") or "")
        if not summary:
            summary = clean_text(
                obj.get("book_preference_summary", "")
                or obj.get("product_preference_summary", "")
                or obj.get("music_taste_summary", "")
                or ""
            )

        def is_bad_skill_label(text):
            if not text:
                return True
            low = " ".join(str(text or "").strip().lower().split())
            if self._is_book_domain() and any(
                marker in low
                for marker in [
                    "music", "artist", "artists", "band", "bands", "album", "albums", "song", "songs",
                    "listening", "vocal", "instrumental", "dj", "hip-hop", "hip hop", "rock", "metal",
                ]
            ):
                return True
            store = self.tree_engine.user_policy_store
            return (
                store._is_generic_item_protocol_rule(text)
                or store._is_rule_like_preference_attribute(text)
                or store._is_weak_meta_preference_attribute(text)
            )

        def bounded_float(value, default):
            try:
                return max(0.0, min(1.0, float(value)))
            except Exception:
                return float(default)

        taste_clusters = obj.get("taste_clusters")
        if not isinstance(taste_clusters, list):
            taste_clusters = []
        clean_clusters = []
        for row in taste_clusters:
            row = dict(row or {}) if isinstance(row, dict) else {"cluster": str(row)}
            cluster = clean_text(row.get("cluster", "") or row.get("name", ""))
            if is_bad_skill_label(cluster):
                continue
            evidence_artists = row.get("evidence_artists", [])
            if not isinstance(evidence_artists, list):
                evidence_artists = [evidence_artists]
            rule = clean_text(row.get("ranking_rule", "") or row.get("rule", ""))
            if cluster:
                clean_clusters.append(
                    {
                        "cluster": cluster,
                        "confidence": bounded_float(row.get("confidence", 0.56), 0.56),
                        "source": "llm_initial_style_cluster",
                        "evidence_artists": [clean_text(x) for x in evidence_artists if clean_text(x)][:8],
                        "ranking_rule": rule,
                    }
                )
            if len(clean_clusters) >= 6:
                break
        if len(rules) < 3:
            for row in clean_clusters:
                rule_text = row.get("ranking_rule") or (
                    f"Treat candidates matching the user's {row.get('cluster')} cluster as plausible when they bridge to evidence {terms['item_plural']}: "
                    f"{', '.join(row.get('evidence_artists', [])[:4])}."
                )
                rule_text = clean_text(rule_text)
                if rule_text and not self.tree_engine.user_policy_store._is_generic_item_protocol_rule(rule_text):
                    rules.append(self.tree_engine.user_policy_store._rule(rule_text, 0.52))
                if len(rules) >= 3:
                    break
        if len(rules) < 3:
            return None
        long_term_preference = pref_obj.get("long_term_preference")
        if not isinstance(long_term_preference, list) or not long_term_preference:
            if clean_clusters:
                long_term_preference = [row["cluster"] for row in clean_clusters[:5]]
            else:
                long_term_preference = [summary] if summary else ([str(target_profile)] if str(target_profile or "").strip() else [])

        core_preference = {
            "long_term_preference": [clean_text(x) for x in long_term_preference if clean_text(x)][:5],
            "novelty_tolerance": bounded_float(pref_obj.get("novelty_tolerance", obj.get("novelty_tolerance", 0.5)), 0.5),
            "stability_preference": bounded_float(pref_obj.get("stability_preference", obj.get("stability_preference", 0.5)), 0.5),
            "taste_clusters": clean_clusters,
        }
        preferences = obj.get("preferences")
        if not isinstance(preferences, list):
            preferences = pref_obj.get("preferences", [])
        clean_preferences = []
        for row in list(preferences or []):
            row = dict(row or {}) if isinstance(row, dict) else {"attribute": str(row)}
            attr = clean_text(row.get("attribute", "") or row.get("cluster", ""))
            if is_bad_skill_label(attr):
                continue
            evidence_artists = row.get("evidence_artists", [])
            if not isinstance(evidence_artists, list):
                evidence_artists = [evidence_artists]
            if attr:
                clean_preferences.append(
                    {
                        "attribute": attr,
                        "confidence": bounded_float(row.get("confidence", 0.50), 0.50),
                        "source": clean_text(row.get("source", "llm_label_supplement") or "llm_label_supplement"),
                        "evidence_artists": [clean_text(x) for x in evidence_artists if clean_text(x)][:8],
                        "evidence": clean_text(row.get("evidence", "") or row.get("ranking_rule", "")),
                    }
                )
            if len(clean_preferences) >= 8:
                break
        if clean_preferences:
            core_preference["preferences"] = clean_preferences
        recent_signals = obj.get("recent_signals")
        if not isinstance(recent_signals, list):
            recent_signals = pref_obj.get("recent_signals", [])
        clean_recent = []
        for row in list(recent_signals or []):
            row = dict(row or {}) if isinstance(row, dict) else {"attribute": str(row)}
            attr = clean_text(row.get("attribute", "") or row.get("cluster", ""))
            if is_bad_skill_label(attr):
                continue
            if attr:
                clean_recent.append(
                    {
                        "attribute": attr,
                        "confidence": bounded_float(row.get("confidence", 0.44), 0.44),
                        "source": clean_text(row.get("source", "llm_label_supplement_recent") or "llm_label_supplement_recent"),
                        "evidence_artists": [clean_text(x) for x in list(row.get("evidence_artists", []) or []) if clean_text(x)][:8],
                        "evidence": clean_text(row.get("evidence", "")),
                    }
                )
            if len(clean_recent) >= 5:
                break
        if clean_recent:
            core_preference["recent_signals"] = clean_recent
        decision_style = clean_text(obj.get("decision_style", "") or pref_obj.get("decision_style", ""))
        if decision_style:
            core_preference["decision_style"] = decision_style
        evidence = {
            "source": "llm_initial_core_skill",
            "music_taste_summary": summary,
            "product_preference_summary": summary if self._is_product_domain() else "",
            "book_preference_summary": summary if self._is_book_domain() else "",
            "raw_rule_count": int(len(raw_rules)),
            "taste_clusters": clean_clusters,
        }
        if isinstance(obj.get("initial_reasoning_evidence"), list):
            evidence["initial_reasoning_evidence"] = [clean_text(x) for x in obj.get("initial_reasoning_evidence", [])[:8]]
        return {
            "core_rules": rules,
            "core_preference": core_preference,
            "core_initial_evidence": evidence,
        }

    def _generate_llm_initial_core_skill(self, u_raw, history_str, target_profile, stat_payload=None):
        if not bool(getattr(self.args, "com_llm_init_user_core_skill", True)):
            return None
        terms = self._domain_terms()
        history_clues = self._extract_history_clues(history_str, max_n=80)
        history_view = ", ".join(history_clues) if history_clues else str(history_str or "")[:1600]
        stat_view = json.dumps(dict(stat_payload or {}), ensure_ascii=False)[:5000]
        system_prompt = (
            f"You initialize a COM User Item Selection Skill for {terms['dataset_label']}. "
            f"The code provides deterministic statistics as evidence only: user history clusters, recency, popularity, and {terms['stat_weak_signal']}. "
            f"Do not copy those statistical labels into the skill. Translate them into transferable {terms['item_word']} preference categories. "
            "The final skill must be written by you as concrete domain/user-preference labels, not as raw statistics. "
            "Do not choose an item, do not write candidate-list workflow, and do not write generic recommender advice. "
            "Output strict JSON only."
        )
        user_prompt = (
            f"UserId: {u_raw}\n"
            f"Dataset: {str(getattr(self.args, 'dataset', 'default') or 'default')}\n"
            f"{terms['history_label']}: {history_view if history_view else 'none'}\n"
            f"DeterministicStatInit:\n{stat_view if stat_view else 'none'}\n\n"
            "Initialize the Item Selection Skill only.\n"
            f"Use DeterministicStatInit only as background evidence for interpreting the long-term history. Do not output raw StatInit labels such as {terms['raw_stat_labels']}.\n"
            f"Produce 3-6 taste_clusters as concrete {terms['item_word']} preference categories: {terms['transferable_categories']}.\n"
            f"Each cluster must include evidence_artists and a ranking_rule. Here evidence_artists means {terms['evidence_field_meaning']} for compatibility with the existing policy schema.\n"
            f"Do not collapse the user into only the largest cluster. {terms['minority_clause']}\n"
            f"Then write item_reasoning_rules that describe this user's {terms['item_word']} preference and decision biases only. Do not mention scanning 20 candidates, top-k, top 5, shortlist construction, listwise ranking, PriorHint handling, or generic keep/drop procedure.\n"
            f"Rules must use domain evidence such as {terms['rules_evidence']}. Do not require live web lookup or unavailable external statistics.\n"
            f"Preference attributes must not be exact {terms['item_word']} names, raw popularity buckets, raw recency anchors, or raw name-shape signals. If a statistical clue is useful, convert it into a domain preference label first.\n"
            "Do not initialize communication who/what/how or post-feedback rules here.\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            f'  "{terms["summary_key"]}": "one concise user-specific summary",\n'
            '  "taste_clusters": [\n'
            f'    {{"cluster": "short cluster name", "evidence_artists": ["{terms["evidence_field_meaning"]}"], "ranking_rule": "how candidates matching this cluster should be ranked"}}\n'
            "  ],\n"
            f'  "preferences": [{{"attribute": "transferable user preference signal", "confidence": 0.30, "source": "llm_label_supplement", "evidence_artists": ["{terms["item_word"]}"], "evidence": "brief"}}],\n'
            f'  "recent_signals": [{{"attribute": "recent signal if supported", "confidence": 0.30, "source": "llm_label_supplement_recent", "evidence_artists": ["{terms["item_word"]}"], "evidence": "brief"}}],\n'
            '  "item_reasoning_rules": [\n'
            '    {"rule": "personalized reasoning rule", "confidence": 0.30}\n'
            "  ],\n"
            '  "core_preference": {\n'
            '    "long_term_preference": ["short preference cluster"],\n'
            '    "decision_style": "one short phrase",\n'
            '    "novelty_tolerance": 0.0,\n'
            '    "stability_preference": 0.0\n'
            "  },\n"
            '  "initial_reasoning_evidence": ["brief evidence from history"]\n'
            "}\n"
            "Use 3 to 6 taste_clusters and 4 to 7 item_reasoning_rules. Confidence should be 0.40 to 0.68 because this is inferred from history and should assist, not override, direct history evidence."
        )
        try:
            if not hasattr(self.args, "max_retry_num"):
                setattr(self.args, "max_retry_num", 2)
            if not hasattr(self.args, "temperature"):
                setattr(self.args, "temperature", 0.2)
            resp = llm_request(system_prompt, user_prompt, self.args)
        except Exception as exc:
            print(f"[com] LLM core skill init failed user={u_raw}: {exc}")
            return None
        payload = self._extract_json_object(resp)
        parsed = self._sanitize_llm_core_skill_payload(payload, target_profile)
        if parsed is None:
            print(f"[com] LLM core skill init unparseable user={u_raw}; fallback deterministic core rules")
            return None
        return parsed

    def _bootstrap_user_policy_files(self, root_dir):
        created = 0
        refreshed = 0
        total = 0
        seen_users = set()
        force_rebuild = bool(getattr(self.args, "com_rebuild_initial_user_policy", False))
        rebuild_comm_only = bool(getattr(self.args, "com_rebuild_communication_user_policy_only", False)) and not force_rebuild
        refresh_comm_evidence = bool(getattr(self.args, "com_refresh_communication_initial_evidence", True))
        sample_num = int(getattr(self.args, "com_train_sample_num", 0) or getattr(self.args, "com_test_sample_num", 0) or 0)
        sample_seed = int(getattr(self.args, "com_test_sample_seed", 2026))

        def refresh_policy_comm_evidence(policy, selected_comm_evidence):
            policy = dict(policy or {})
            policy["communication_initial_evidence"] = dict(selected_comm_evidence or {})
            comm_skill = dict(policy.get("communication_selection_skill", {}) or {})
            active_rules = []
            for row in list(comm_skill.get("active_rules", []) or []):
                if str((row or {}).get("rule", "") or "").startswith("Initial who preference is "):
                    continue
                active_rules.append(row)
            active_rules.append(
                self.tree_engine.user_policy_store._initial_communication_rules(
                    communication_evidence=selected_comm_evidence
                )[-1]
            )
            comm_skill["active_rules"] = active_rules
            who_pref, selected_comm_evidence = self.tree_engine.user_policy_store._initial_who_preference(
                selected_comm_evidence
            )
            comm_skill["who_preferences"] = [
                {"attribute": key, "confidence": float(value), "confidence_label": self.tree_engine.user_policy_store._confidence_label(value)}
                for key, value in sorted(who_pref.items(), key=lambda kv: kv[1], reverse=True)
            ] or comm_skill.get("who_preferences", [])
            policy["communication_selection_skill"] = comm_skill
            return policy

        def rebuild_policy_communication_route(policy, selected_comm_evidence, history_summary=""):
            policy = self.tree_engine.user_policy_store.normalize_policy(dict(policy or {}))
            item_skill = dict(policy.get("item_selection_skill", {}) or {})
            policy["communication_route_skill"] = self.tree_engine.user_policy_store._initial_communication_route_skill(
                communication_evidence=dict(selected_comm_evidence or {}),
                item_skill=item_skill,
                history_summary=str(history_summary or ""),
            )
            return self.tree_engine.user_policy_store.normalize_policy(policy)

        def clear_communication_slim_cache(user_raw):
            paths = self.tree_engine.user_policy_store._paths(user_raw)
            for key in ["slim_cache_communication_json", "slim_cache_json"]:
                try:
                    if paths[key].exists():
                        paths[key].unlink()
                except Exception:
                    pass

        # COM train initializes user skills from the validation prediction task:
        # history -> valid target. Test later uses the held-out test task with
        # the trained/fixed skills.
        for stage, suffix in [("train", "_val")]:
            try:
                dataset_map = self._load_agent_dataset(root_dir, stage=stage, suffix=suffix)
            except Exception as exc:
                print(f"[com] skip user policy bootstrap stage={stage}: {exc}")
                continue
            stage_items = list(dataset_map.items())
            fixed_users = self._fixed_user_ids()
            if fixed_users:
                stage_dict = {str(u): (u, sample) for u, sample in stage_items}
                missing = [u for u in fixed_users if str(u) not in stage_dict]
                if missing:
                    print(f"[com] fixed bootstrap users missing from dataset: {missing[:10]}{'...' if len(missing) > 10 else ''}")
                stage_items = [stage_dict[str(u)] for u in fixed_users if str(u) in stage_dict]
                print(f"[com] fixed bootstrap users: {len(stage_items)} / {len(dataset_map)}")
            elif sample_num > 0:
                bootstrap_seed = sample_seed if self._com_train_sample_offset() >= 0 else sample_seed + (17 if suffix else 0)
                stage_items = self._select_com_train_sample(
                    stage_items,
                    sample_num=sample_num,
                    sample_seed=bootstrap_seed,
                    label="bootstrap train",
                    user_key_fn=lambda item: item[0],
                )
            bootstrap_items = []
            for u_raw, sample in stage_items:
                key = str(u_raw)
                if key in seen_users:
                    continue
                seen_users.add(key)
                bootstrap_items.append((u_raw, sample))

            total += len(bootstrap_items)

            def process_bootstrap_item(item):
                u_raw, sample = item
                key = str(u_raw)
                history_str = str(sample.get("seq_str", "") or "")
                u_int = self._to_internal_id(key, is_user=True)
                target_profile = ""
                communication_evidence = self._build_communication_initial_evidence(
                    u_raw=key,
                    u_int=u_int,
                    sample=sample,
                    history_str=history_str,
                )
                policy_exists = bool(self.tree_engine.user_policy_store.policy_exists(key))
                if rebuild_comm_only and policy_exists:
                    full_policy, source = self.tree_engine.user_policy_store.load_full_policy(
                        user_raw=key,
                        history_summary=history_str,
                        target_profile=target_profile,
                        stage="train",
                        communication_evidence=communication_evidence,
                        force_bootstrap=False,
                    )
                    _, selected_comm_evidence = self.tree_engine.user_policy_store._initial_who_preference(
                        dict(communication_evidence or {})
                    )
                    full_policy = rebuild_policy_communication_route(full_policy, selected_comm_evidence, history_summary=history_str)
                    self.tree_engine.user_policy_store.save_full_policy(
                        full_policy,
                        snapshot_reason="communication_route_skill_rebootstrap",
                    )
                    initial_paths = self.tree_engine.user_policy_store._existing_paths(key, initial=True)
                    if initial_paths is not None and initial_paths["policy_json"].exists():
                        initial_policy = load_json(initial_paths["policy_json"], default=None)
                        if initial_policy is not None:
                            initial_policy = rebuild_policy_communication_route(initial_policy, selected_comm_evidence, history_summary=history_str)
                            self.tree_engine.user_policy_store.save_initial_policy_if_missing(
                                initial_policy,
                                overwrite=True,
                            )
                    clear_communication_slim_cache(key)
                    self.tree_engine.user_policy_store.append_evolution_log(
                        key,
                        {
                            "event": "communication_route_skill_reinitialized",
                            "stage": "bootstrap",
                            "preserved_item_selection_skill": True,
                            "source": str(source or "persisted"),
                        },
                    )
                    return {"user": key, "created": 0, "refreshed": 1}
                if rebuild_comm_only and not policy_exists:
                    print(f"[com] skip communication-only init user={key}: policy.json missing; initialize item skill first")
                    return {"user": key, "created": 0, "refreshed": 0}
                core_payload = None
                if force_rebuild or not self.tree_engine.user_policy_store.policy_exists(key):
                    stat_payload = self._build_stat_initial_core_skill(
                        u_raw=key,
                        sample=sample,
                        history_str=history_str,
                        target_profile=target_profile,
                    )
                    llm_payload = self._generate_llm_initial_core_skill(
                        u_raw=key,
                        history_str=history_str,
                        target_profile=target_profile,
                        stat_payload=stat_payload,
                    )
                    core_payload = self._merge_llm_label_payload(stat_payload, llm_payload)
                full_policy, source = self.tree_engine.user_policy_store.load_full_policy(
                    user_raw=key,
                    history_summary=history_str,
                    target_profile=target_profile,
                    stage="train",
                    communication_evidence=communication_evidence,
                    core_rules=(core_payload or {}).get("core_rules"),
                    core_preference=(core_payload or {}).get("core_preference"),
                    core_initial_evidence=(core_payload or {}).get("core_initial_evidence"),
                    force_bootstrap=force_rebuild,
                )
                item_created = 1 if source in ["bootstrapped", "rebootstrapped"] else 0
                item_refreshed = 0
                if source == "persisted" and refresh_comm_evidence:
                    _, selected_comm_evidence = self.tree_engine.user_policy_store._initial_who_preference(
                        dict(communication_evidence or {})
                    )
                    old_comm_evidence = dict((full_policy or {}).get("communication_initial_evidence", {}) or {})
                    if old_comm_evidence != selected_comm_evidence:
                        full_policy = refresh_policy_comm_evidence(full_policy, selected_comm_evidence)
                        self.tree_engine.user_policy_store.save_full_policy(
                            full_policy,
                            snapshot_reason="communication_initial_evidence_refresh",
                        )
                        item_refreshed += 1
                    initial_paths = self.tree_engine.user_policy_store._existing_paths(key, initial=True)
                    initial_policy = None
                    if initial_paths is not None:
                        initial_policy = load_json(initial_paths["policy_json"], default=None)
                    initial_state = dict((initial_policy or {}).get("policy_evolution_state", {}) or {})
                    initial_comm_evidence = dict((initial_policy or {}).get("communication_initial_evidence", {}) or {})
                    if (
                        initial_policy is not None
                        and int(initial_state.get("num_updates", 0) or 0) == 0
                        and initial_comm_evidence != selected_comm_evidence
                    ):
                        initial_policy = refresh_policy_comm_evidence(initial_policy, selected_comm_evidence)
                        self.tree_engine.user_policy_store.save_initial_policy_if_missing(
                            initial_policy,
                            overwrite=True,
                        )
                return {"user": key, "created": item_created, "refreshed": item_refreshed}

            bootstrap_workers = max(1, int(getattr(self.args, "agent_workers", 1) or 1))
            if bootstrap_items:
                print(f"[com] bootstrap user policy workers={bootstrap_workers}, llm_init={bool(getattr(self.args, 'com_llm_init_user_core_skill', True))}")
            if bootstrap_workers <= 1 or len(bootstrap_items) <= 1:
                iterator = tqdm(bootstrap_items, total=len(bootstrap_items), desc="COM User Skill Bootstrap")
                for item in iterator:
                    result = process_bootstrap_item(item)
                    created += int(result.get("created", 0) or 0)
                    refreshed += int(result.get("refreshed", 0) or 0)
            else:
                with ThreadPoolExecutor(max_workers=bootstrap_workers) as executor:
                    futures = {executor.submit(process_bootstrap_item, item): item[0] for item in bootstrap_items}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="COM User Skill Bootstrap"):
                        try:
                            result = future.result()
                        except Exception as exc:
                            u_raw = futures.get(future, "unknown")
                            print(f"[com] bootstrap user policy failed user={u_raw}: {exc}")
                            continue
                        created += int(result.get("created", 0) or 0)
                        refreshed += int(result.get("refreshed", 0) or 0)
        print(
            f"[com] user policy bootstrap done: users={total}, created={created}, refreshed={refreshed}, "
            f"force_rebuild={force_rebuild}, rebuild_comm_only={rebuild_comm_only}"
        )
        return {"users": int(total), "created": int(created), "refreshed": int(refreshed)}

    def train(self, save_dir=None, root_dir=None, **kwargs):
        self.tree_engine.bootstrap(stage="train")
        ext_path = str(getattr(self.args, "com_prior_csv_path", "") or "").strip()
        ext_val_path = str(getattr(self.args, "com_prior_val_csv_path", "") or "").strip()
        has_external_prior = bool(ext_path or ext_val_path)
        if self.tool_model is None:
            if not has_external_prior:
                raise ValueError("tool_model is required unless external COM prior csv is provided")
        elif not hasattr(self.tool_model, "recbole_model") or self.tool_model.recbole_model is None:
            if not has_external_prior:
                raise ValueError("tool_model is not loaded")
        else:
            self.model = self.tool_model.recbole_model
            self.tool_model.arlib_dataset = self.data

            try:
                self.state_size = int(self.tool_model.recbole_config["MAX_ITEM_LIST_LENGTH"])
            except Exception:
                pass

        self._bootstrap_user_policy_files(root_dir)
        if bool(getattr(self.args, "com_bootstrap_user_policy_only", False)):
            print("[com] bootstrap user policy only; skip prior generation and COM interaction training")
            self.train_results = {}
            return
        if ext_path or ext_val_path:
            print(
                "[com] use external prior csv; skip internal prior generation "
                f"(test={ext_path or 'auto'}, val={ext_val_path or 'auto'})"
            )
        else:
            candidates_test = self._load_candidate_json(root_dir)
            self._generate_prior_csv(candidates_test, output_dir=save_dir, suffix="")

            candidates_val = self._load_candidate_json(root_dir, suffix="_val")
            self._generate_prior_csv(candidates_val, output_dir=save_dir, suffix="_val")

        print("Training com (interaction + evolution)")
        self.train_results = {}
        _reset_llm_usage_tracker()
        self._run_stage_interactions(
            stage="train",
            result_attr="train_results",
            save_dir=save_dir,
            root_dir=root_dir,
        )

    def _process_single_user(
        self,
        u_raw,
        com_args,
        agent_dataset,
        prior_recs,
        gt_map,
        enable_debug=False,
        collect_trace=False,
        stage="test",
    ):
        sample = agent_dataset.get(u_raw)
        if not sample:
            return None

        u_int = self._to_internal_id(u_raw, is_user=True)
        if u_int is None:
            return None

        seq_int = list(sample.get("seq", []))[-self.state_size :]
        cands_int = list(sample.get("cans", []))
        sample_target_items = []
        sample_target = sample.get("target")
        if sample_target is not None:
            try:
                sample_target_items.append(int(sample_target))
            except Exception:
                pass
        if len(cands_int) <= 1:
            fallback = self._fallback_candidates_from_tool(seq_int, k=max(20, self.max_N))
            for iid in fallback:
                if iid not in cands_int:
                    cands_int.append(iid)

        if not cands_int:
            return None

        gt_items = set(sample_target_items) if sample_target_items else (gt_map.get(int(u_int), set()) if gt_map else set())
        original_cands_int = list(cands_int)
        missing_targets = [int(iid) for iid in gt_items if int(iid) not in set(cands_int)]
        target_candidate_overlap_before = bool(set(cands_int) & set(gt_items))
        if missing_targets:
            if bool(getattr(self.args, "com_strict_target_candidate_check", False)):
                missing_names = [str(self._get_item_name(int(iid))) for iid in missing_targets]
                raise ValueError(
                    "COM candidate set does not contain the current stage target item: "
                    f"user={u_raw}; missing_targets={missing_names}"
                )
            if bool(getattr(self.args, "com_ensure_target_in_candidates", True)):
                for iid in missing_targets:
                    if int(iid) not in cands_int:
                        cands_int.append(int(iid))
        target_candidate_overlap_after = bool(set(cands_int) & set(gt_items))

        score_dict = self._score_candidates(seq_int, cands_int)
        if not score_dict:
            score_dict = {iid: float(-rank) for rank, iid in enumerate(cands_int)}

        prior_hint = prior_recs.get(u_raw, "")

        user_agent = ComUserAgent(com_args)
        advisor_agent = ComAdvisorAgent(com_args)
        final_iid, trace_rounds, raw_trace, interaction_summary = self.tree_engine.run_interaction(
            host=self,
            user_agent=user_agent,
            advisor_agent=advisor_agent,
            sample=sample,
            u_raw=u_raw,
            u_int=u_int,
            cands_int=cands_int,
            score_dict=score_dict,
            prior_hint=prior_hint,
            gt_items=gt_items,
            collect_trace=collect_trace,
            stage=str(stage or "test"),
        )

        ranking = [final_iid] + [iid for iid in cands_int if iid != final_iid]
        ranking = ranking[: self.max_N]

        structured_trace_obj = None
        raw_trace_obj = None
        if collect_trace:
            top1_hit = bool(final_iid in gt_items)
            final_item_name = str(self._get_item_name(final_iid))
            sample_meta = {
                "stage": str(stage or "test"),
                "user_id": str(u_raw),
                "user_int": int(u_int),
                "candidate_count": int(len(cands_int)),
                "original_candidate_count": int(len(original_cands_int)),
                "candidate_repair_applied": bool(len(cands_int) != len(original_cands_int)),
                "target_candidate_overlap_before_repair": bool(target_candidate_overlap_before),
                "target_candidate_overlap_after_repair": bool(target_candidate_overlap_after),
                "target_item_ids": [int(iid) for iid in gt_items],
                "target_item_names": [str(self._get_item_name(int(iid))) for iid in gt_items],
                "target_source": str(sample.get("target_source", str(stage or "test"))),
                "candidate_target_injected_by_dataset": bool(sample.get("candidate_target_injected", False)),
                "original_test_target_id": int(sample.get("original_test_target", sample_target or -1)),
                "original_test_target_name": str(
                    self._get_item_name(int(sample.get("original_test_target", sample_target)))
                    if sample.get("original_test_target", sample_target) is not None
                    else ""
                ),
                "missing_target_item_ids_before_repair": [int(iid) for iid in missing_targets],
                "missing_target_item_names_before_repair": [str(self._get_item_name(int(iid))) for iid in missing_targets],
                "friend_mode": "skill_driven",
                "prior_hint": str(prior_hint),
                "final_item": final_item_name,
                "final_item_id": int(final_iid),
                "is_same_as_prior": bool(final_item_name == str(prior_hint)),
                "top1_hit": bool(top1_hit),
            }
            if self.dialogue_include_history:
                sample_meta["history"] = str(sample.get("seq_str", ""))

            structured_trace_obj = {
                "sample_meta": dict(sample_meta),
                "rounds": trace_rounds,
            }
            raw_trace_obj = {
                "sample_meta": dict(sample_meta),
                "raw_trace": raw_trace,
            }

        outcome_summary = dict(interaction_summary or {})
        outcome_summary.update(
            {
                "stage": str(stage or "test"),
                "user_id": str(u_raw),
                "user_int": int(u_int),
                "final_item_id": int(final_iid),
                "final_item_name": str(self._get_item_name(final_iid)),
                "target_item_ids": [int(iid) for iid in gt_items],
                "target_item_names": [str(self._get_item_name(int(iid))) for iid in gt_items],
                "target_candidate_overlap_before_repair": bool(target_candidate_overlap_before),
                "target_candidate_overlap_after_repair": bool(target_candidate_overlap_after),
            }
        )
        return u_int, ranking, structured_trace_obj, raw_trace_obj, outcome_summary

    def _compute_overall_topk_metrics(self):
        top_raw = str(getattr(self.args, "topK", "1"))
        topk = sorted(set([int(x) for x in top_raw.split(",") if x.strip().isdigit()]))
        if not topk:
            topk = [1]

        gt_map = {}
        test_data = getattr(self.data, "test_data", [])
        for x in test_data:
            u_int, i_int = int(x[0]), int(x[1])
            gt_map.setdefault(u_int, set()).add(i_int)

        users = [u for u in self.agent_results.keys() if u in gt_map]
        if not users:
            return {
                "topk": topk,
                "num_eval_users": 0,
                "precision": [0.0 for _ in topk],
                "hr": [0.0 for _ in topk],
                "recall": [0.0 for _ in topk],
                "ndcg": [0.0 for _ in topk],
            }

        precision = []
        hr = []
        recall = []
        ndcg = []

        for k in topk:
            hit_cnt = 0
            hr_hit_users = 0
            total_prec_den = 0
            total_recall_den = 0
            dcg_sum = 0.0
            idcg_sum = 0.0

            for u in users:
                pred = list(self.agent_results.get(u, []))[:k]
                true_items = gt_map[u]
                inter = set(pred) & true_items

                hit_cnt += len(inter)
                total_prec_den += k
                total_recall_den += len(true_items)
                if len(inter) > 0:
                    hr_hit_users += 1

                dcg_u = 0.0
                for r, iid in enumerate(pred):
                    if iid in true_items:
                        dcg_u += 1.0 / math.log2(r + 2)
                idcg_u = 0.0
                for r in range(min(k, len(true_items))):
                    idcg_u += 1.0 / math.log2(r + 2)
                dcg_sum += dcg_u
                idcg_sum += idcg_u

            precision.append(hit_cnt / max(1, total_prec_den))
            hr.append(hr_hit_users / max(1, len(users)))
            recall.append(hit_cnt / max(1, total_recall_den))
            ndcg.append(dcg_sum / max(1e-12, idcg_sum))

        return {
            "topk": topk,
            "num_eval_users": int(len(users)),
            "precision": precision,
            "hr": hr,
            "recall": recall,
            "ndcg": ndcg,
        }

    @staticmethod
    def _write_eval_metrics(save_dir, metrics, stage="test"):
        if not save_dir:
            return
        out_dir = os.path.join(save_dir, "eval_metrics")
        os.makedirs(out_dir, exist_ok=True)

        latest_path = os.path.join(out_dir, f"{stage}_metrics_latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        history_path = os.path.join(out_dir, f"{stage}_metrics_history.jsonl")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_dialogue_trace(save_dir, traces, stage="test"):
        if (not save_dir) or (not traces):
            return

        out_dir = os.path.join(save_dir, "dialogue_trace")
        os.makedirs(out_dir, exist_ok=True)

        ts = int(time.time())
        run_path = os.path.join(out_dir, f"{stage}_dialogue_{ts}.jsonl")
        with open(run_path, "w", encoding="utf-8") as f:
            for row in traces:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        latest_path = os.path.join(out_dir, f"{stage}_dialogue_latest.jsonl")
        with open(latest_path, "w", encoding="utf-8") as f:
            for row in traces:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[com] saved dialogue traces -> {latest_path} users={len(traces)}")

    @staticmethod
    def _write_llm_usage_stats(save_dir, usage_stats, stage="test", print_to_terminal=False):
        if (not save_dir) or (not usage_stats):
            return
        out_dir = os.path.join(save_dir, "eval_metrics")
        os.makedirs(out_dir, exist_ok=True)

        payload = {
            "stage": stage,
            "ts": int(time.time()),
            "llm_usage": usage_stats,
        }

        latest_path = os.path.join(out_dir, f"{stage}_llm_usage_latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        history_path = os.path.join(out_dir, f"{stage}_llm_usage_history.jsonl")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if print_to_terminal:
            calls = int(usage_stats.get("calls", 0))
            prompt_toks = int(usage_stats.get("prompt_tokens", 0))
            completion_toks = int(usage_stats.get("completion_tokens", 0))
            total_toks = int(usage_stats.get("total_tokens", 0))
            est_cost = float(usage_stats.get("estimated_cost", 0.0))
            currency = str(usage_stats.get("currency", "USD")) if isinstance(usage_stats, dict) else "USD"
            print(
                f"[com] llm usage: calls={calls}, prompt_tokens={prompt_toks}, "
                f"completion_tokens={completion_toks}, total_tokens={total_toks}, "
                f"estimated_cost={est_cost:.6f} {currency}"
            )
            per_phase = usage_stats.get("per_phase", {}) if isinstance(usage_stats, dict) else {}
            tracked_phases = [
                "communication_evolution_decision",
                "tree_common_cause_analysis",
                "tree_node_generation",
            ]
            for phase in tracked_phases:
                row = dict((per_phase or {}).get(phase, {}) or {})
                print(
                    f"[com] llm usage phase={phase}: calls={int(row.get('calls', 0) or 0)}, "
                    f"prompt_tokens={int(row.get('prompt_tokens', 0) or 0)}, "
                    f"completion_tokens={int(row.get('completion_tokens', 0) or 0)}, "
                    f"total_tokens={int(row.get('total_tokens', 0) or 0)}, "
                    f"estimated_cost={float(row.get('estimated_cost', 0.0) or 0.0):.6f} {currency}"
                )

    @staticmethod
    def _write_failed_users(save_dir, failed_users, stage="test"):
        if not save_dir:
            return
        out_dir = os.path.join(save_dir, "eval_metrics")
        os.makedirs(out_dir, exist_ok=True)

        latest_path = os.path.join(out_dir, f"{stage}_failed_users_latest.jsonl")
        with open(latest_path, "w", encoding="utf-8") as f:
            for row in failed_users or []:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        history_path = os.path.join(out_dir, f"{stage}_failed_users_history.jsonl")
        payload = {
            "stage": str(stage),
            "ts": int(time.time()),
            "failed_count": int(len(failed_users or [])),
            "failed_users": list(failed_users or []),
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_failed_user_queue(save_dir, failed_queue, stage="train", run_path=None, latest_path=None, ts=None):
        if not save_dir or str(stage or "").lower() != "train":
            return {}
        out_dir = os.path.join(save_dir, "dialogue_trace")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(ts or time.time())
        run_path = run_path or os.path.join(out_dir, f"failed_user_queue_{ts}.jsonl")
        latest_path = latest_path or os.path.join(out_dir, "failed_user_queue_latest.jsonl")
        rows = [dict(row or {}) for row in list(failed_queue or []) if isinstance(row, dict)]
        for path in [run_path, latest_path]:
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "stage": str(stage or "train"),
            "ts": ts,
            "failed_count": int(len(rows)),
            "queue_path": latest_path,
            "outcome_counts": {
                outcome: sum(1 for row in rows if str(row.get("outcome_signal", "") or "") == outcome)
                for outcome in ["TW", "WW"]
            },
            "attribution_counts": {
                key: sum(1 for row in rows if str(row.get("failure_attribution", "") or "") == key)
                for key in sorted({str(row.get("failure_attribution", "") or "") for row in rows if str(row.get("failure_attribution", "") or "")})
            },
        }
        history_path = os.path.join(out_dir, "failed_user_queue_history.jsonl")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"[com] saved failed-user replay queue -> {latest_path} users={len(rows)}")
        return {"latest_path": latest_path, "run_path": run_path, "summary": summary}

    @staticmethod
    def _reset_eval_progress(save_dir, stage="test"):
        if not save_dir:
            return
        out_dir = os.path.join(save_dir, "eval_metrics")
        os.makedirs(out_dir, exist_ok=True)
        current_path = os.path.join(out_dir, f"{stage}_progress_current.jsonl")
        with open(current_path, "w", encoding="utf-8") as f:
            f.write("")

    @staticmethod
    def _write_eval_progress(save_dir, metrics, stage="test"):
        if not save_dir:
            return
        out_dir = os.path.join(save_dir, "eval_metrics")
        os.makedirs(out_dir, exist_ok=True)
        latest_path = os.path.join(out_dir, f"{stage}_progress_latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        history_path = os.path.join(out_dir, f"{stage}_progress.jsonl")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        current_path = os.path.join(out_dir, f"{stage}_progress_current.jsonl")
        with open(current_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    @staticmethod
    def _counter_top(counter, limit=20):
        return [
            {"key": str(key), "count": int(value)}
            for key, value in sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:limit]
        ]

    def _summarize_dialogue_traces(self, traces):
        outcome_counts = defaultdict(int)
        user_outcome_counts = defaultdict(int)
        failure_level_counts = defaultdict(int)
        path_counts = defaultdict(int)
        communication_action_counts = defaultdict(int)
        tree_operation_counts = defaultdict(int)
        final_action_counts = defaultdict(int)
        path_breakdowns = {
            "by_why": defaultdict(lambda: defaultdict(int)),
            "by_who": defaultdict(lambda: defaultdict(int)),
            "by_how": defaultdict(lambda: defaultdict(int)),
            "by_path": defaultdict(lambda: defaultdict(int)),
        }
        candidate_target_overlap_count = 0
        focus_target_overlap_count = 0
        advisor_pool_empty_count = 0
        original_advisor_pool_empty_count = 0
        advisor_pool_rerouted_count = 0
        final_advisor_pool_empty_count = 0
        target_injected_into_focus_count = 0
        committee_tie_count = 0
        committee_winner_target_count = 0
        committee_winner_followed_count = 0
        committee_winner_overridden_count = 0
        committee_target_winner_final_miss_count = 0
        committee_non_target_winner_final_target_count = 0
        target_focus_with_advisor_evidence_count = 0
        target_focus_without_advisor_evidence_count = 0
        target_focus_silent_count = 0
        protocol_issue_round_count = 0
        protocol_issue_count = 0
        missing_advisor_evidence_count = 0
        unchallenged_support_count = 0
        advisor_feedback_round_count = 0
        advisor_feedback_total_count = 0
        initial_hit_count = 0
        final_hit_count = 0
        prior_hit_count = 0
        proposal_equals_prior_count = 0
        target_in_shortlist_to_final_hit_count = 0
        target_in_shortlist_to_final_miss_count = 0
        initial_hit_to_final_miss_count = 0
        switch_count = 0
        switch_to_target_count = 0
        switch_away_from_target_count = 0
        switch_between_non_targets_count = 0
        keep_target_count = 0
        keep_wrong_count = 0
        skipped_with_target_shortlist_final_miss_count = 0
        round_count = 0
        interaction_count = 0
        interaction_initial_shortlist_count = 0
        interaction_initial_shortlist_final_hit_count = 0
        interaction_initial_shortlist_final_miss_count = 0
        interaction_initial_hit_count = 0
        interaction_initial_hit_final_miss_count = 0
        interaction_switch_to_target_count = 0
        interaction_switch_away_from_target_count = 0
        interaction_communication_started_count = 0
        interaction_no_communication_count = 0
        interaction_shortlist_no_comm_final_miss_count = 0
        communication_train_eligible_count = 0
        communication_train_skipped_count = 0
        communication_train_skipped_by_gate_count = 0
        round1_continue_count = 0
        round2_repair_round_count = 0
        round2_repair_success_count = 0

        def rate(num, den):
            return float(float(num) / float(max(1, den)))

        def norm(value):
            return " ".join(str(value or "").strip().lower().split())

        def contains_target(value, target_names):
            value_norm = norm(value)
            if not value_norm:
                return False
            return value_norm in {norm(x) for x in (target_names or []) if norm(x)}

        def path_key_from(path):
            return " -> ".join(
                [
                    str((path or {}).get('why', "") or ""),
                    str((path or {}).get("who", "") or ""),
                    str((path or {}).get("what", "") or ""),
                    str((path or {}).get("how", "") or ""),
                ]
            )

        def update_breakdown(group_name, key, evaluation, communication_action):
            key = str(key or "unknown")
            bucket = path_breakdowns[group_name][key]
            bucket["round_count"] += 1
            outcome = str((evaluation or {}).get("outcome_signal", "") or "")
            if outcome:
                bucket[f"outcome_{outcome}"] += 1
            if (evaluation or {}).get("initial_hit"):
                bucket["initial_hit_count"] += 1
            if (evaluation or {}).get("final_hit"):
                bucket["final_hit_count"] += 1
            if (evaluation or {}).get("focus_target_overlap"):
                bucket["target_in_shortlist_count"] += 1
                if (evaluation or {}).get("final_hit"):
                    bucket["target_in_shortlist_to_final_hit_count"] += 1
            if outcome == "WT":
                bucket["gain_count"] += 1
            if outcome == "TW":
                bucket["damage_count"] += 1
            if (evaluation or {}).get("advisor_pool_empty"):
                bucket["advisor_pool_empty_count"] += 1
            if (evaluation or {}).get("advisor_pool_rerouted"):
                bucket["advisor_pool_rerouted_count"] += 1
            if str(communication_action or "") in ["skip", "stage1_only"]:
                bucket["no_communication_count"] += 1

        def format_breakdowns(rows, limit=30):
            formatted = []
            for key, raw in sorted(rows.items(), key=lambda kv: (-int(kv[1].get("round_count", 0)), str(kv[0])))[:limit]:
                bucket = dict(raw)
                total_rows = int(bucket.get("round_count", 0) or 0)
                target_shortlist = int(bucket.get("target_in_shortlist_count", 0) or 0)
                gain = int(bucket.get("gain_count", 0) or 0)
                damage = int(bucket.get("damage_count", 0) or 0)
                formatted.append(
                    {
                        "key": str(key),
                        **{k: int(v) for k, v in bucket.items()},
                        "success_rate": rate(bucket.get("final_hit_count", 0), total_rows),
                        "target_shortlist_to_final_hit_rate": rate(
                            bucket.get("target_in_shortlist_to_final_hit_count", 0),
                            target_shortlist,
                        ),
                        "gain_rate": rate(gain, total_rows),
                        "damage_rate": rate(damage, total_rows),
                        "net_gain_count": int(gain - damage),
                        "advisor_pool_empty_rate": rate(bucket.get("advisor_pool_empty_count", 0), total_rows),
                        "advisor_pool_reroute_rate": rate(bucket.get("advisor_pool_rerouted_count", 0), total_rows),
                    }
                )
            return formatted

        for trace_obj in traces or []:
            rounds = list((trace_obj or {}).get("rounds", []) or [])
            if rounds:
                interaction_count += 1
                first_eval = dict((rounds[0] or {}).get("stage6_evaluation", {}) or {})
                last_eval = dict((rounds[-1] or {}).get("stage6_evaluation", {}) or {})
                initial_shortlist = bool(first_eval.get("focus_target_overlap", False))
                initial_hit = bool(first_eval.get("initial_hit", False))
                final_hit = bool(last_eval.get("final_hit", False))
                if initial_hit and final_hit:
                    user_outcome_counts["TT"] += 1
                elif initial_hit and not final_hit:
                    user_outcome_counts["TW"] += 1
                elif (not initial_hit) and final_hit:
                    user_outcome_counts["WT"] += 1
                else:
                    user_outcome_counts["WW"] += 1
                any_communication = False
                for round_obj in rounds:
                    action = str(
                        ((round_obj or {}).get("stage1_decision_and_trigger", {}) or {}).get(
                            "communication_action",
                            "",
                        )
                        or ""
                    ).strip().lower()
                    if action in ["start", "continue"]:
                        any_communication = True
                        break
                if initial_shortlist:
                    interaction_initial_shortlist_count += 1
                    if final_hit:
                        interaction_initial_shortlist_final_hit_count += 1
                    else:
                        interaction_initial_shortlist_final_miss_count += 1
                        if not any_communication:
                            interaction_shortlist_no_comm_final_miss_count += 1
                if initial_hit:
                    interaction_initial_hit_count += 1
                    if not final_hit:
                        interaction_initial_hit_final_miss_count += 1
                        interaction_switch_away_from_target_count += 1
                elif final_hit:
                    interaction_switch_to_target_count += 1
                if any_communication:
                    interaction_communication_started_count += 1
                else:
                    interaction_no_communication_count += 1

            for round_obj in rounds:
                round_count += 1
                stage1 = dict(round_obj.get("stage1_decision_and_trigger", {}) or {})
                communication_action = str(stage1.get("communication_action", "") or "").strip().lower()
                if communication_action:
                    communication_action_counts[communication_action] += 1
                evaluation = dict(round_obj.get("stage6_evaluation", {}) or {})
                target_names = list(evaluation.get("target_item_names", []) or [])
                if evaluation.get("candidate_target_overlap"):
                    candidate_target_overlap_count += 1
                if evaluation.get("focus_target_overlap"):
                    focus_target_overlap_count += 1
                    if evaluation.get("final_hit"):
                        target_in_shortlist_to_final_hit_count += 1
                    else:
                        target_in_shortlist_to_final_miss_count += 1
                if evaluation.get("advisor_pool_empty"):
                    advisor_pool_empty_count += 1
                if evaluation.get("original_advisor_pool_empty"):
                    original_advisor_pool_empty_count += 1
                if evaluation.get("advisor_pool_rerouted"):
                    advisor_pool_rerouted_count += 1
                if evaluation.get("final_advisor_pool_empty"):
                    final_advisor_pool_empty_count += 1
                if evaluation.get("target_injected_into_focus"):
                    target_injected_into_focus_count += 1
                if evaluation.get("communication_train_eligible"):
                    communication_train_eligible_count += 1
                if evaluation.get("communication_skipped_by_training_gate"):
                    communication_train_skipped_count += 1
                    communication_train_skipped_by_gate_count += 1
                if evaluation.get("committee_tie"):
                    committee_tie_count += 1
                if evaluation.get("initial_hit"):
                    initial_hit_count += 1
                    if not evaluation.get("final_hit"):
                        initial_hit_to_final_miss_count += 1
                if evaluation.get("final_hit"):
                    final_hit_count += 1
                if evaluation.get("prior_hit"):
                    prior_hit_count += 1
                if evaluation.get("proposal_equals_prior"):
                    proposal_equals_prior_count += 1
                outcome = str(evaluation.get("outcome_signal", "") or "")
                if outcome:
                    outcome_counts[outcome] += 1

                path = dict((round_obj.get("stage2_path_selection", {}) or {}).get("selected_path", {}) or {})
                path_key = path_key_from(path)
                if path_key.strip(" ->"):
                    path_counts[path_key] += 1
                update_breakdown("by_why", path.get('why', "unknown"), evaluation, communication_action)
                update_breakdown("by_who", path.get("who", "unknown"), evaluation, communication_action)
                update_breakdown("by_how", path.get("how", "unknown"), evaluation, communication_action)
                update_breakdown("by_path", path_key or "unknown", evaluation, communication_action)

                stage3 = dict(round_obj.get("stage3_communication_execution", {}) or {})
                committee = dict(stage3.get("committee_result", {}) or {})
                advisor_feedbacks = list(stage3.get("advisor_feedbacks", []) or [])
                if advisor_feedbacks:
                    advisor_feedback_round_count += 1
                    advisor_feedback_total_count += len(advisor_feedbacks)
                protocol_issues = list(committee.get("protocol_issues", []) or [])
                if protocol_issues:
                    protocol_issue_round_count += 1
                    protocol_issue_count += len(protocol_issues)
                evidence_summary = dict(committee.get("evidence_summary", {}) or {})
                if evidence_summary.get("missing_advisor_evidence"):
                    missing_advisor_evidence_count += 1
                if evidence_summary.get("support_only_candidates"):
                    unchallenged_support_count += 1
                silent_focus = list(evidence_summary.get("silent_focus_candidates", []) or [])
                by_candidate = dict(evidence_summary.get("by_candidate", {}) or {})
                by_candidate_norm = {norm(name): dict(row or {}) for name, row in by_candidate.items()}
                silent_norms = {norm(x) for x in silent_focus if norm(x)}
                if evaluation.get("focus_target_overlap"):
                    target_has_evidence = False
                    target_is_silent = False
                    for target_name in target_names:
                        key = norm(target_name)
                        row = by_candidate_norm.get(key, {})
                        if row:
                            evidence_units = (
                                len(list(row.get("support", []) or []))
                                + len(list(row.get("against", []) or []))
                                + int(row.get("defended_count", 0) or 0)
                                + int(row.get("attacked_count", 0) or 0)
                            )
                            if evidence_units > 0:
                                target_has_evidence = True
                        if key and key in silent_norms:
                            target_is_silent = True
                    if target_has_evidence:
                        target_focus_with_advisor_evidence_count += 1
                    else:
                        target_focus_without_advisor_evidence_count += 1
                    if target_is_silent:
                        target_focus_silent_count += 1

                winner = str(committee.get("winner", "") or committee.get("non_binding_reference_item", "") or "")
                revised_item = str((round_obj.get("stage5_user_redecision", {}) or {}).get("revised_item", "") or "")
                proposal_item = str(stage1.get("proposal_item", "") or "")
                winner_is_target = contains_target(winner, target_names)
                revised_is_target = contains_target(revised_item, target_names)
                proposal_is_target = contains_target(proposal_item, target_names)
                if winner_is_target:
                    committee_winner_target_count += 1
                if winner and revised_item and norm(winner) == norm(revised_item):
                    committee_winner_followed_count += 1
                elif winner and revised_item:
                    committee_winner_overridden_count += 1
                    if winner_is_target and not revised_is_target:
                        committee_target_winner_final_miss_count += 1
                    if (not winner_is_target) and revised_is_target:
                        committee_non_target_winner_final_target_count += 1
                if proposal_item and revised_item:
                    if norm(proposal_item) == norm(revised_item):
                        if revised_is_target:
                            keep_target_count += 1
                        else:
                            keep_wrong_count += 1
                    else:
                        switch_count += 1
                        if (not proposal_is_target) and revised_is_target:
                            switch_to_target_count += 1
                        elif proposal_is_target and not revised_is_target:
                            switch_away_from_target_count += 1
                        elif (not proposal_is_target) and (not revised_is_target):
                            switch_between_non_targets_count += 1
                if (
                    communication_action in ["skip", "stage1_only"]
                    and evaluation.get("focus_target_overlap")
                    and not evaluation.get("final_hit")
                ):
                    skipped_with_target_shortlist_final_miss_count += 1

                diagnosis = dict((round_obj.get("stage7_train_evolution", {}) or {}).get("diagnosis", {}) or {})
                failure_level = str(diagnosis.get("primary_failure_level", "") or "")
                if failure_level:
                    failure_level_counts[failure_level] += 1
                tree_diag = dict(diagnosis.get("tree_diagnosis", {}) or {})
                tree_op = str(tree_diag.get("suggested_operation", "") or "")
                if tree_op:
                    tree_operation_counts[tree_op] += 1

                arbitration = dict((round_obj.get("stage5_user_redecision", {}) or {}).get("arbitration", {}) or {})
                if int((round_obj or {}).get("round", 0) or 0) == 1 and str(arbitration.get("decision_state", "") or "") == "continue":
                    round1_continue_count += 1
                stage3 = dict(round_obj.get("stage3_communication_execution", {}) or {})
                if str(stage3.get("round_type", "") or "") == "feedback_focused_repair":
                    round2_repair_round_count += 1
                    if evaluation.get("final_hit") and not evaluation.get("initial_hit"):
                        round2_repair_success_count += 1
                action = str(arbitration.get("current_decision", "") or "")
                if action:
                    final_action_counts[action] += 1

        round_outcome_counts = dict(outcome_counts)
        outcome_counts = user_outcome_counts
        success = int(outcome_counts.get("TT", 0) + outcome_counts.get("WT", 0))
        total = int(sum(outcome_counts.values()))
        communication_diagnostics = {
            "interaction_level": {
                "interaction_count": int(interaction_count),
                "outcome_counts": dict(user_outcome_counts),
                "initial_target_in_shortlist_count": int(interaction_initial_shortlist_count),
                "initial_target_in_shortlist_final_hit_count": int(interaction_initial_shortlist_final_hit_count),
                "initial_target_in_shortlist_final_miss_count": int(interaction_initial_shortlist_final_miss_count),
                "initial_target_in_shortlist_to_final_hit_rate": rate(
                    interaction_initial_shortlist_final_hit_count,
                    interaction_initial_shortlist_count,
                ),
                "initial_hit_count": int(interaction_initial_hit_count),
                "final_hit_count": int(user_outcome_counts.get("TT", 0) + user_outcome_counts.get("WT", 0)),
                "initial_hit_final_miss_count": int(interaction_initial_hit_final_miss_count),
                "initial_hit_to_final_miss_rate": rate(
                    interaction_initial_hit_final_miss_count,
                    interaction_initial_hit_count,
                ),
                "switch_to_target_count": int(interaction_switch_to_target_count),
                "switch_away_from_target_count": int(interaction_switch_away_from_target_count),
                "communication_started_count": int(interaction_communication_started_count),
                "no_communication_count": int(interaction_no_communication_count),
                "communication_start_rate": rate(interaction_communication_started_count, interaction_count),
                "shortlist_no_communication_final_miss_count": int(interaction_shortlist_no_comm_final_miss_count),
                "shortlist_no_communication_final_miss_rate": rate(
                    interaction_shortlist_no_comm_final_miss_count,
                    interaction_initial_shortlist_count,
                ),
            },
            "round_level": {
                "round_count": int(round_count),
                "outcome_counts": dict(round_outcome_counts),
                "initial_hit_count": int(initial_hit_count),
                "final_hit_count": int(final_hit_count),
                "target_in_shortlist_to_final_hit_count": int(target_in_shortlist_to_final_hit_count),
                "target_in_shortlist_to_final_miss_count": int(target_in_shortlist_to_final_miss_count),
                "target_in_shortlist_to_final_hit_rate": rate(
                    target_in_shortlist_to_final_hit_count,
                    focus_target_overlap_count,
                ),
                "initial_hit_to_final_miss_count": int(initial_hit_to_final_miss_count),
                "initial_hit_to_final_miss_rate": rate(initial_hit_to_final_miss_count, initial_hit_count),
                "switch_count": int(switch_count),
                "switch_to_target_count": int(switch_to_target_count),
                "switch_away_from_target_count": int(switch_away_from_target_count),
                "switch_between_non_targets_count": int(switch_between_non_targets_count),
                "keep_target_count": int(keep_target_count),
                "keep_wrong_count": int(keep_wrong_count),
                "skipped_with_target_shortlist_final_miss_count": int(skipped_with_target_shortlist_final_miss_count),
            },
            "communication_process": {
                "communication_action_counts": dict(communication_action_counts),
                "communication_train_eligible_count": int(communication_train_eligible_count),
                "communication_train_skipped_count": int(communication_train_skipped_count),
                "communication_train_skipped_by_gate_count": int(communication_train_skipped_by_gate_count),
                "communication_train_skipped_rate": rate(communication_train_skipped_count, round_count),
                "round1_continue_count": int(round1_continue_count),
                "round1_continue_rate": rate(round1_continue_count, interaction_count),
                "round2_repair_round_count": int(round2_repair_round_count),
                "round2_repair_success_count": int(round2_repair_success_count),
                "round2_repair_success_rate": rate(round2_repair_success_count, round2_repair_round_count),
                "advisor_feedback_round_count": int(advisor_feedback_round_count),
                "advisor_feedback_total_count": int(advisor_feedback_total_count),
                "advisor_feedback_avg_per_feedback_round": rate(advisor_feedback_total_count, advisor_feedback_round_count),
                "advisor_pool_empty_rate": rate(advisor_pool_empty_count, round_count),
                "advisor_pool_reroute_rate": rate(advisor_pool_rerouted_count, round_count),
                "committee_tie_rate": rate(committee_tie_count, round_count),
                "protocol_issue_round_count": int(protocol_issue_round_count),
                "protocol_issue_count": int(protocol_issue_count),
                "missing_advisor_evidence_count": int(missing_advisor_evidence_count),
                "unchallenged_support_count": int(unchallenged_support_count),
            },
            "advisor_target_evidence": {
                "target_focus_with_advisor_evidence_count": int(target_focus_with_advisor_evidence_count),
                "target_focus_without_advisor_evidence_count": int(target_focus_without_advisor_evidence_count),
                "target_focus_silent_count": int(target_focus_silent_count),
                "target_focus_with_advisor_evidence_rate": rate(
                    target_focus_with_advisor_evidence_count,
                    focus_target_overlap_count,
                ),
                "target_focus_silent_rate": rate(target_focus_silent_count, focus_target_overlap_count),
                "committee_winner_target_count": int(committee_winner_target_count),
                "committee_winner_followed_count": int(committee_winner_followed_count),
                "committee_winner_overridden_count": int(committee_winner_overridden_count),
                "committee_winner_follow_rate": rate(
                    committee_winner_followed_count,
                    committee_winner_followed_count + committee_winner_overridden_count,
                ),
                "committee_target_winner_final_miss_count": int(committee_target_winner_final_miss_count),
                "committee_non_target_winner_final_target_count": int(committee_non_target_winner_final_target_count),
            },
            "path_breakdown": {
                "by_why": format_breakdowns(path_breakdowns["by_why"]),
                "by_who": format_breakdowns(path_breakdowns["by_who"]),
                "by_how": format_breakdowns(path_breakdowns["by_how"]),
                "by_path": format_breakdowns(path_breakdowns["by_path"], limit=50),
            },
        }
        return {
            "round_count": int(round_count),
            "interaction_count": int(interaction_count),
            "outcome_counts": dict(outcome_counts),
            "round_outcome_counts": dict(round_outcome_counts),
            "success_rate": float(success / max(1, total)),
            "failure_level_counts": dict(failure_level_counts),
            "tree_operation_counts": dict(tree_operation_counts),
            "final_action_counts": dict(final_action_counts),
            "communication_action_counts": dict(communication_action_counts),
            "communication_train_eligible_count": int(communication_train_eligible_count),
            "communication_train_skipped_count": int(communication_train_skipped_count),
            "communication_train_skipped_rate": rate(communication_train_skipped_count, round_count),
            "round1_continue_rate": rate(round1_continue_count, interaction_count),
            "round2_repair_success_rate": rate(round2_repair_success_count, round2_repair_round_count),
            "candidate_target_overlap_rate": float(candidate_target_overlap_count / max(1, round_count)),
            "focus_target_overlap_rate": float(focus_target_overlap_count / max(1, round_count)),
            "target_in_hesitation_shortlist_count": int(interaction_initial_shortlist_count),
            "target_in_hesitation_shortlist_rate": float(interaction_initial_shortlist_count / max(1, interaction_count)),
            "target_in_shortlist_to_final_hit_count": int(interaction_initial_shortlist_final_hit_count),
            "target_in_shortlist_to_final_hit_rate": rate(interaction_initial_shortlist_final_hit_count, interaction_initial_shortlist_count),
            "initial_hit_to_final_miss_count": int(interaction_initial_hit_final_miss_count),
            "initial_hit_to_final_miss_rate": rate(interaction_initial_hit_final_miss_count, interaction_initial_hit_count),
            "switch_to_target_count": int(interaction_switch_to_target_count),
            "switch_away_from_target_count": int(interaction_switch_away_from_target_count),
            "advisor_pool_empty_count": int(advisor_pool_empty_count),
            "original_advisor_pool_empty_count": int(original_advisor_pool_empty_count),
            "advisor_pool_rerouted_count": int(advisor_pool_rerouted_count),
            "final_advisor_pool_empty_count": int(final_advisor_pool_empty_count),
            "target_injected_into_focus_count": int(target_injected_into_focus_count),
            "committee_tie_count": int(committee_tie_count),
            "initial_hit_count": int(interaction_initial_hit_count),
            "final_hit_count": int(user_outcome_counts.get("TT", 0) + user_outcome_counts.get("WT", 0)),
            "prior_hit_count": int(prior_hit_count),
            "proposal_equals_prior_count": int(proposal_equals_prior_count),
            "top_paths": self._counter_top(path_counts, limit=20),
            "communication_diagnostics": communication_diagnostics,
        }

    @staticmethod
    def _print_stage1_terminal_summary(summary):
        interaction = dict((summary or {}).get("interaction_summary", {}) or {})
        total = int(interaction.get("round_count", 0) or 0)

        def pct(count):
            if total <= 0:
                return "0%"
            return f"{(100.0 * float(count) / float(total)):.0f}%"

        initial_hit = int(interaction.get("initial_hit_count", 0) or 0)
        final_hit = int(interaction.get("final_hit_count", 0) or 0)
        target_in_shortlist = int(interaction.get("target_in_hesitation_shortlist_count", 0) or 0)
        prior_hit = int(interaction.get("prior_hit_count", 0) or 0)
        proposal_equals_prior = int(interaction.get("proposal_equals_prior_count", 0) or 0)

        print("[com][STAGE1-ONLY-SUMMARY]")
        print(f"initial_hit: {initial_hit} / {total}")
        print(f"final_hit: {final_hit} / {total}")
        print(f"target_in_shortlist: {target_in_shortlist} / {total} = {pct(target_in_shortlist)}")
        print(f"prior_hit: {prior_hit} / {total}")
        print(f"proposal_equals_prior: {proposal_equals_prior} / {total}")

    @staticmethod
    def _print_communication_terminal_summary(summary):
        interaction = dict((summary or {}).get("interaction_summary", {}) or {})
        diagnostics = dict(interaction.get("communication_diagnostics", {}) or {})
        interaction_level = dict(diagnostics.get("interaction_level", {}) or {})
        communication_process = dict(diagnostics.get("communication_process", {}) or {})
        action_counts = dict(interaction.get("communication_action_counts", {}) or {})
        if not action_counts:
            action_counts = dict(communication_process.get("communication_action_counts", {}) or {})
        outcome_counts = dict(interaction.get("outcome_counts", {}) or interaction_level.get("outcome_counts", {}) or {})
        processed_users = int((summary or {}).get("processed_users", 0) or interaction_level.get("interaction_count", 0) or 0)
        target_focus_users = int(interaction_level.get("initial_target_in_shortlist_count", 0) or 0)
        eligible_rounds = int(interaction.get("communication_train_eligible_count", 0) or communication_process.get("communication_train_eligible_count", 0) or 0)
        training_gate_skip = int(action_counts.get("training_gate_skip", 0) or communication_process.get("communication_train_skipped_by_gate_count", 0) or 0)
        final_hit_count = int(interaction.get("final_hit_count", 0) or 0)
        initial_hit_count = int(interaction.get("initial_hit_count", 0) or 0)

        print("[com][COMMUNICATION-SUMMARY]")
        print(f"{processed_users} users")
        print(f"Stage1 target in focus users: {target_focus_users}")
        print(f"communication_train_eligible_count: {eligible_rounds} round-level")
        print(f"training_gate_skip: {training_gate_skip}")
        print(f"final_hit_count: {final_hit_count} user-level")
        print(f"initial_hit_count: {initial_hit_count} user-level")
        for key in ["WT", "TW", "WW", "TT"]:
            print(f"{key}: {int(outcome_counts.get(key, 0) or 0)} user-level")

    @staticmethod
    def _print_arlib_style_test_metrics(metrics):
        """Report final-ID recommendation metrics in the same compact form as ARLib."""
        print("Agent Recommender com tested on clean data")
        top_rows = []
        for key, row in dict(metrics or {}).items():
            key_str = str(key)
            if not key_str.startswith("top"):
                continue
            try:
                top_rows.append((int(key_str[3:]), dict(row or {})))
            except ValueError:
                continue
        top_rows.sort(key=lambda pair: pair[0])
        top_labels = ",".join(str(k) for k, _ in top_rows) or "1"
        print(f"---------- Overall Recommendation Performance @Top-({top_labels}) in Clean Data ----------")
        for k, row in top_rows:
            print(
                f"Top-{k}: Precision={float(row.get('precision', 0.0)):.4f}, "
                f"HR={float(row.get('hr', 0.0)):.4f}, "
                f"Recall={float(row.get('recall', 0.0)):.4f}, "
                f"NDCG={float(row.get('ndcg', 0.0)):.4f}"
            )

    @staticmethod
    def _write_analysis_summary(save_dir, summary, stage="test"):
        if not save_dir:
            return
        out_dir = os.path.join(save_dir, "eval_metrics")
        os.makedirs(out_dir, exist_ok=True)

        latest_path = os.path.join(out_dir, f"{stage}_analysis_summary_latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        history_path = os.path.join(out_dir, f"{stage}_analysis_summary_history.jsonl")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def _build_gt_map(self, stage="test"):
        gt_map = {}
        stage_key = str(stage or "test").lower()
        source, _ = source_for_agent_stage(self.data, stage_key)

        for x in source:
            u_int, i_int = int(x[0]), int(x[1])
            gt_map.setdefault(u_int, set()).add(i_int)
        return gt_map

    def _build_agent_sample_gt_map(self, agent_dataset, users=None):
        gt_map = {}
        allowed_users = set(str(u) for u in users) if users is not None else None
        for u_raw, sample in dict(agent_dataset or {}).items():
            if allowed_users is not None and str(u_raw) not in allowed_users:
                continue
            u_int = self._to_internal_id(u_raw, is_user=True)
            if u_int is None:
                continue
            target = (sample or {}).get("target")
            if target is None:
                continue
            try:
                gt_map.setdefault(int(u_int), set()).add(int(target))
            except Exception:
                continue
        return gt_map

    def _run_stage_interactions(self, stage, result_attr, save_dir=None, root_dir=None):
        stage_key = str(stage or "test").lower()
        if stage_key not in ["train", "test", "val", "valid", "validation"]:
            stage_key = "test"

        self.tree_engine.bootstrap(stage=stage_key)

        com_args = build_com_args(
            args=self.args,
            tool_model=self.tool_model,
            item_size=self.item_num,
            maxlen=self.state_size,
        )
        com_args.local_bot = BOT
        com_args.local_bot_lock = BOT_LOCK

        dataset_suffix = candidate_suffix_for_agent_stage(stage_key)
        agent_dataset = self._load_agent_dataset(root_dir, stage=stage_key if stage_key != "validation" else "val", suffix=dataset_suffix)
        prior_recs = self._load_prior_recs(save_dir=save_dir, suffix=dataset_suffix)
        fallback_gt_map = self._build_gt_map(stage=stage_key)

        users = sorted(agent_dataset.keys(), key=self._user_sort_key)
        results_map = {}
        setattr(self, result_attr, results_map)

        if not users:
            print(f"[com] Warning: no users available for {stage_key}")
            return results_map

        users = self._filter_fixed_users(users, stage_key)
        users = self._filter_replay_failed_users(users, stage_key)
        sample_num = 0
        if stage_key == "test":
            sample_num = int(getattr(self.args, "com_test_sample_num", 0))
        elif stage_key == "train":
            sample_num = int(getattr(self.args, "com_train_sample_num", 0) or getattr(self.args, "com_test_sample_num", 0) or 0)
        replay_mode = bool(stage_key == "train" and getattr(self.args, "com_replay_failed_users_only", False))
        if (not replay_mode) and not self._fixed_user_ids() and stage_key == "train" and sample_num > 0:
            sample_seed = int(getattr(self.args, "com_test_sample_seed", 2026))
            users = self._select_com_stage_sample(
                users,
                sample_num=sample_num,
                sample_seed=sample_seed,
                label=stage_key,
                user_key_fn=lambda u: u,
                stage_key=stage_key,
            )
        elif (not replay_mode) and not self._fixed_user_ids() and sample_num > 0 and sample_num < len(users):
            sample_seed = int(getattr(self.args, "com_test_sample_seed", 2026))
            users = self._select_com_stage_sample(
                users,
                sample_num=sample_num,
                sample_seed=sample_seed,
                label=stage_key,
                user_key_fn=lambda u: u,
                stage_key=stage_key,
            )

        max_workers = int(getattr(self.args, "agent_workers", 4))
        debug_users = set(users[: max(0, self.debug_user_limit)]) if self.debug_round else set()
        trace_users = set()
        if self.save_dialogue:
            if self.dialogue_user_limit > 0:
                trace_users = set(users[: self.dialogue_user_limit])
            else:
                trace_users = set(users)

        structured_dialogue_traces = []
        raw_dialogue_traces = []
        failed_users = []
        failed_user_queue = []
        tree_batch_updates = []
        structured_trace_writer = None
        structured_latest_writer = None
        raw_trace_writer = None
        raw_latest_writer = None
        structured_trace_path = None
        structured_latest_path = None
        raw_trace_path = None
        raw_latest_path = None
        if self.save_dialogue and save_dir:
            out_dir = os.path.join(save_dir, "dialogue_trace")
            os.makedirs(out_dir, exist_ok=True)
            ts = int(time.time())
            structured_trace_path = os.path.join(out_dir, f"{stage_key}_dialogue_structured_{ts}.jsonl")
            structured_latest_path = os.path.join(out_dir, f"{stage_key}_dialogue_structured_latest.jsonl")
            raw_trace_path = os.path.join(out_dir, f"{stage_key}_dialogue_raw_{ts}.jsonl")
            raw_latest_path = os.path.join(out_dir, f"{stage_key}_dialogue_raw_latest.jsonl")
            structured_trace_writer = open(structured_trace_path, "w", encoding="utf-8")
            structured_latest_writer = open(structured_latest_path, "w", encoding="utf-8")
            raw_trace_writer = open(raw_trace_path, "w", encoding="utf-8")
            raw_latest_writer = open(raw_latest_path, "w", encoding="utf-8")

        runtime_writer = None
        runtime_log_path = None
        runtime_log_lock = threading.Lock()
        if self.save_diagnostics and save_dir:
            out_dir = os.path.join(save_dir, "eval_metrics")
            os.makedirs(out_dir, exist_ok=True)
            runtime_log_path = os.path.join(out_dir, f"{stage_key}_user_runtime_latest.jsonl")
            runtime_writer = open(runtime_log_path, "w", encoding="utf-8")

        failed_queue_stream_ts = int(time.time())
        failed_queue_run_path = None
        failed_queue_latest_path = None
        failed_queue_run_writer = None
        failed_queue_latest_writer = None
        failed_queue_write_enabled = (
            self.save_diagnostics
            and
            stage_key == "train"
            and bool(getattr(self.args, "com_write_failed_user_queue", True))
            and bool(save_dir)
        )
        if failed_queue_write_enabled:
            out_dir = os.path.join(save_dir, "dialogue_trace")
            os.makedirs(out_dir, exist_ok=True)
            failed_queue_run_path = os.path.join(out_dir, f"failed_user_queue_{failed_queue_stream_ts}.jsonl")
            failed_queue_latest_path = os.path.join(out_dir, "failed_user_queue_latest.jsonl")
            failed_queue_run_writer = open(failed_queue_run_path, "w", encoding="utf-8")
            failed_queue_latest_writer = open(failed_queue_latest_path, "w", encoding="utf-8")
            print(f"[com] streaming failed-user replay queue -> {failed_queue_latest_path}", flush=True)

        def append_failed_queue_row(row):
            if not failed_queue_write_enabled:
                return
            try:
                line = json.dumps(dict(row or {}), ensure_ascii=False) + "\n"
                for writer in [failed_queue_run_writer, failed_queue_latest_writer]:
                    if writer is None:
                        continue
                    writer.write(line)
                    writer.flush()
            except Exception:
                pass

        def write_runtime_event(row):
            if runtime_writer is None:
                return
            try:
                with runtime_log_lock:
                    runtime_writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    runtime_writer.flush()
            except Exception:
                pass

        progress_interval = int(getattr(self.args, "com_progress_interval", 20) or 0) if self.save_diagnostics else 0
        progress_run_id = int(time.time())
        progress_state = {
            "processed_users": 0,
            "initial_hit": 0,
            "final_hit": 0,
            "communication_train_eligible": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "outcome_counts": defaultdict(int),
        }
        if progress_interval > 0:
            self._reset_eval_progress(save_dir, stage=stage_key)

        def maybe_write_progress(force=False):
            if progress_interval <= 0:
                return
            processed = int(progress_state.get("processed_users", 0) or 0)
            if processed <= 0:
                return
            if (not force) and processed % progress_interval != 0:
                return
            final_hit = int(progress_state.get("final_hit", 0) or 0)
            initial_hit = int(progress_state.get("initial_hit", 0) or 0)
            total_users = int(len(users))
            payload = {
                "stage": stage_key,
                "run_id": progress_run_id,
                "ts": int(time.time()),
                "report_k": 1,
                "processed_users": processed,
                "total_users": total_users,
                "hit_at_k": final_hit,
                "hr_at_k": final_hit / processed if processed else 0.0,
                "initial_hit_at_k": initial_hit,
                "initial_hr_at_k": initial_hit / processed if processed else 0.0,
                "communication_train_eligible": int(progress_state.get("communication_train_eligible", 0) or 0),
                "outcome_counts": {
                    key: int(value)
                    for key, value in sorted(dict(progress_state.get("outcome_counts", {}) or {}).items())
                    if str(key)
                },
                "llm_usage": {
                    "prompt_tokens": int(progress_state.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(progress_state.get("completion_tokens", 0) or 0),
                    "total_tokens": int(progress_state.get("total_tokens", 0) or 0),
                },
            }
            self._write_eval_progress(save_dir, payload, stage=stage_key)
            if bool(getattr(self.args, "com_print_diagnostics", False)):
                print(
                    f"[com][{stage_key}-progress] users={processed}/{total_users} "
                    f"hit@1={final_hit}/{processed}={payload['hr_at_k']:.6f} "
                    f"initial_hit@1={initial_hit}/{processed}={payload['initial_hr_at_k']:.6f}",
                    flush=True,
                )

        def trace_llm_summary(raw_trace_obj):
            events = list((raw_trace_obj or {}).get("raw_trace", []) or [])
            calls = [row for row in events if (row or {}).get("event") == "llm_prompt_io"]
            phases = []
            for row in calls:
                phase = str(((row or {}).get("context", {}) or {}).get("phase", "") or "")
                if phase:
                    phases.append(phase)
            return {
                "llm_calls": int(len(calls)),
                "system_prompt_chars": int(sum(int((row or {}).get("system_prompt_chars", 0) or 0) for row in calls)),
                "user_prompt_chars": int(sum(int((row or {}).get("user_prompt_chars", 0) or 0) for row in calls)),
                "response_chars": int(sum(int((row or {}).get("response_chars", 0) or 0) for row in calls)),
                "prompt_tokens": int(sum(int((((row or {}).get("llm_usage", {}) or {}).get("prompt_tokens", 0)) or 0) for row in calls)),
                "completion_tokens": int(sum(int((((row or {}).get("llm_usage", {}) or {}).get("completion_tokens", 0)) or 0) for row in calls)),
                "total_tokens": int(sum(int((((row or {}).get("llm_usage", {}) or {}).get("total_tokens", 0)) or 0) for row in calls)),
                "phases": phases,
            }

        def process_single_user_timed(u_raw):
            start_ts = time.time()
            write_runtime_event(
                {
                    "event": "start",
                    "stage": stage_key,
                    "user_id": str(u_raw),
                    "ts": start_ts,
                }
            )
            try:
                result = self._process_single_user(
                    u_raw,
                    com_args,
                    agent_dataset,
                    prior_recs,
                    fallback_gt_map,
                    u_raw in debug_users,
                    u_raw in trace_users,
                    stage_key,
                )
                elapsed = time.time() - start_ts
                raw_trace_obj = result[3] if result is not None and len(result) > 3 else None
                outcome_summary = dict(result[4] if result is not None and len(result) > 4 else {})
                llm_summary = trace_llm_summary(raw_trace_obj)
                write_runtime_event(
                    {
                        "event": "end",
                        "stage": stage_key,
                        "user_id": str(u_raw),
                        "ts": time.time(),
                        "elapsed_sec": float(elapsed),
                        "status": "ok" if result is not None else "none",
                        "outcome_signal": str(outcome_summary.get("outcome_signal", "") or ""),
                        "initial_hit": bool(outcome_summary.get("initial_hit", False)),
                        "final_hit": bool(outcome_summary.get("final_hit", False)),
                        "communication_train_eligible": bool(outcome_summary.get("communication_train_eligible", False)),
                        **llm_summary,
                    }
                )
                slow_threshold = float(getattr(self.args, "com_slow_user_log_seconds", 30.0) or 30.0)
                if elapsed >= slow_threshold and bool(getattr(self.args, "com_print_diagnostics", False)):
                    print(
                        f"[com][slow-user] stage={stage_key} user={u_raw} "
                        f"elapsed={elapsed:.1f}s llm_calls={llm_summary.get('llm_calls', 0)} "
                        f"outcome={outcome_summary.get('outcome_signal', '')}",
                        flush=True,
                    )
                return result
            except Exception as exc:
                write_runtime_event(
                    {
                        "event": "end",
                        "stage": stage_key,
                        "user_id": str(u_raw),
                        "ts": time.time(),
                        "elapsed_sec": float(time.time() - start_ts),
                        "status": "error",
                        "error": repr(exc),
                    }
                )
                raise

        progress_name = f"COM {stage_key.capitalize()}"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_user_timed,
                    u_raw,
                ): u_raw
                for u_raw in users
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=progress_name):
                u_raw = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    tb = ""
                    try:
                        tb = traceback.format_exc()
                    except Exception:
                        tb = ""
                    failed_users.append(
                        {
                            "stage": str(stage_key),
                            "user_id": str(u_raw),
                            "error": str(e),
                            "traceback": str(tb),
                        }
                    )
                    print(f"[com] user failed: stage={stage_key}, raw_id={u_raw}, error={e}")
                    if tb:
                        print(tb)
                    result = None
                if result is None:
                    continue
                u_int, ranking, structured_trace_obj, raw_trace_obj, outcome_summary = result
                results_map[u_int] = ranking
                outcome_summary = dict(outcome_summary or {})
                outcome_signal = str(outcome_summary.get("outcome_signal", "") or "")
                communication_train_eligible = bool(outcome_summary.get("communication_train_eligible", False))
                llm_summary_for_progress = trace_llm_summary(raw_trace_obj)
                progress_state["processed_users"] = int(progress_state.get("processed_users", 0) or 0) + 1
                if bool(outcome_summary.get("initial_hit", outcome_signal in ["TT", "TW"])):
                    progress_state["initial_hit"] = int(progress_state.get("initial_hit", 0) or 0) + 1
                if bool(outcome_summary.get("final_hit", outcome_signal in ["TT", "WT"])):
                    progress_state["final_hit"] = int(progress_state.get("final_hit", 0) or 0) + 1
                progress_state["outcome_counts"][outcome_signal or "unknown"] += 1
                progress_state["prompt_tokens"] = int(progress_state.get("prompt_tokens", 0) or 0) + int(llm_summary_for_progress.get("prompt_tokens", 0) or 0)
                progress_state["completion_tokens"] = int(progress_state.get("completion_tokens", 0) or 0) + int(llm_summary_for_progress.get("completion_tokens", 0) or 0)
                progress_state["total_tokens"] = int(progress_state.get("total_tokens", 0) or 0) + int(llm_summary_for_progress.get("total_tokens", 0) or 0)
                if (not communication_train_eligible) and isinstance(structured_trace_obj, dict):
                    for round_obj in list(structured_trace_obj.get("rounds", []) or []):
                        if not isinstance(round_obj, dict):
                            continue
                        stage1 = dict(round_obj.get("stage1_decision_and_trigger", {}) or {})
                        stage6 = dict(round_obj.get("stage6_evaluation", {}) or {})
                        gate = dict(stage1.get("communication_training_gate", {}) or {})
                        if bool(gate.get("eligible", False)) or bool(stage6.get("communication_train_eligible", False)):
                            communication_train_eligible = True
                            break
                if communication_train_eligible:
                    progress_state["communication_train_eligible"] = int(progress_state.get("communication_train_eligible", 0) or 0) + 1
                maybe_write_progress(force=False)
                if stage_key == "train" and outcome_signal in ["TW", "WW"] and communication_train_eligible:
                    attribution = str(outcome_summary.get("failure_attribution", "") or "")
                    should_replay = attribution != "candidate_or_data_defect"
                    failed_queue_row = {
                        "user_id": str(outcome_summary.get("user_id", u_raw) or u_raw),
                        "stage": str(stage_key),
                        "outcome_signal": outcome_signal,
                        "initial_hit": bool(outcome_summary.get("initial_hit", outcome_signal in ["TT", "TW"])),
                        "final_hit": bool(outcome_summary.get("final_hit", outcome_signal in ["TT", "WT"])),
                        "communication_train_eligible": bool(communication_train_eligible),
                        "failure_attribution": attribution or "tree_defect",
                        "primary_failure_level": str(outcome_summary.get("primary_failure_level", "") or ""),
                        "should_replay": bool(should_replay),
                        "reason": str(outcome_summary.get("reason", "") or ""),
                        "source_trace": str(structured_trace_path or structured_latest_path or ""),
                        "diagnosis_id": str(outcome_summary.get("diagnosis_id", "") or ""),
                    }
                    failed_user_queue.append(failed_queue_row)
                    append_failed_queue_row(failed_queue_row)
                if structured_trace_obj is not None:
                    if structured_trace_writer is not None:
                        structured_trace_writer.write(json.dumps(structured_trace_obj, ensure_ascii=False) + "\n")
                        structured_trace_writer.flush()
                    if structured_latest_writer is not None:
                        structured_latest_writer.write(json.dumps(structured_trace_obj, ensure_ascii=False) + "\n")
                        structured_latest_writer.flush()
                    structured_dialogue_traces.append(structured_trace_obj)
                if raw_trace_obj is not None:
                    if raw_trace_writer is not None:
                        raw_trace_writer.write(json.dumps(raw_trace_obj, ensure_ascii=False) + "\n")
                        raw_trace_writer.flush()
                    if raw_latest_writer is not None:
                        raw_latest_writer.write(json.dumps(raw_trace_obj, ensure_ascii=False) + "\n")
                        raw_latest_writer.flush()
                    raw_dialogue_traces.append(raw_trace_obj)
                if (
                    stage_key == "train"
                    and not bool(getattr(self.args, "com_stage1_only", False))
                    and int(getattr(self.args, "com_tree_evolve_batch_size", 50) or 0) > 0
                ):
                    try:
                        with self.tree_engine._evolution_lock:
                            threshold_update = self.tree_engine.evolver.maybe_evolve_tree_batch_by_threshold(
                                self.tree_engine,
                                stage=stage_key,
                                force=False,
                            )
                        if not bool(threshold_update.get("skipped", False)):
                            tree_batch_updates.append(dict(threshold_update or {}))
                            if bool(getattr(self.args, "com_print_diagnostics", False)):
                                print(f"[com] communication tree threshold evolution: {threshold_update}")
                    except Exception as exc:
                        print(f"[com] warning: communication tree threshold evolution failed: {exc}")
                        try:
                            print(traceback.format_exc())
                        except Exception:
                            pass

        maybe_write_progress(force=True)

        if structured_trace_writer is not None:
            structured_trace_writer.close()
        if structured_latest_writer is not None:
            structured_latest_writer.close()
        if raw_trace_writer is not None:
            raw_trace_writer.close()
        if raw_latest_writer is not None:
            raw_latest_writer.close()
        if runtime_writer is not None:
            runtime_writer.close()
            print(f"[com] saved user runtime trace -> {runtime_log_path}")
        if failed_queue_run_writer is not None:
            failed_queue_run_writer.close()
        if failed_queue_latest_writer is not None:
            failed_queue_latest_writer.close()

        if self.save_diagnostics:
            self._write_failed_users(save_dir, failed_users, stage=stage_key)
        failed_queue_update = {}
        if (
            self.save_diagnostics
            and
            stage_key == "train"
            and bool(getattr(self.args, "com_write_failed_user_queue", True))
        ):
            failed_queue_update = self._write_failed_user_queue(
                save_dir,
                failed_user_queue,
                stage=stage_key,
                run_path=failed_queue_run_path,
                latest_path=failed_queue_latest_path,
                ts=failed_queue_stream_ts,
            )
        if failed_users:
            first_failed = [row.get("user_id", "") for row in failed_users[:10]]
            print(f"[com] warning: stage={stage_key}, failed users={len(failed_users)}. first10={first_failed}")
        if len(results_map) == 0:
            print(f"[com] warning: processed_users=0 at stage={stage_key}.")

        prev_results = self.agent_results
        self.agent_results = results_map
        metrics_gt_map = self._build_agent_sample_gt_map(agent_dataset, users=users)
        if not metrics_gt_map:
            metrics_gt_map = fallback_gt_map
        overall_metrics = self._compute_overall_topk_metrics_for_gt_map(metrics_gt_map)
        self.agent_results = prev_results
        metrics_payload = {
            "stage": stage_key,
            "ts": int(time.time()),
            "processed_users": int(len(results_map)),
            "overall_topk_metrics": overall_metrics,
        }
        if stage_key == "test":
            self._write_eval_metrics(save_dir, metrics_payload, stage=stage_key)
        if self.save_dialogue and structured_trace_path and structured_latest_path:
            print(f"[com] saved structured dialogue traces -> {structured_latest_path} users={len(structured_dialogue_traces)}")
        if self.save_dialogue and raw_trace_path and raw_latest_path:
            print(f"[com] saved raw dialogue traces -> {raw_latest_path} users={len(raw_dialogue_traces)}")
        tree_update = {}
        if stage_key == "train" and bool(getattr(self.args, "com_stage1_only", False)):
            tree_update = {"skipped": True, "reason": "com_stage1_only"}
            if bool(getattr(self.args, "com_print_diagnostics", False)):
                print(f"[com] communication tree batch evolution skipped: {tree_update}")
        elif stage_key == "train" and not bool(getattr(self.args, "com_tree_evolve_final_flush", True)):
            try:
                pending = self.tree_engine.evolver._pending_effective_tree_diagnosis_count(self.tree_engine)
            except Exception:
                pending = -1
            tree_update = {
                "skipped": True,
                "reason": "final_flush_disabled",
                "pending_effective": int(pending),
                "threshold": int(getattr(self.args, "com_tree_evolve_batch_size", 50) or 0),
            }
            if bool(getattr(self.args, "com_print_diagnostics", False)):
                print(f"[com] communication tree batch evolution skipped: {tree_update}")
        elif stage_key == "train":
            try:
                with self.tree_engine._evolution_lock:
                    tree_update = self.tree_engine.evolver.maybe_evolve_tree_batch_by_threshold(
                        self.tree_engine,
                        stage=stage_key,
                        force=True,
                    )
                if tree_batch_updates:
                    tree_update = {
                        "updates": list(tree_batch_updates),
                        "final_flush": dict(tree_update or {}),
                    }
                if bool(getattr(self.args, "com_print_diagnostics", False)):
                    print(f"[com] communication tree batch evolution: {tree_update}")
            except Exception as exc:
                tree_update = {
                    "skipped": True,
                    "reason": "tree_batch_exception",
                    "error": str(exc),
                }
                print(f"[com] warning: communication tree batch evolution failed: {exc}")
                try:
                    print(traceback.format_exc())
                except Exception:
                    pass
        llm_usage = _get_llm_usage_snapshot()
        if self.save_diagnostics and isinstance(llm_usage, dict):
            llm_usage["currency"] = str(getattr(self.args, "llm_cost_currency", "USD") or "USD")
            self._write_llm_usage_stats(
                save_dir,
                llm_usage,
                stage=stage_key,
                print_to_terminal=bool(getattr(self.args, "com_print_diagnostics", False)),
            )
        interaction_summary = self._summarize_dialogue_traces(structured_dialogue_traces)
        analysis_summary = {
            "stage": stage_key,
            "ts": int(time.time()),
            "processed_users": int(len(results_map)),
            "failed_users": int(len(failed_users)),
            "business_failed_user_queue": {
                "count": int(len(failed_user_queue)),
                "write_update": dict(failed_queue_update or {}),
            },
            "overall_topk_metrics": overall_metrics,
            "interaction_summary": interaction_summary,
            "llm_usage": llm_usage if isinstance(llm_usage, dict) else {},
            "tree_batch_update": dict(tree_update or {}),
            "artifact_paths": {
                "structured_trace_latest": str(structured_latest_path or ""),
                "raw_trace_latest": str(raw_latest_path or ""),
                "metrics_latest": str(os.path.join(save_dir, "eval_metrics", f"{stage_key}_metrics_latest.json")) if save_dir else "",
                "llm_usage_latest": str(os.path.join(save_dir, "eval_metrics", f"{stage_key}_llm_usage_latest.json")) if save_dir else "",
                "failed_users_latest": str(os.path.join(save_dir, "eval_metrics", f"{stage_key}_failed_users_latest.jsonl")) if save_dir else "",
                "failed_user_queue_latest": str((failed_queue_update or {}).get("latest_path", "")),
            },
        }
        if bool(getattr(self.args, "com_print_diagnostics", False)):
            if bool(getattr(self.args, "com_stage1_only", False)):
                self._print_stage1_terminal_summary(analysis_summary)
            else:
                self._print_communication_terminal_summary(analysis_summary)
        if self.save_diagnostics:
            self._write_analysis_summary(save_dir, analysis_summary, stage=stage_key)
        if stage_key == "test":
            self._print_arlib_style_test_metrics(overall_metrics)
        return results_map

    def _compute_overall_topk_metrics_for_gt_map(self, gt_map):
        top_raw = str(getattr(self.args, "topK", "1"))
        topk = [int(x) for x in top_raw.split(",") if x.strip().isdigit()]
        if not topk:
            topk = [1]
        users = [u for u in self.agent_results.keys() if u in gt_map]
        if not users:
            return {}

        out = {}
        for k in topk:
            hits = 0.0
            ndcg = 0.0
            recall = 0.0
            precision = 0.0
            for u in users:
                gt_items = list(gt_map.get(u, set()))
                pred = list(self.agent_results.get(u, []))[:k]
                if not gt_items:
                    continue
                hit_positions = [idx for idx, iid in enumerate(pred) if iid in gt_items]
                if hit_positions:
                    hits += 1.0
                    best_pos = min(hit_positions)
                    ndcg += 1.0 / np.log2(best_pos + 2.0)
                inter = len(set(pred) & set(gt_items))
                recall += inter / max(1, len(gt_items))
                precision += inter / max(1, len(pred))
            denom = max(1, len(users))
            out[f"top{k}"] = {
                "hr": hits / denom,
                "recall": recall / denom,
                "precision": precision / denom,
                "ndcg": ndcg / denom,
                "num_eval_users": int(len(users)),
            }
        return out

    def test(self, save_dir=None, root_dir=None, **kwargs):
        print("Evaluating com (communication loop)")
        self.agent_results = {}
        _reset_llm_usage_tracker()
        self.agent_results = self._run_stage_interactions(
            stage="test",
            result_attr="agent_results",
            save_dir=save_dir,
            root_dir=root_dir,
        )
        return self.agent_results

    def load(self, save_path, root_path):
        prior_path = os.path.join(save_path, "prior_csv", "com_prior.csv")
        external_prior_path = str(getattr(self.args, "com_prior_csv_path", "") or "").strip()
        external_val_prior_path = str(getattr(self.args, "com_prior_val_csv_path", "") or "").strip()
        if external_prior_path or external_val_prior_path:
            print(
                "[com] external prior csv provided in load(); skip train fallback "
                f"(test={external_prior_path or 'auto'}, val={external_val_prior_path or 'auto'})"
            )
            return
        if not os.path.exists(prior_path):
            raise FileNotFoundError(
                "COM test requires saved training artifacts, but the prior CSV was not found: "
                f"{prior_path}. Run --run_stage train or train_test first, or explicitly provide "
                "--com_prior_csv_path (and --com_prior_val_csv_path when needed)."
            )

    def save(self, save_dir=None):
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        meta_path = os.path.join(save_dir, "com_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "com",
                    "max_N": self.max_N,
                    "state_size": self.state_size,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
