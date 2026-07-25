import os
import json
import random
import collections


class BenchmarkSplitter:
    """
    默认留一法（LOO）：
    - 按 user 分组、按 timestamp 升序排序
    - valid：倒数第2条作为 target
    - test：最后1条作为 target
    - train：sliding-window 生成 (history -> target)
    - history 仅截断到 MAX_ITEM_LIST_LENGTH，不补 0
    - candidates.json 输出到：save_dir/agentname/dataset/candidates.json
    - [新增] 支持 filter_num：过滤交互少于该值的用户
    """

    def __init__(self, raw_loader, agent_args, rec_args, dataset, agentname, save_dir):
        self.raw_loader = raw_loader          # DataLoader_recbole_Raw(recommend_args)
        self.agent_args = agent_args          # agent_parse_args()
        self.rec_args = rec_args              # recommend_parse_args()

        self.dataset_dir = raw_loader.dataset_dir
        self.agentname = agentname
        self.save_dir = save_dir
        self.dataset = dataset

        ds = dataset
        self.train_path = os.path.join(self.dataset_dir, f"{ds}.train.inter")
        self.valid_path = os.path.join(self.dataset_dir, f"{ds}.valid.inter")
        self.test_path = os.path.join(self.dataset_dir, f"{ds}.test.inter")

        self.candidate_save_model_name = str(
            getattr(self.rec_args, "candidate_save_model_name", "") or ""
        ).strip()
        self.candidate_owner = self.candidate_save_model_name or self.agentname
        self.cand_dir = os.path.join(self.save_dir, self.candidate_owner, self.dataset)
        self.candidates_path = os.path.join(self.cand_dir, "candidates.json")

        # 分隔符/字段名：优先从 rec_args 拿
        self.field_sep = getattr(self.rec_args, "field_separator", "\t")
        self.seq_sep = getattr(self.rec_args, "seq_separator", " ")

        self.USER_ID_FIELD = getattr(self.rec_args, "USER_ID_FIELD", "user_id")
        self.ITEM_ID_FIELD = getattr(self.rec_args, "ITEM_ID_FIELD", "item_id")
        self.RATING_FIELD = getattr(self.rec_args, "RATING_FIELD", "rating")
        self.TIME_FIELD = getattr(self.rec_args, "TIME_FIELD", "timestamp")

        self.max_hist_len = int(getattr(self.rec_args, "MAX_ITEM_LIST_LENGTH", 50))
        self.SEQ_FIELD = f"{self.ITEM_ID_FIELD}_list"

        self.cand_num = 20
        self.seed = int(getattr(self.rec_args, "seed", 2025))
        self.candidate_generate_mode = str(
            getattr(self.rec_args, "candidate_generate_mode", "random")
        ).strip().lower()

    def _benchmark_files_exist(self):
        # 只有三份文件都存在才算“已划分”
        return all(os.path.isfile(p) for p in [self.train_path, self.valid_path, self.test_path])

    @staticmethod
    def _parse_user_limit(raw):
        if raw is None:
            return None
        s = str(raw).strip().lower()
        if s in ("", "all", "full", "none", "-1", "0"):
            return None
        try:
            n = int(float(s))
            return n if n > 0 else None
        except Exception:
            return None

    def split_and_save(self, overwrite=None, generate_candidates=True):
        files_exist = self._benchmark_files_exist()
        force_split = bool(getattr(self.rec_args, "split_data", False))
        need_split = (not files_exist) or force_split

        if not need_split:
            print("[BenchmarkSplitter] Benchmark files already exist and split_data=False, skip splitting.")
            if generate_candidates and str(getattr(self.rec_args, "run_stage", "")).lower() == "split":
                print("[BenchmarkSplitter] run_stage=split, regenerate candidates from existing benchmark files.")
                valid_rows = self._load_saved_inter(self.valid_path)
                test_rows = self._load_saved_inter(self.test_path)
                user_interactions = self._raw_user_interactions()
                self._save_candidates_json(test_rows, valid_rows, user_interactions)
            return

        reason = "missing benchmark files" if (not files_exist) else "split_data=True"
        print(f"[BenchmarkSplitter] Splitting triggered ({reason}).")

        filter_num = int(getattr(self.agent_args, "MIN_ITEM_LIST_LENGTH", 5))
        if filter_num < 3:
            if filter_num > 0:
                print(f"[BenchmarkSplitter] Warning: filter_num={filter_num} is too small for LOO. Auto-adjusted to 3.")
            filter_num = 3

        # 从 agent args 读取 val/test 用户数量限制（兼容 rec_args 回退）
        val_raw = getattr(self.agent_args, "val_user_num", getattr(self.rec_args, "val_user_num", "all"))
        test_raw = getattr(self.agent_args, "test_user_num", getattr(self.rec_args, "test_user_num", "all"))
        val_num = self._parse_user_limit(val_raw)
        test_num = self._parse_user_limit(test_raw)

        print(f"[BenchmarkSplitter] Filter Threshold: Users with < {filter_num} interactions will be dropped.")
        if val_num is None:
            print("[BenchmarkSplitter] Val User Limit: all")
        else:
            print(f"[BenchmarkSplitter] Val User Limit: {val_num}")
        if test_num is not None:
            print(f"[BenchmarkSplitter] Test User Limit: {test_num}")
        else:
            print("[BenchmarkSplitter] Test User Limit: all")

        data = self.raw_loader.inter_data
        if not data:
            raise RuntimeError("[BenchmarkSplitter] No interaction data loaded from raw .inter")

        user_interactions = collections.defaultdict(list)
        for (u, i, r, t) in data:
            user_interactions[u].append((u, i, float(r), float(t)))

        # 新增：先得到通过 filter 的用户列表，然后抽样 eval_users
        rng = random.Random(self.seed)
        eligible_users = [u for u, inters in user_interactions.items() if len(inters) >= filter_num]
        total_users = len(user_interactions)
        kept_users = len(eligible_users)
        dropped_users = total_users - kept_users
        total_interactions = len(data)
        kept_interactions = sum(len(user_interactions[u]) for u in eligible_users)
        dropped_interactions = total_interactions - kept_interactions

        users_shuffled = list(eligible_users)
        rng.shuffle(users_shuffled)

        if test_num is None or test_num >= kept_users:
            test_eval_users = set(eligible_users)
        else:
            test_eval_users = set(users_shuffled[:test_num])

        if val_num is None or val_num >= kept_users:
            val_eval_users = set(eligible_users)
        else:
            # Prefer overlap with test users when both limits are finite.
            test_order = [u for u in users_shuffled if u in test_eval_users]
            if len(test_order) >= val_num:
                val_eval_users = set(test_order[:val_num])
            else:
                val_eval_users = set(test_order)
                need = val_num - len(test_order)
                remain = [u for u in users_shuffled if u not in test_eval_users]
                val_eval_users.update(remain[:need])
                print(
                    f"[BenchmarkSplitter] Info: val({val_num}) > test overlap pool({len(test_order)}); "
                    f"filled remaining {max(0, need)} users from non-test set."
                )

        train_rows, valid_rows, test_rows = [], [], []

        # 遍历用户进行划分
        for u, inters in user_interactions.items():
            if len(inters) < filter_num:
                continue

            inters.sort(key=lambda x: x[3])  # timestamp asc
            seq_items = [x[1] for x in inters]

            test_idx = len(inters) - 1
            valid_idx = len(inters) - 2
            train_last_target_idx = len(inters) - 3

            # train: sliding window（保留你原逻辑：对所有 kept 用户都生成）
            for k in range(1, train_last_target_idx + 1):
                target = inters[k]
                hist = seq_items[:k]
                if self.max_hist_len > 0:
                    hist = hist[-self.max_hist_len:]
                if len(hist) == 0:
                    continue
                train_rows.append((u, target[1], target[2], target[3], hist))

            if u in val_eval_users:
                valid_target = inters[valid_idx]
                valid_hist = seq_items[:valid_idx]
                if self.max_hist_len > 0:
                    valid_hist = valid_hist[-self.max_hist_len:]
                valid_rows.append((u, valid_target[1], valid_target[2], valid_target[3], valid_hist))

            if u in test_eval_users:
                test_target = inters[test_idx]
                test_hist = seq_items[:test_idx]
                if self.max_hist_len > 0:
                    test_hist = test_hist[-self.max_hist_len:]
                test_rows.append((u, test_target[1], test_target[2], test_target[3], test_hist))

        print(f"[BenchmarkSplitter] Total Users: {total_users}")
        print(f"[BenchmarkSplitter] Total Interactions: {total_interactions}")
        print(f"[BenchmarkSplitter] Filtered (<{filter_num}): {dropped_users} users dropped.")
        print(f"[BenchmarkSplitter] Filtered Interactions: {dropped_interactions}")
        print(f"[BenchmarkSplitter] Kept: {kept_users} users.")
        print(f"[BenchmarkSplitter] Kept Interactions: {kept_interactions}")
        overlap_users = len(val_eval_users.intersection(test_eval_users))
        print(f"[BenchmarkSplitter] Eval Users (valid): {len(val_eval_users)}")
        print(f"[BenchmarkSplitter] Eval Users (test): {len(test_eval_users)}")
        print(f"[BenchmarkSplitter] Eval Users Overlap (val∩test): {overlap_users}")
        print(f"[BenchmarkSplitter] Train Samples: {len(train_rows)}, Valid: {len(valid_rows)}, Test: {len(test_rows)}")

        self._save_inter(self.train_path, train_rows)
        self._save_inter(self.valid_path, valid_rows)
        self._save_inter(self.test_path, test_rows)

        if generate_candidates:
            self._save_candidates_json(test_rows, valid_rows, user_interactions)


    def _save_inter(self, path, rows):
        header = [
            f"{self.USER_ID_FIELD}:token",
            f"{self.ITEM_ID_FIELD}:token",
            f"{self.RATING_FIELD}:float",
            f"{self.TIME_FIELD}:float",
            f"{self.SEQ_FIELD}:token_seq",
        ]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.field_sep.join(header) + "\n")
            for (u, i, r, t, hist) in rows:
                hist_str = self.seq_sep.join(str(x) for x in hist)
                f.write(self.field_sep.join([str(u), str(i), str(float(r)), str(float(t)), hist_str]) + "\n")
        print(f"Saved {path}")

    def _load_saved_inter(self, path):
        rows = []
        with open(path, "r", encoding="utf-8-sig") as f:
            header = f.readline().rstrip("\n").split(self.field_sep)
            fields = [h.split(":")[0] for h in header]
            user_idx = fields.index(self.USER_ID_FIELD) if self.USER_ID_FIELD in fields else 0
            item_idx = fields.index(self.ITEM_ID_FIELD) if self.ITEM_ID_FIELD in fields else 1
            rating_idx = fields.index(self.RATING_FIELD) if self.RATING_FIELD in fields else None
            time_idx = fields.index(self.TIME_FIELD) if self.TIME_FIELD in fields else None
            seq_idx = fields.index(self.SEQ_FIELD) if self.SEQ_FIELD in fields else None
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(self.field_sep)
                if len(parts) <= max(user_idx, item_idx):
                    continue
                u = parts[user_idx]
                i = parts[item_idx]
                r = float(parts[rating_idx]) if rating_idx is not None and parts[rating_idx] else 1.0
                t = float(parts[time_idx]) if time_idx is not None and time_idx < len(parts) and parts[time_idx] else 0.0
                hist = []
                if seq_idx is not None and seq_idx < len(parts) and parts[seq_idx].strip():
                    hist = [x for x in parts[seq_idx].split(self.seq_sep) if x]
                rows.append((u, i, r, t, hist))
        return rows

    def _raw_user_interactions(self):
        user_interactions = collections.defaultdict(list)
        for (u, i, r, t) in self.raw_loader.inter_data:
            user_interactions[u].append((u, i, float(r), float(t)))
        return user_interactions

    def _save_candidates_json(self, *args, **kwargs):
        raise RuntimeError(
            "Candidate generation belongs to an external base recommender in this COM-only release. "
            "Pass --com_candidates_json_path or enable --com_standalone_data_candidates."
        )
