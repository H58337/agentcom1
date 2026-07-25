import pandas as pd
import torch.utils.data as data

from recommender.agent.agent_sequence_split import source_for_agent_stage


class COMDataset(data.Dataset):
    def __init__(
        self,
        stage="test",
        sep=", ",
        arlib_dataset=None,
        candidates_map=None,
        state_size=50,
    ):
        if arlib_dataset is None:
            raise ValueError("arlib_dataset is required")

        self.arlib_dataset = arlib_dataset
        self.stage = stage
        self.sep = sep
        self.candidates_map = candidates_map if candidates_map is not None else {}
        self.state_size = int(state_size)

        # If candidates_map is provided, restrict dataset to those users.
        self.allowed_users_raw = set(self.candidates_map.keys()) if self.candidates_map else set()

        self.item_id2meta = self._get_meta_from_arlib()
        self.session_data = self._build_session_from_arlib()

    def __len__(self):
        return len(self.session_data)

    def __getitem__(self, idx):
        temp = self.session_data.iloc[idx]
        u_int = int(temp["user_id"])
        u_raw = str(self.arlib_dataset.id2user[u_int])
        target_iid = int(temp["next"])

        cands_internal = []
        cands_raw = self.candidates_map.get(u_raw, [])
        for c_raw in cands_raw:
            iid = self.arlib_dataset.item.get(str(c_raw))
            if iid is None and str(c_raw).isdigit():
                iid = self.arlib_dataset.item.get(int(c_raw))
            if iid is not None:
                cands_internal.append(int(iid))

        if not cands_internal:
            raise RuntimeError(
                f"Missing SASRec/external candidate set for user={u_raw}, stage={self.stage}. "
                "Generate SASRec candidates or provide --com_candidates_json_path."
            )
        target_injected = False
        candidate_source = "sasrec_or_external"

        seq_int = list(temp["seq_unpad"])
        seq_names = [self.item_id2meta.get(i, f"Item_{i}") for i in seq_int]
        cand_names = [self.item_id2meta.get(i, f"Item_{i}") for i in cands_internal]

        return {
            "user_id": u_int,
            "stage": str(temp.get("stage", self.stage)),
            "seq": seq_int,
            "len_seq": len(seq_int),
            "seq_str": self.sep.join(seq_names),
            "target": target_iid,
            "target_str": self.item_id2meta.get(target_iid, f"Item_{target_iid}"),
            "target_source": str(temp.get("target_source", self.stage)),
            "candidate_target_injected": bool(target_injected),
            "candidate_source": candidate_source,
            "original_test_target": int(temp.get("original_test_target", target_iid)),
            "original_test_target_str": self.item_id2meta.get(
                int(temp.get("original_test_target", target_iid)),
                f"Item_{int(temp.get('original_test_target', target_iid))}",
            ),
            "cans": cands_internal,
            "len_cans": len(cands_internal),
            "cans_str": self.sep.join(cand_names),
        }

    def _get_meta_from_arlib(self):
        meta_dict = {}
        for iid, raw_id in self.arlib_dataset.id2item.items():
            meta = self.arlib_dataset.get_item_meta(iid)
            if meta and "movie_title" in meta:
                meta_dict[iid] = str(meta["movie_title"])
            elif meta and "title" in meta:
                meta_dict[iid] = str(meta["title"])
            else:
                meta_dict[iid] = str(raw_id)
        return meta_dict

    def _build_session_from_arlib(self):
        rows = []
        source, target_source = source_for_agent_stage(self.arlib_dataset, self.stage)

        for entry in source:
            u_int, i_int = int(entry[0]), int(entry[1])
            if self.allowed_users_raw:
                u_raw = str(self.arlib_dataset.id2user.get(u_int))
                if u_raw not in self.allowed_users_raw:
                    continue
            full_history = list(entry[4])
            target_iid = i_int
            history = list(full_history)[-self.state_size :]
            if not history:
                continue
            rows.append(
                {
                    "user_id": u_int,
                    "stage": str(self.stage),
                    "seq_unpad": history,
                    "next": int(target_iid),
                    "original_test_target": int(i_int),
                    "target_source": target_source,
                }
            )

        print(f"[COMDataset] Built from ARLib {self.stage} set. Total: {len(rows)}")
        return pd.DataFrame(rows)
