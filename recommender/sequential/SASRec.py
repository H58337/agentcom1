# sequential/sasrec_recbole.py
import os
import glob
from typing import Any, Dict, Optional, Tuple, List, Union
import torch
import numpy as np
import pandas as pd

from recbole.quick_start import run_recbole, load_data_and_model


def _latest_checkpoint(checkpoint_dir: str, model_name: str = "SASRec") -> Optional[str]:
    """
    Return latest saved checkpoint path in checkpoint_dir for a given model name.
    """
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    pattern = os.path.join(checkpoint_dir, f"{model_name}-*.pth")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


class SASRec:
    """
    A thin wrapper around RecBole's training pipeline for SASRec:
      - train(): calls run_recbole(...) to train & evaluate [page:1]
      - load():  calls load_data_and_model(...) to reload config/model/dataset/dataloaders [page:2]

    Assumptions:
      - args.dataset exists and matches your atomic-file directory/prefix.
      - data_path is taken from args.data_datadir or data.data_datadir, and must be the parent directory
        that contains the {dataset}/ folder.
    """

    def __init__(self, args, data=None, config_file_list=None, dir=None ,agentargs=None):
        self.args = args
        self.agentargs = agentargs
        self.data = data
        self.config_file_list = config_file_list

        self.model_name = "SASRec"
        self.model_file: Optional[str] = None
        self.checkpoint_dir = dir
        # Store ARLib dataset reference for ID conversion
        self.arlib_dataset = data

    def _build_config_dict(self) -> dict:
        """
        Minimal config_dict; mainly for paths.
        Hyperparameters and evaluation settings should be provided via config_file_list (YAML).
        """
        config_dict = {
            "data_path": self.args.data_path,
            "checkpoint_dir": self.checkpoint_dir,
            "reproducibility": True,
            "show_progress": True,
            "MAX_ITEM_LIST_LENGTH": self.agentargs.MAX_ITEM_LIST_LENGTH,
        }

        # If seed is explicitly provided in args, we respect it
        if hasattr(self.args, "seed"):
            config_dict["seed"] = self.args.seed

        return config_dict

    def train(self, save_path) -> str:
        """
        Train+evaluate via run_recbole. RecBole will save the best model during training.
        Returns the latest SASRec checkpoint path.
        """
        dataset = getattr(self.args, "dataset", None)
        if not dataset:
            raise ValueError("args.dataset is required.")

        config_dict = self._build_config_dict()

        # 关键：接收 run_recbole 的返回结果（包含 valid/test 指标等）[page:0]
        results = run_recbole(
            model=self.model_name,
            dataset=dataset,
            config_file_list=self.config_file_list,
            config_dict=config_dict
        )

        ckpt = _latest_checkpoint(config_dict["checkpoint_dir"], model_name=self.model_name)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {config_dict['checkpoint_dir']} for {self.model_name}")

        if save_path:
            import shutil
            import os

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.move(ckpt, save_path)  # 重命名/移动

            # 新增：把推荐指标保存到同目录 txt
            metrics_path = os.path.join(
                os.path.dirname(save_path),
                f"{os.path.splitext(os.path.basename(save_path))[0]}_metrics.txt"
            )

            with open(metrics_path, "w", encoding="utf-8") as f:
                f.write(f"model={self.model_name}\n")
                f.write(f"dataset={dataset}\n")
                f.write(f"checkpoint={save_path}\n")
                f.write(f"best_valid_score={results.get('best_valid_score')}\n")          # [page:0]
                f.write(f"valid_score_bigger={results.get('valid_score_bigger')}\n")      # [page:0]

                f.write("\n[best_valid_result]\n")
                best_valid_result = results.get("best_valid_result", {})                  # [page:0]
                for k, v in best_valid_result.items():
                    f.write(f"{k}={v}\n")

                f.write("\n[test_result]\n")
                test_result = results.get("test_result", {})                              # [page:0]
                for k, v in test_result.items():
                    f.write(f"{k}={v}\n")

            self.model_file = save_path
            return save_path

        self.model_file = ckpt
        return ckpt


    def load(self, model_file: Optional[str] = None):
        """
        Load config/model/dataset/dataloaders from a saved checkpoint.
        Returns: self (SASRec instance with loaded RecBole model)
        """
        if model_file is None:
            model_file = self.model_file
        if not model_file:
            raise ValueError("model_file is not set. Call train() first or pass model_file explicitly.")

        # RecBole checkpoints include pickled config/dataset objects. PyTorch 2.6+
        # defaults torch.load(weights_only=True), which breaks these trusted local
        # checkpoints unless RecBole explicitly passes weights_only=False.
        original_torch_load = torch.load

        def _torch_load_recbole_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            if not torch.cuda.is_available():
                kwargs.setdefault("map_location", torch.device("cpu"))
            checkpoint = original_torch_load(*args, **kwargs)
            if not torch.cuda.is_available() and isinstance(checkpoint, dict) and "config" in checkpoint:
                try:
                    checkpoint["config"]["device"] = torch.device("cpu")
                except Exception:
                    pass
            return checkpoint

        torch.load = _torch_load_recbole_compat
        try:
            # Load RecBole model and store as internal attributes
            config, model, dataset,_,_,_ = load_data_and_model(model_file=model_file)
        finally:
            torch.load = original_torch_load
        
        self.recbole_config = config
        self.recbole_model = model
        self.recbole_dataset = dataset
        self.recbole_model.eval()
        
        # Move to device
        self.device = config['device']
        self.recbole_model.to(self.device)
        
        # Set ARLib dataset reference
        # Note: self.data may be None or from SASRec init, 
        # arlib_dataset will be set properly when used by AFL
        if self.data is not None:
            self.arlib_dataset = self.data
        
        return self  







    # -------------------------------------------------------------------------
    # Helper Method for Building User History Interaction
    # -------------------------------------------------------------------------
    
    def _arlib_user_to_raw(self, arlib_user_id: int) -> Optional[str]:
        """Convert ARLib internal user ID to raw user ID"""
        if not hasattr(self, 'arlib_dataset') or self.arlib_dataset is None:
            return None
        return self.arlib_dataset.id2user.get(arlib_user_id)
    
    def _arlib_item_to_raw(self, arlib_item_id: int) -> Optional[str]:
        """Convert ARLib internal item ID to raw item ID"""
        if not hasattr(self, 'arlib_dataset') or self.arlib_dataset is None:
            return None
        return self.arlib_dataset.id2item.get(arlib_item_id)
    
    def _raw_item_to_arlib(self, raw_item_id: str) -> Optional[int]:
        """Convert raw item ID to ARLib internal item ID"""
        if not hasattr(self, 'arlib_dataset') or self.arlib_dataset is None:
            return None
        return self.arlib_dataset.item.get(str(raw_item_id))
    
    
    
    def predict_with_sequence(self, item_sequence: Union[List[str], List[int]], 
                             candidate_ids: Union[List[str], List[int]]) -> dict:
        """
        Predict scores for candidates given an arbitrary item sequence.
        
        Supports two input modes:
        1. Raw ID mode (List[str]): Original dataset IDs as strings
        2. ARLib internal ID mode (List[int]): ARLib's internal integer IDs (0-based)
        
        Args:
            item_sequence: Sequence of items (raw IDs or ARLib internal IDs)
            candidate_ids: Candidate items (raw IDs or ARLib internal IDs)
            
        Returns:
            dict: {item_id: score} using same ID format as input
        """
        if not hasattr(self, 'recbole_model') or self.recbole_model is None:
            raise ValueError("Model not loaded. Please call load_tool() first.")
        
        dataset = self.recbole_dataset
        iid_field = dataset.iid_field
        
        # Auto-detect input format: raw ID (str) or ARLib internal ID (int)
        use_arlib_ids = False
        if item_sequence and isinstance(item_sequence[0], int):
            use_arlib_ids = True
        
        # Convert sequence to RecBole internal IDs
        seq_ids = []
        if use_arlib_ids:
            # ARLib internal IDs -> raw IDs -> RecBole internal IDs
            for arlib_iid in item_sequence:
                raw_iid = self._arlib_item_to_raw(arlib_iid)
                if raw_iid and str(raw_iid) in dataset.field2token_id[iid_field]:
                    seq_ids.append(dataset.field2token_id[iid_field][str(raw_iid)])
        else:
            # Raw IDs -> RecBole internal IDs
            for item_id in item_sequence:
                if str(item_id) in dataset.field2token_id[iid_field]:
                    seq_ids.append(dataset.field2token_id[iid_field][str(item_id)])
        
        if not seq_ids:
            return {}
        
        # Pad/truncate sequence
        max_len = self.recbole_config['MAX_ITEM_LIST_LENGTH']
        seq_len = len(seq_ids)
        
        if seq_len > max_len:
            seq_ids = seq_ids[-max_len:]
            seq_len = max_len
        else:
            seq_ids = seq_ids + [0] * (max_len - seq_len)
        
        # Build interaction and predict
        from recbole.data.interaction import Interaction
        # Use model's field names instead of hardcoded strings
        item_seq_field = self.recbole_model.ITEM_SEQ
        item_len_field = self.recbole_model.ITEM_SEQ_LEN
        interaction = Interaction({
            item_seq_field: torch.tensor([seq_ids]).to(self.device),
            item_len_field: torch.tensor([seq_len]).to(self.device)
        })
        
        with torch.no_grad():
            scores = self.recbole_model.full_sort_predict(interaction)[0].cpu().numpy()
        
        # Build result dict using input ID format
        result = {}
        if use_arlib_ids:
            # Return with ARLib internal IDs as keys
            for arlib_item_id in candidate_ids:
                raw_iid = self._arlib_item_to_raw(arlib_item_id)
                if raw_iid and str(raw_iid) in dataset.field2token_id[iid_field]:
                    recbole_iid = dataset.field2token_id[iid_field][str(raw_iid)]
                    if 0 <= recbole_iid < len(scores):
                        result[arlib_item_id] = float(scores[recbole_iid])
        else:
            # Return with raw IDs as keys
            for item_id in candidate_ids:
                if str(item_id) not in dataset.field2token_id[iid_field]:
                    continue
                recbole_iid = dataset.field2token_id[iid_field][str(item_id)]
                if 0 <= recbole_iid < len(scores):
                    result[item_id] = float(scores[recbole_iid])
        
        return result


    def recommend_topk_all_items(
        self,
        item_sequence: Union[List[int], List[str]],
        k: int = 20,
        filter_seen: bool = True,
        exclude_items: Optional[Union[List[int], List[str]]] = None,
        with_scores: bool = True,
        with_names: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        全量推荐 Top-K（不依赖候选集）

        输入:
        - item_sequence: 历史序列；支持 ARLib internal(int) 或 raw token(str)，会自动识别
        - exclude_items: 额外排除的物品列表；同样支持 int/str 自动识别
        输出: List[dict]，每个 dict 默认包含：
        - item_int: ARLib internal item id（若无法映射则为 None）
        - item_raw: raw item id/token（字符串）
        - score: 分数（可选）
        - name: 展示名（可选，依赖你已有 _get_item_display_name）
        """
        if not hasattr(self, 'recbole_model') or self.recbole_model is None:
            raise ValueError("Model not loaded. Please call load() first.")
        if not item_sequence:
            return []

        dataset = self.recbole_dataset
        iid_field = dataset.iid_field

        # RecBole: 0 is always padding for token-like features
        pad_id = dataset.field2token_id[iid_field].get("[PAD]", 0)

        def _to_recbole_internal_iids(ids: Union[List[int], List[str]]) -> List[int]:
            """(arlib int / raw str) -> recbole internal iid list"""
            if not ids:
                return []

            # auto-detect
            if isinstance(ids[0], int):
                out = []
                for arlib_iid in ids:
                    raw_iid = self._arlib_item_to_raw(int(arlib_iid))
                    if raw_iid is None:
                        continue
                    s = str(raw_iid)
                    if s in dataset.field2token_id[iid_field]:
                        out.append(int(dataset.field2token_id[iid_field][s]))
                return out

            # raw string ids
            out = []
            for raw_iid in ids:
                s = str(raw_iid)
                if s in dataset.field2token_id[iid_field]:
                    out.append(int(dataset.field2token_id[iid_field][s]))
            return out

        # 1) sequence -> recbole internal ids
        seq_iids = _to_recbole_internal_iids(item_sequence)
        if not seq_iids:
            return []

        # 2) pad / truncate
        max_len = int(self.recbole_config["MAX_ITEM_LIST_LENGTH"])
        seq_len = len(seq_iids)
        if seq_len > max_len:
            seq_iids = seq_iids[-max_len:]
            seq_len = max_len
        else:
            seq_iids = seq_iids + [pad_id] * (max_len - seq_len)

        # 3) build Interaction and full sort predict (scores for all items)
        from recbole.data.interaction import Interaction
        item_seq_field = self.recbole_model.ITEM_SEQ
        item_len_field = self.recbole_model.ITEM_SEQ_LEN

        interaction = Interaction({
            item_seq_field: torch.tensor([seq_iids], device=self.device),
            item_len_field: torch.tensor([seq_len], device=self.device)
        })

        with torch.no_grad():
            scores = self.recbole_model.full_sort_predict(interaction)[0]  # [n_items]

        # 4) filter
        scores = scores.clone()
        if 0 <= pad_id < scores.shape[0]:
            scores[pad_id] = -float("inf")

        if filter_seen:
            seen = {int(x) for x in seq_iids if int(x) != pad_id}
            if seen:
                idx = torch.tensor(list(seen), device=scores.device, dtype=torch.long)
                scores[idx] = -float("inf")

        if exclude_items:
            ex_iids = _to_recbole_internal_iids(exclude_items)
            if ex_iids:
                idx = torch.tensor(list(set(ex_iids)), device=scores.device, dtype=torch.long)
                scores[idx] = -float("inf")

        # 5) topk
        k_eff = max(1, min(int(k), int(scores.shape[0])))
        top_scores, top_iids = torch.topk(scores, k=k_eff)
        top_scores = top_scores.detach().cpu().tolist()
        top_iids = top_iids.detach().cpu().tolist()

        # 6) recbole internal iid -> raw token
        # Dataset.id2token maps internal ids to external tokens
        top_raw = dataset.id2token(iid_field, top_iids)

        # 7) pack results (同时给 raw 与 ARLib internal)
        results: List[Dict[str, Any]] = []
        for raw_iid, s in zip(top_raw, top_scores):
            raw_iid = str(raw_iid)
            arlib_iid = self._raw_item_to_arlib(raw_iid)

            row = {
                "item_raw": raw_iid,
                "item_int": int(arlib_iid) if arlib_iid is not None else None,
            }
            if with_scores:
                row["score"] = float(s)
            if with_names:
                # 你的 _get_item_display_name 看起来接收的是 internal id
                row["name"] = self._get_item_display_name(row["item_int"]) if row["item_int"] is not None else None
            results.append(row)

        return results

    def encode_user_sequence(self, item_sequence: Union[List[int], List[str]]) -> Optional[np.ndarray]:
        """Return the SASRec sequence representation used as a user embedding.

        COM uses this representation for the ``similar-users`` advisor source.
        It is computed from the user's observed interaction sequence only.
        """
        if not hasattr(self, "recbole_model") or self.recbole_model is None:
            raise ValueError("Model not loaded. Please call load() first.")
        if not item_sequence:
            return None

        dataset = self.recbole_dataset
        iid_field = dataset.iid_field
        use_arlib_ids = isinstance(item_sequence[0], int)
        seq_ids = []
        if use_arlib_ids:
            for arlib_iid in item_sequence:
                raw_iid = self._arlib_item_to_raw(int(arlib_iid))
                if raw_iid is not None and str(raw_iid) in dataset.field2token_id[iid_field]:
                    seq_ids.append(int(dataset.field2token_id[iid_field][str(raw_iid)]))
        else:
            for raw_iid in item_sequence:
                if str(raw_iid) in dataset.field2token_id[iid_field]:
                    seq_ids.append(int(dataset.field2token_id[iid_field][str(raw_iid)]))
        if not seq_ids:
            return None

        max_len = int(self.recbole_config["MAX_ITEM_LIST_LENGTH"])
        pad_id = int(dataset.field2token_id[iid_field].get("[PAD]", 0))
        seq_ids = seq_ids[-max_len:]
        seq_len = len(seq_ids)
        seq_ids = seq_ids + [pad_id] * (max_len - seq_len)

        from recbole.data.interaction import Interaction

        interaction = Interaction(
            {
                self.recbole_model.ITEM_SEQ: torch.tensor([seq_ids], device=self.device),
                self.recbole_model.ITEM_SEQ_LEN: torch.tensor([seq_len], device=self.device),
            }
        )
        with torch.no_grad():
            embedding = self.recbole_model.forward(
                interaction[self.recbole_model.ITEM_SEQ],
                interaction[self.recbole_model.ITEM_SEQ_LEN],
            )[0]
        return embedding.detach().float().cpu().numpy()
