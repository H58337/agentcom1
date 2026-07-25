"""COM-only entry point.

This release intentionally excludes ARLib's attack framework and unrelated
recommenders. It supports COM training, testing, and benchmark preparation.
"""

import os
import random
import sys

import numpy as np
import torch

from conf.agent_parser import agent_parse_args
from conf.recommend_parser import recommend_parse_args
from recommender.agent.com import com
from recommender.sequential.SASRec import SASRec
from util.DataLoader import DataLoader_recbole, DataLoader_recbole_Raw
from util.split_data import BenchmarkSplitter


def _merge_args():
    args = recommend_parse_args()
    agent_args = agent_parse_args()
    for key, value in vars(agent_args).items():
        setattr(args, key, value)
    args.model_name = "com"
    args.model_type = "agent"
    args.agent = True
    return args, agent_args


def _seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_data(args, agent_args, run_stage):
    dataset_dir = os.path.join(str(args.data_path), str(args.dataset))
    split_paths = [
        os.path.join(dataset_dir, f"{args.dataset}.{split}.inter")
        for split in ("train", "valid", "test")
    ]
    need_prepare = (
        run_stage == "split"
        or bool(getattr(args, "split_data", False))
        or not all(os.path.exists(path) for path in split_paths)
    )
    if need_prepare:
        raw_loader = DataLoader_recbole_Raw(args)
        splitter = BenchmarkSplitter(
            raw_loader=raw_loader,
            agent_args=agent_args,
            rec_args=args,
            dataset=args.dataset,
            agentname="com",
            save_dir=args.save_dir,
        )
        # SASRec is trained after the benchmark split is available. Candidate
        # generation then uses that trained backbone inside COM.
        splitter.split_and_save(generate_candidates=False)
        print("[COM] split complete: SASRec candidates will be generated after the backbone is available.")
        if run_stage == "split":
            return None
    return DataLoader_recbole(args)


def _sasrec_checkpoint_path(args):
    explicit = str(getattr(args, "sasrec_checkpoint_path", "") or "").strip()
    if explicit:
        return explicit
    return os.path.join(
        str(args.save_dir),
        "SASRec",
        str(args.dataset),
        "clean",
        "tool",
        "SASRec_model.pth",
    )


def _load_sasrec_backbone(args, agent_args, data, run_stage):
    checkpoint_path = _sasrec_checkpoint_path(args)
    config_path = str(getattr(args, "sasrec_config_file_path", "") or "").strip()
    if not config_path:
        config_path = os.path.join("tool_conf", "default", "SASRec.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SASRec config not found: {config_path}")

    checkpoint_dir = os.path.dirname(checkpoint_path)
    backbone = SASRec(
        args,
        data=data,
        config_file_list=[config_path],
        dir=checkpoint_dir,
        agentargs=agent_args,
    )
    should_train = bool(getattr(args, "sasrec_force_retrain", False)) or not os.path.exists(checkpoint_path)
    if should_train:
        if run_stage == "test":
            raise FileNotFoundError(
                f"SASRec checkpoint not found for test stage: {checkpoint_path}. "
                "Run --run_stage train or train_test first."
            )
        print(f"[COM] training SASRec backbone -> {checkpoint_path}")
        backbone.train(checkpoint_path)
    else:
        print(f"[COM] loading SASRec backbone -> {checkpoint_path}")
    backbone.load(checkpoint_path)
    backbone.arlib_dataset = data
    return backbone


def main():
    if any(flag in sys.argv[1:] for flag in ("-h", "--help")):
        print("COM-only entry point\n")
        recommend_parse_args(return_parser=True).print_help()
        print("\nCOM and LLM options:\n")
        agent_parse_args(return_parser=True).print_help()
        return

    args, agent_args = _merge_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(args, "gpu_id", "0"))
    _seed_everything(getattr(args, "seed", 2026))

    run_stage = str(getattr(args, "run_stage", "train") or "train").lower()
    if run_stage == "auto":
        run_stage = "train"
    data = _prepare_data(args, agent_args, run_stage)
    if data is None:
        return

    model = com(args, data)
    model.tool_model = _load_sasrec_backbone(args, agent_args, data, run_stage)
    save_dir = os.path.join(args.save_dir, "com", args.dataset, "clean")
    root_dir = os.path.join(args.save_dir, "com", args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    if run_stage == "train":
        model.train(save_dir=save_dir, root_dir=root_dir)
        if bool(getattr(args, "save", True)):
            model.save(save_dir)
    elif run_stage == "test":
        model.load(save_dir, root_dir)
        model.test(save_dir=save_dir, root_dir=root_dir)
    elif run_stage == "train_test":
        model.train(save_dir=save_dir, root_dir=root_dir)
        if bool(getattr(args, "save", True)):
            model.save(save_dir)
        model.test(save_dir=save_dir, root_dir=root_dir)
    else:
        raise ValueError("--run_stage must be split, train, test, or train_test")


if __name__ == "__main__":
    main()
