import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def recommend_parse_args(return_parser=False):
    """Arguments shared by the COM-only entry point and data loaders."""
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--dataset", default="librarything", help="processed dataset name")
    parser.add_argument("--data_path", default="data/clean/", help="processed data root")
    parser.add_argument("--training_data", default="/train.txt")
    parser.add_argument("--val_data", default="/val.txt")
    parser.add_argument("--test_data", default="/test.txt")
    parser.add_argument("--meta_file", default="/id2meta.json")
    parser.add_argument("--load_meta", type=str2bool, default=True, help="load item metadata")
    parser.add_argument("--split_data", type=str2bool, default=False, help="rebuild train/valid/test splits")
    parser.add_argument("--split_type", default="loo", help="split type retained for dataset preparation")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--model_type", default="agent")
    parser.add_argument("--model_name", default="com")
    parser.add_argument("--cuda", type=str2bool, default=True, help="use CUDA when available")
    parser.add_argument("--gpu_id", default="0", help="CUDA device id")
    parser.add_argument("--seed", type=int, default=2018)
    parser.add_argument("--topK", default="1")
    parser.add_argument(
        "--run_stage",
        default="auto",
        choices=["auto", "split", "train", "test", "train_test"],
        help="COM pipeline stage",
    )

    parser.add_argument("--load", type=str2bool, default=True)
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--save_dir", default="./modelsaved/", help="output directory")

    if return_parser:
        return parser
    return parser.parse_known_args()[0]
