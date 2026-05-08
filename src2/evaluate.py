import os
import random

import numpy as np
import torch

from src.config import build_eval_arg_parser
from src.engine import AeroEvaluator
from src.utils import ensure_output_dirs


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = build_eval_arg_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.eval_gpu
    set_random_seed(args.seed)

    output_dirs = ensure_output_dirs(args.save_dir)
    evaluator = AeroEvaluator(args=args, output_dirs=output_dirs)
    evaluator.evaluate_checkpoint()


if __name__ == "__main__":
    main()
