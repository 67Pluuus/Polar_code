"""Explicit stage CLI. Non-model stages import no inference implementation eagerly."""

import argparse
import math
import os
from pathlib import Path

from .storage import relative_path, run_lock, stage_dir


MODELS = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen1.5-MoE-A2.7B-Chat",
          "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3-8B"]


def parser():
    result = argparse.ArgumentParser(description="Offline PoLar MCTS supervision stages")
    commands = result.add_subparsers(dest="stage", required=True)
    for name in ("environment", "prepare", "search", "merge", "validate"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-name", required=True)
        sub.add_argument("--clean", action="store_true", help="Explicitly clear ONLY this stage's run directory")
        if name != "search":
            sub.add_argument("--clean-only", action="store_true", help="With --clean, clear this stage then exit")
        if name in {"environment", "prepare"}:
            sub.add_argument("--data-path", required=True)
        if name in {"environment", "search"}:
            sub.add_argument("--model-path", required=True, help="Complete local snapshot, relative to project root")
        if name == "prepare":
            sub.add_argument("--data-source", required=True, choices=["hkust-nlp/dart-math-pool-math"])
            sub.add_argument("--source-revision", required=True, help="Record the downloaded dataset revision/tag")
            sub.add_argument("--difficulties", type=int, nargs="+", required=True, choices=range(1, 6))
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--split-policy", choices=["proportional", "official"], required=True)
            sub.add_argument("--train-fraction", type=float, required=True)
            sub.add_argument("--validation-fraction", type=float, required=True)
            sub.add_argument("--max-questions-per-diff", type=int, required=True, help="0 uses all available unique questions")
        if name == "search":
            sub.add_argument("--model-id", required=True, choices=MODELS)
            sub.add_argument("--model-revision", required=True)
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--simulations", type=int, required=True)
            sub.add_argument("--exploration", type=float, required=True)
            sub.add_argument("--length-penalty", type=float, required=True)
            sub.add_argument("--max-block", type=int, required=True, choices=range(1, 5))
            sub.add_argument("--max-repeats", type=int, required=True, choices=range(1, 5))
            sub.add_argument("--max-length-factor", type=float, required=True)
            sub.add_argument("--max-new-tokens", type=int, required=True)
            sub.add_argument("--temperature", type=float, required=True)
            sub.add_argument("--completion-timeout", type=int, required=True)
    return result


def validate_args(args):
    if not Path("PoLar_code/polar/data.py").is_file():
        raise ValueError("Run from the project root containing ./PoLar_code and ./Polar_data")
    stage_dir(args.run_name, {"prepare": "prepared", "merge": "merged",
                              "validate": "validation"}.get(args.stage, args.stage))
    for key in ("data_path", "model_path"):
        if hasattr(args, key):
            relative_path(getattr(args, key))
    if args.stage == "prepare":
        if len(set(args.difficulties)) != len(args.difficulties):
            raise ValueError("Duplicate difficulty arguments")
        args.difficulties = sorted(args.difficulties)
        if (not 0 < args.train_fraction < 1 or not 0 < args.validation_fraction < 1
                or args.train_fraction + args.validation_fraction >= 1 or args.max_questions_per_diff < 0):
            raise ValueError("Invalid split fractions or question limit")
    if args.stage == "search":
        if min(args.simulations, args.max_new_tokens, args.completion_timeout) <= 0:
            raise ValueError("Simulations, generated tokens, and timeout must be positive")
        if any(not math.isfinite(v) or v < 0 for v in (args.exploration, args.length_penalty, args.temperature)):
            raise ValueError("UCB coefficients and temperature must be finite and nonnegative")
        if not math.isfinite(args.max_length_factor) or args.max_length_factor < 1:
            raise ValueError("Max program length must include the baseline (factor >= 1)")


def main():
    args = parser().parse_args()
    validate_args(args)
    from .environment import configure_runtime
    configure_runtime(args.run_name)
    if args.stage == "search":
        from .distributed_search import distributed_search
        distributed_search(args)
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Only search accepts multi-rank torchrun; other stages are single-process")
    with run_lock(args.run_name):
        if args.clean_only:
            if not args.clean:
                raise ValueError("--clean-only requires explicit --clean")
            from .storage import clean_stage
            clean_stage(args.run_name, {"prepare": "prepared", "merge": "merged",
                                       "validate": "validation"}.get(args.stage, args.stage))
            return
        if args.stage == "environment":
            from .environment import check_environment
            check_environment(args)
        elif args.stage == "prepare":
            from .prepare import prepare
            prepare(args)
        elif args.stage == "merge":
            from .merge import merge
            merge(args)
        elif args.stage == "validate":
            from .validate import validate
            validate(args)
