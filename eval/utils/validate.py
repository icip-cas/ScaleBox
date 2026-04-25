import os


MODE_FIELDS = ("use_server", "use_ray")


def validate_mode_args(args):
    """中文：校验采样模式参数及路径必填项。
    English: Validate sampling mode arguments and required path arguments.
    """
    if args.sample_only and args.eval_only:
        raise ValueError("`--sample_only` and `--eval_only` cannot be used together.")

    if args.eval_only:
        if not args.benchmark_data_path:
            raise ValueError("`--benchmark_data_path` is required when using `--eval_only`.")
        if not args.eval_path:
            raise ValueError("`--eval_path` is required when using `--eval_only`.")
        return

    if not args.benchmark_data_path:
        raise ValueError("`--benchmark_data_path` is required.")
    if args.eval_path:
        raise ValueError("`--eval_path` can only be used with `--eval_only`.")

    enabled_modes = [field for field in MODE_FIELDS if getattr(args, field, False)]
    if len(enabled_modes) != 1:
        raise ValueError("Exactly one of --use_server or --use_ray must be set.")

    if args.use_server and not args.model_path:
        raise ValueError("`--model_path` is required when using `--use_server`.")
    if args.use_ray and not args.model_path:
        raise ValueError("`--model_path` is required when using `--use_ray`.")


def validate_benchmark_data_path(benchmark_data_path, benchmark):
    if not os.path.exists(benchmark_data_path):
        raise FileNotFoundError(f"Benchmark data path does not exist: {benchmark_data_path}")
    if benchmark in {"livecodebench", "aethercode"}:
        if not os.path.isdir(benchmark_data_path):
            raise ValueError(f"Benchmark `{benchmark}` expects `--benchmark_data_path` to be a directory")
    else:
        if not os.path.isfile(benchmark_data_path):
            raise ValueError(f"Benchmark `{benchmark}` expects `--benchmark_data_path` to be a JSONL file")


def validate_eval_path(eval_path):
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Eval path does not exist: {eval_path}")
    if not os.path.isfile(eval_path):
        raise ValueError("`--eval_path` must point to a JSONL file with id/prompt/response")


def validate_resume_sample_args(args):
    if args.resume_sample and args.eval_only:
        raise ValueError("`--resume_sample` cannot be used with `--eval_only`.")
    if args.resume_sample and not args.resume_sample_path:
        raise ValueError("`--resume_sample_path` is required when using `--resume_sample`.")
    if args.resume_sample_path and not args.resume_sample:
        raise ValueError("`--resume_sample_path` can only be used with `--resume_sample`.")
    if args.resume_sample_path:
        if not os.path.exists(args.resume_sample_path):
            raise FileNotFoundError(f"Resume sample path does not exist: {args.resume_sample_path}")
        if not os.path.isfile(args.resume_sample_path):
            raise ValueError("`--resume_sample_path` must point to a JSONL file.")


def validate_thinking(prompt_type, thinking):
    """中文：校验 thinking 只用于 qwen3。
    English: Validate that thinking is only used with qwen3.
    """
    if thinking and prompt_type != "qwen3":
        raise ValueError("`--thinking` is only supported when `--prompt_type=qwen3`.")
