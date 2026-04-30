import yaml

SUPPORTED_BENCHMARKS = {
    "mbpp",
    "mbppplus",
    "humaneval",
    "humanevalplus",
    "livecodebench",
    "aethercode",
    "multipl_e",
}

OLD_PATH_FIELDS = ("input_path", "source_input_path", "dataset_path")
NESTED_TOP_LEVEL_FIELDS = ("run", "benchmark", "model", "sampling", "sandbox", "output")
MISSING = object()

NESTED_CONFIG_FIELD_MAP = {
    ("run", "sample_only"): "sample_only",
    ("run", "eval_only"): "eval_only",
    ("run", "resume_sample"): "resume_sample",
    ("run", "resume_sample_path"): "resume_sample_path",
    ("run", "eval_path"): "eval_path",
    ("run", "use_server"): "use_server",
    ("run", "use_ray"): "use_ray",
    ("benchmark", "name"): "benchmark",
    ("benchmark", "data_path"): "benchmark_data_path",
    ("benchmark", "version"): "version",
    ("benchmark", "begin_date"): "begin_date",
    ("benchmark", "end_date"): "end_date",
    ("benchmark", "special_judge_file"): "special_judge_file",
    ("benchmark", "language"): "language",
    ("model", "model_path"): "model_path",
    ("model", "model_name"): "model_name",
    ("model", "prompt_type"): "prompt_type",
    ("model", "thinking"): "thinking",
    ("model", "num_gpus_total"): "num_gpus_total",
    ("model", "num_gpus_per_model"): "num_gpus_per_model",
    ("model", "npu"): "npu",
    ("model", "mem_fraction"): "mem_fraction",
    ("model", "batch_size"): "batch_size",
    ("model", "max_model_len"): "max_model_len",
    ("model", "vllm_server", "base_port"): "vllm_server_base_port",
    ("model", "vllm_server", "host"): "vllm_server_host",
    ("model", "vllm_server", "dtype"): "vllm_server_dtype",
    ("model", "vllm_server", "wait_timeout"): "vllm_server_wait_timeout",
    ("sampling", "temperature"): "temperature",
    ("sampling", "top_p"): "top_p",
    ("sampling", "top_k"): "top_k",
    ("sampling", "min_p"): "min_p",
    ("sampling", "max_completion_tokens"): "max_completion_tokens",
    ("sampling", "n_sample"): "n_sample",
    ("sampling", "stop_token"): "stop_token",
    ("sandbox", "endpoint"): "endpoint",
    ("sandbox", "run_timeout"): "run_timeout",
    ("sandbox", "compile_timeout"): "compile_timeout",
    ("sandbox", "total_timeout"): "total_timeout",
    ("sandbox", "run_all_cases"): "run_all_cases",
    ("sandbox", "save_full_scalebox_result"): "save_full_scalebox_result",
    ("sandbox", "extra"): "extra",
    ("output", "output_dir"): "output_dir",
}

ALLOWED_SECTION_FIELDS = {
    "run": {"sample_only", "eval_only", "eval_path", "resume_sample", "resume_sample_path", "use_server", "use_ray"},
    "benchmark": {"name", "data_path", "version", "begin_date", "end_date", "special_judge_file", "language"},
    "model": {
        "model_path",
        "model_name",
        "prompt_type",
        "thinking",
        "num_gpus_total",
        "num_gpus_per_model",
        "npu",
        "mem_fraction",
        "batch_size",
        "max_model_len",
        "vllm_server",
    },
    "sampling": {"temperature", "top_p", "top_k", "min_p", "max_completion_tokens", "n_sample", "stop_token"},
    "sandbox": {"endpoint", "run_timeout", "compile_timeout", "total_timeout", "run_all_cases", "save_full_scalebox_result", "extra"},
    "output": {"output_dir"},
}

ALLOWED_VLLM_SERVER_FIELDS = {"base_port", "host", "dtype", "wait_timeout"}

def get_nested_value(config, path):
    current_value = config
    for key in path:
        if not isinstance(current_value, dict) or key not in current_value:
            return MISSING
        current_value = current_value[key]
    return current_value

def validate_nested_config_shape(config):
    unexpected_top_level_fields = [key for key in config if key not in NESTED_TOP_LEVEL_FIELDS]
    if unexpected_top_level_fields:
        valid_sections = ", ".join(NESTED_TOP_LEVEL_FIELDS)
        unexpected_fields_str = ", ".join(unexpected_top_level_fields[:10])
        raise ValueError(
            "Config must use the nested YAML format. "
            f"Top-level sections must be chosen from: {valid_sections}. "
            f"Unexpected top-level key(s): {unexpected_fields_str}"
        )

    if "benchmark" not in config:
        raise ValueError("Config must use the nested YAML format and include a `benchmark` section.")

    for section, value in config.items():
        if not isinstance(value, dict):
            raise ValueError(f"`{section}` must be a mapping in nested YAML config.")

    benchmark_config = config["benchmark"]
    if "name" not in benchmark_config:
        raise ValueError("Nested config must include `benchmark.name`.")

    for section, allowed_fields in ALLOWED_SECTION_FIELDS.items():
        section_config = config.get(section)
        if not section_config:
            continue
        unexpected_section_fields = [key for key in section_config if key not in allowed_fields]
        if unexpected_section_fields:
            unexpected_fields_str = ", ".join(unexpected_section_fields[:10])
            raise ValueError(
                f"Unexpected key(s) in `{section}`: {unexpected_fields_str}. "
                f"Allowed keys: {', '.join(sorted(allowed_fields))}"
            )

    model_config = config.get("model", {})
    if "vllm_server" in model_config and not isinstance(model_config["vllm_server"], dict):
        raise ValueError("`model.vllm_server` must be a mapping in nested YAML config.")
    if isinstance(model_config.get("vllm_server"), dict):
        unexpected_vllm_server_fields = [
            key for key in model_config["vllm_server"] if key not in ALLOWED_VLLM_SERVER_FIELDS
        ]
        if unexpected_vllm_server_fields:
            unexpected_fields_str = ", ".join(unexpected_vllm_server_fields[:10])
            raise ValueError(
                f"Unexpected key(s) in `model.vllm_server`: {unexpected_fields_str}. "
                f"Allowed keys: {', '.join(sorted(ALLOWED_VLLM_SERVER_FIELDS))}"
            )

    sandbox_config = config.get("sandbox", {})
    if "extra" in sandbox_config and not isinstance(sandbox_config["extra"], dict):
        raise ValueError("`sandbox.extra` must be a mapping in nested YAML config.")

    for field in OLD_PATH_FIELDS:
        if field in benchmark_config:
            raise ValueError(f"`benchmark.{field}` is no longer supported; please use `benchmark.data_path` instead.")
    if "huggingFace" in benchmark_config:
        raise ValueError("`benchmark.huggingFace` is no longer supported; please use `benchmark.data_path` instead.")

def flatten_nested_config(config):
    validate_nested_config_shape(config)

    flattened_config = {}
    for path, flat_key in NESTED_CONFIG_FIELD_MAP.items():
        value = get_nested_value(config, path)
        if value is MISSING:
            continue
        flattened_config[flat_key] = value
    return flattened_config

def load_config(config_path):
    # Load a single-benchmark YAML config file.
    with open(config_path, "r", encoding="utf-8") as file:
        try:
            raw_config = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(f"Failed to parse YAML config: {config_path}") from error

    if raw_config is None or not isinstance(raw_config, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")

    config = flatten_nested_config(raw_config)

    benchmark = config.get("benchmark")
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    if benchmark == "livecodebench" and not config.get("version"):
        raise ValueError("`livecodebench` config must include `benchmark.version`.")
    if benchmark == "aethercode" and not config.get("version"):
        raise ValueError("`aethercode` config must include `benchmark.version`.")
    if benchmark == "aethercode" and not config.get("special_judge_file"):
        raise ValueError("`aethercode` config must include `benchmark.special_judge_file`.")
    return config

def merge_config_into_args(args, config):
    # Merge config values into args while preserving explicit CLI overrides.
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == args._defaults.get(key):
            setattr(args, key, value)
    return args
