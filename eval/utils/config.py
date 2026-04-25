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


def load_config(config_path):
    """中文：加载单 benchmark 的 YAML 配置文件。
    English: Load a single-benchmark YAML config file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(f"Failed to parse YAML config: {config_path}") from error

    if config is None or not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")

    if "benchmark" not in config:
        raise ValueError("Config must use the new YAML format and include a `benchmark` field")

    benchmark = config.get("benchmark")
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    for field in OLD_PATH_FIELDS:
        if field in config:
            raise ValueError(f"`{field}` is no longer supported; please use `benchmark_data_path` instead.")
    if benchmark == "multipl_e" and "huggingFace" in config:
        raise ValueError("`multipl_e` no longer supports `huggingFace`; please provide `benchmark_data_path` instead.")
    if benchmark == "livecodebench" and not config.get("version"):
        raise ValueError("`livecodebench` config must include `version`.")
    if benchmark == "aethercode" and not config.get("version"):
        raise ValueError("`aethercode` config must include `version`.")
    if benchmark == "aethercode" and not config.get("special_judge_file"):
        raise ValueError("`aethercode` config must include `special_judge_file`.")
    return config


def merge_config_into_args(args, config):
    """中文：把配置文件中的参数写回 args，CLI 显式值优先。
    English: Merge config values into args while preserving explicit CLI overrides.
    """
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == args._defaults.get(key):
            setattr(args, key, value)
    return args
