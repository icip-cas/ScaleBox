import os


def set_hf_cache():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from pathlib import Path

    # Get the current working directory
    current_dir = Path.cwd()
    # Create the cache directory structure
    cache_dir = current_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    # Set all related environment variables
    os.environ["HF_HOME"] = str(cache_dir)  # Hugging Face main cache directory
    os.environ["TRANSFORMERS_CACHE"] = str(
        cache_dir / "transformers"
    )  # Transformers model cache
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")  # Datasets cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(
        cache_dir / "hub"
    )  # Hugging Face Hub cache
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")  # Other related cache
