import os


def set_hf_cache():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from pathlib import Path

    # 获取当前工作目录
    current_dir = Path.cwd()
    # 创建缓存目录结构
    cache_dir = current_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    # 设置所有相关环境变量
    os.environ["HF_HOME"] = str(cache_dir)  # Hugging Face 主缓存目录
    os.environ["TRANSFORMERS_CACHE"] = str(
        cache_dir / "transformers"
    )  # Transformer 模型缓存
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")  # 数据集缓存
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(
        cache_dir / "hub"
    )  # Hugging Face Hub 缓存
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")  # 其他相关的缓存
