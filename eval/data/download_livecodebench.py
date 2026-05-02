import argparse
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
logger = get_logger(__name__)

REPO_ID = "livecodebench/code_generation_lite"
REPO_TYPE = "dataset"
FILES_TO_DOWNLOAD = [
    "code_generation_lite.py",
    "README.md",
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download livecodebench/code_generation_lite and build a local datasets directory."
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="Optional Hugging Face endpoint override, e.g. https://hf-mirror.com",
    )
    return parser.parse_args()

def resolve_paths() -> tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    cache_dir = script_dir / "hf_livecodebench"
    target_dir = script_dir / "livecodebench" / "code_generation_lite"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    return script_dir, cache_dir, target_dir

def resolve_endpoint(hf_endpoint: str | None) -> str | None:
    if hf_endpoint:
        return hf_endpoint
    return None

def resolve_downloaded_file(local_path: str, cache_dir: Path, filename: str) -> Path:
    path = Path(local_path)
    if path.exists():
        return path

    fallback = cache_dir / filename
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Downloaded file not found for {filename!r}: returned={path}, fallback={fallback}"
    )

def download_file(filename: str, cache_dir: Path, endpoint: str | None) -> Path:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=str(cache_dir),
        endpoint=endpoint,
    )
    return resolve_downloaded_file(local_path, cache_dir, filename)

def sync_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main() -> int:
    setup_logger()
    args = parse_args()
    _, cache_dir, target_dir = resolve_paths()
    endpoint = resolve_endpoint(args.hf_endpoint)

    logger.info(f"endpoint: {endpoint or 'https://huggingface.co'}")
    logger.info(f"dataset: {REPO_ID}")
    logger.info(f"cache dir: {cache_dir}")
    logger.info(f"target dir: {target_dir}")

    synced = 0
    for filename in FILES_TO_DOWNLOAD:
        src = download_file(filename, cache_dir, endpoint)
        dst = target_dir / filename
        sync_file(src, dst)
        synced += 1
        logger.info(f"{filename} -> {dst}")

    missing = [str(target_dir / name) for name in FILES_TO_DOWNLOAD if not (target_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing files after sync: {missing}")

    logger.info(f"synced files: {synced}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted.")
        raise SystemExit(130)
    except Exception as exc:
        logger.error(f"Error: {exc}")
        raise SystemExit(1)
