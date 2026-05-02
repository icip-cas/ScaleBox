import argparse
import shutil
import sys
from pathlib import Path

# Add parent directory to path to import logger
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download
from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

REPO_ID = "m-a-p/AetherCode"
REPO_TYPE = "dataset"
ALLOWED_PATTERNS = ["README.md", "**/*.parquet"]
PARQUET_GLOB = "test-*.parquet"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download m-a-p/AetherCode and keep parquet files locally."
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
    cache_dir = script_dir / "hf_aethercode"
    output_dir = script_dir / "aethercode"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return script_dir, cache_dir, output_dir

def configure_hf_endpoint(hf_endpoint: str | None) -> str:
    if hf_endpoint:
        return hf_endpoint
    return "https://huggingface.co"

def snapshot_repo(cache_dir: Path, endpoint: str) -> Path:
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(cache_dir),
        endpoint=endpoint,
        allow_patterns=ALLOWED_PATTERNS,
    )
    return Path(local_path)

def find_version_files(cache_dir: Path) -> dict[str, list[Path]]:
    version_files: dict[str, list[Path]] = {}
    for parquet_path in sorted(cache_dir.rglob(PARQUET_GLOB)):
        try:
            relative_path = parquet_path.relative_to(cache_dir)
        except ValueError:
            continue
        if relative_path.parts and relative_path.parts[0] == ".cache":
            continue
        version_name = relative_path.parent.as_posix()
        if version_name in {"", "."}:
            continue
        version_files.setdefault(version_name, []).append(parquet_path)
    return version_files

def sync_version_parquet(version_name: str, parquet_files: list[Path], cache_dir: Path, output_dir: Path) -> None:
    version_dir = output_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in parquet_files:
        relative_path = parquet_path.relative_to(cache_dir)
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parquet_path, destination)

def main() -> int:
    setup_logger()
    args = parse_args()
    _, cache_dir, output_dir = resolve_paths()
    endpoint = configure_hf_endpoint(args.hf_endpoint)

    logger.info(f"endpoint: {endpoint}")
    logger.info(f"dataset: {REPO_ID}")
    logger.info(f"cache dir: {cache_dir}")
    logger.info(f"output dir: {output_dir}")

    snapshot_repo(cache_dir, endpoint)
    version_files = find_version_files(cache_dir)
    if not version_files:
        raise RuntimeError(
            f"No parquet files matching {PARQUET_GLOB!r} found under {cache_dir}"
        )

    synced_files = 0
    for version_name, parquet_files in sorted(version_files.items()):
        sync_version_parquet(version_name, parquet_files, cache_dir, output_dir)
        synced_files += len(parquet_files)
        logger.info(f"{version_name}: {len(parquet_files)} parquet files")

    logger.info(f"versions: {len(version_files)}")
    logger.info(f"synced parquet files: {synced_files}")
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
