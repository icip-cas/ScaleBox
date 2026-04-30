import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger
logger = get_logger(__name__)

REPO_ID = "evalplus/mbppplus"
REPO_TYPE = "dataset"
KNOWN_FIELD_ORDER = [
    "id",
    "code",
    "prompt",
    "source_file",
    "test_imports",
    "test_list",
    "test",
]
INPUT_REQUIRED_FIELDS = {
    "task_id",
    "code",
    "prompt",
    "source_file",
    "test_imports",
    "test_list",
    "test",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download evalplus/mbppplus and convert it to mbppplus.jsonl."
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
    download_dir = script_dir / "hf_evalplus_mbppplus"
    output_path = script_dir / "mbppplus.jsonl"
    download_dir.mkdir(parents=True, exist_ok=True)
    return script_dir, download_dir, output_path

def configure_hf_endpoint(hf_endpoint: str | None) -> str:
    if hf_endpoint:
        return hf_endpoint
    return "https://huggingface.co"

def find_remote_parquet(endpoint: str) -> str:
    repo_files = HfApi(endpoint=endpoint).list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    parquet_files = sorted(
        path
        for path in repo_files
        if path.startswith("data/") and path.endswith(".parquet") and Path(path).name.startswith("test-")
    )
    if not parquet_files:
        raise RuntimeError(f"No test parquet files found in {REPO_ID}.")
    if len(parquet_files) != 1:
        joined = "\n".join(parquet_files)
        raise RuntimeError(f"Expected exactly one test parquet file, found {len(parquet_files)}:\n{joined}")
    return parquet_files[0]

def download_remote_parquet(remote_path: str, download_dir: Path, endpoint: str) -> Path:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=remote_path,
        local_dir=str(download_dir),
        endpoint=endpoint,
    )
    return Path(local_path)

def load_rows(parquet_path: Path) -> list[dict]:
    if pq is None:
        raise RuntimeError("pyarrow is required to read the downloaded parquet file.")

    rows = pq.read_table(parquet_path).to_pylist()
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected parquet payload type: {type(rows).__name__}")
    return rows

def validate_rows(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Downloaded parquet contains no rows.")

    first = rows[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"Expected row to be a dict, got {type(first).__name__}.")

    missing = sorted(INPUT_REQUIRED_FIELDS - set(first.keys()))
    if missing:
        raise RuntimeError(
            f"Missing required fields: {missing}. Available fields: {sorted(first.keys())}"
        )

    seen_task_ids = set()
    duplicates = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"Row {index} is not a dict: {type(row).__name__}")
        row_missing = sorted(INPUT_REQUIRED_FIELDS - set(row.keys()))
        if row_missing:
            raise RuntimeError(f"Row {index} missing required fields: {row_missing}")
        if not isinstance(row["test_list"], list):
            raise RuntimeError(f"Row {index} has non-list test_list: {type(row['test_list']).__name__}")
        if not isinstance(row["test"], str):
            raise RuntimeError(f"Row {index} has non-str test: {type(row['test']).__name__}")
        task_id = row["task_id"]
        if task_id in seen_task_ids:
            duplicates.append(task_id)
        seen_task_ids.add(task_id)

    if duplicates:
        raise RuntimeError(f"Duplicate task_id values found: {sorted(set(duplicates))[:10]}")

def order_row(row: dict) -> dict:
    normalized = dict(row)
    normalized["id"] = normalized.pop("task_id")

    ordered = {key: normalized[key] for key in KNOWN_FIELD_ORDER if key in normalized}
    for key in normalized:
        if key not in ordered:
            ordered[key] = normalized[key]
    return ordered

def write_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=output_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            temp_path = Path(tmp.name)
            for row in rows:
                tmp.write(json.dumps(order_row(row), ensure_ascii=False) + "\n")
        os.replace(temp_path, output_path)
        os.chmod(output_path, 0o664)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

def main() -> int:
    setup_logger()
    args = parse_args()
    _, download_dir, output_path = resolve_paths()
    endpoint = configure_hf_endpoint(args.hf_endpoint)

    logger.info(f"endpoint: {endpoint}")
    logger.info(f"repo: {REPO_ID} ({REPO_TYPE})")
    logger.info(f"download dir: {download_dir}")
    logger.info(f"output jsonl: {output_path}")

    remote_parquet = find_remote_parquet(endpoint)
    logger.info(f"remote parquet: {remote_parquet}")

    local_parquet = download_remote_parquet(remote_parquet, download_dir, endpoint)
    logger.info(f"local parquet: {local_parquet}")

    rows = load_rows(local_parquet)
    validate_rows(rows)
    write_jsonl(rows, output_path)

    logger.info(f"rows: {len(rows)}")
    logger.info(f"wrote: {output_path}")
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
