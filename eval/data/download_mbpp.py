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

from huggingface_hub import hf_hub_download


REPO_ID = "google-research-datasets/mbpp"
REPO_TYPE = "dataset"
SPLIT_ORDER = ["prompt", "test", "validation", "train"]
PARQUET_FILES = {
    "prompt": "sanitized/prompt-00000-of-00001.parquet",
    "test": "sanitized/test-00000-of-00001.parquet",
    "validation": "sanitized/validation-00000-of-00001.parquet",
    "train": "sanitized/train-00000-of-00001.parquet",
}
KNOWN_FIELD_ORDER = [
    "id",
    "code",
    "prompt",
    "source_file",
    "test_imports",
    "test_list",
]
INPUT_REQUIRED_FIELDS = {
    "task_id",
    "code",
    "prompt",
    "source_file",
    "test_imports",
    "test_list",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download google-research-datasets/mbpp sanitized and convert it to mbpp.jsonl."
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
    cache_dir = script_dir / "hf_mbpp_sanitized"
    output_path = script_dir / "mbpp.jsonl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return script_dir, cache_dir, output_path


def configure_hf_endpoint(hf_endpoint: str | None) -> str:
    if hf_endpoint:
        return hf_endpoint
    return "https://huggingface.co"


def download_split_parquet(split: str, cache_dir: Path, endpoint: str) -> Path:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=PARQUET_FILES[split],
        local_dir=str(cache_dir),
        endpoint=endpoint,
    )
    return Path(local_path)


def load_parquet_rows(parquet_path: Path) -> list[dict]:
    if pq is None:
        raise RuntimeError("pyarrow is required to read the downloaded parquet file.")

    rows = pq.read_table(parquet_path).to_pylist()
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected parquet payload type: {type(rows).__name__}")
    return rows


def load_all_rows(cache_dir: Path, endpoint: str) -> list[dict]:
    rows = []
    for split in SPLIT_ORDER:
        parquet_path = download_split_parquet(split, cache_dir, endpoint)
        split_rows = load_parquet_rows(parquet_path)
        print(f"[Load] {split} rows: {len(split_rows)} | {parquet_path}")
        rows.extend(split_rows)
    rows.sort(key=lambda row: row["task_id"])
    return rows


def validate_rows(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Downloaded dataset contains no rows.")

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
        if not isinstance(row["prompt"], str):
            raise RuntimeError(f"Row {index} has non-str prompt: {type(row['prompt']).__name__}")
        if not isinstance(row["code"], str):
            raise RuntimeError(f"Row {index} has non-str code: {type(row['code']).__name__}")
        if not isinstance(row["test_list"], list):
            raise RuntimeError(f"Row {index} has non-list test_list: {type(row['test_list']).__name__}")
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
    args = parse_args()
    _, cache_dir, output_path = resolve_paths()
    endpoint = configure_hf_endpoint(args.hf_endpoint)

    print(f"[HF] endpoint: {endpoint}")
    print(f"[HF] dataset: {REPO_ID} (sanitized)")
    print(f"[Paths] cache dir: {cache_dir}")
    print(f"[Paths] output jsonl: {output_path}")

    rows = load_all_rows(cache_dir, endpoint)
    validate_rows(rows)
    write_jsonl(rows, output_path)

    print(f"[Done] rows: {len(rows)}")
    print(f"[Done] wrote: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
