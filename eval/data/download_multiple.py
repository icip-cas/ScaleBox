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


REPO_ID = "nuprl/MultiPL-E"
REPO_TYPE = "dataset"
PARQUET_FILENAME = "test-00000-of-00001.parquet"
CORE_FIELD_ORDER = [
    "id",
    "language",
    "prompt",
    "tests",
    "stop_tokens",
    "task_id",
]
REQUIRED_FIELDS = {
    "name",
    "language",
    "prompt",
    "tests",
    "stop_tokens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all nuprl/MultiPL-E subsets and rebuild local jsonl files."
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
    cache_dir = script_dir / "hf_multiple_e"
    output_dir = script_dir / "MultiPL-E"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return script_dir, cache_dir, output_dir


def configure_hf_endpoint(hf_endpoint: str | None) -> str:
    if hf_endpoint:
        return hf_endpoint
    return "https://huggingface.co"


def discover_target_names(endpoint: str) -> list[str]:
    repo_files = HfApi(endpoint=endpoint).list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    suffix = f"/{PARQUET_FILENAME}"
    target_names = sorted({path[: -len(suffix)] for path in repo_files if path.endswith(suffix)})
    if not target_names:
        raise RuntimeError(
            f"No remote MultiPL-E subsets ending with {PARQUET_FILENAME!r} were found in {REPO_ID}."
        )
    return target_names


def download_parquet(target_name: str, cache_dir: Path, endpoint: str) -> Path:
    remote_path = f"{target_name}/{PARQUET_FILENAME}"
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=remote_path,
        local_dir=str(cache_dir),
        endpoint=endpoint,
    )
    return Path(local_path)


def load_parquet_rows(parquet_path: Path) -> list[dict]:
    if pq is None:
        raise RuntimeError("pyarrow is required to read MultiPL-E parquet files.")

    rows = pq.read_table(parquet_path).to_pylist()
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected parquet payload type: {type(rows).__name__}")
    return rows


def validate_rows(rows: list[dict], target_name: str) -> None:
    if not rows:
        raise RuntimeError(f"{target_name}: downloaded parquet contains no rows.")

    first = rows[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"{target_name}: expected row to be a dict, got {type(first).__name__}.")

    missing = sorted(REQUIRED_FIELDS - set(first.keys()))
    if missing:
        raise RuntimeError(
            f"{target_name}: missing required fields {missing}. Available fields: {sorted(first.keys())}"
        )

    seen_ids = set()
    duplicates = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"{target_name}: row {index} is not a dict: {type(row).__name__}")
        row_missing = sorted(REQUIRED_FIELDS - set(row.keys()))
        if row_missing:
            raise RuntimeError(f"{target_name}: row {index} missing required fields: {row_missing}")
        if not isinstance(row["name"], str):
            raise RuntimeError(f"{target_name}: row {index} has non-str name: {type(row['name']).__name__}")
        if not isinstance(row["language"], str):
            raise RuntimeError(f"{target_name}: row {index} has non-str language: {type(row['language']).__name__}")
        if not isinstance(row["prompt"], str):
            raise RuntimeError(f"{target_name}: row {index} has non-str prompt: {type(row['prompt']).__name__}")
        if not isinstance(row["tests"], str):
            raise RuntimeError(f"{target_name}: row {index} has non-str tests: {type(row['tests']).__name__}")
        if not isinstance(row["stop_tokens"], list):
            raise RuntimeError(
                f"{target_name}: row {index} has non-list stop_tokens: {type(row['stop_tokens']).__name__}"
            )
        sample_id = row["name"]
        if sample_id in seen_ids:
            duplicates.append(sample_id)
        seen_ids.add(sample_id)

    if duplicates:
        preview = duplicates[:10]
        raise RuntimeError(f"{target_name}: duplicate name values found: {preview}")


def order_row(row: dict) -> dict:
    normalized = dict(row)
    normalized["id"] = normalized.pop("name")

    ordered = {key: normalized[key] for key in CORE_FIELD_ORDER if key in normalized}
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
    _, cache_dir, output_dir = resolve_paths()
    endpoint = configure_hf_endpoint(args.hf_endpoint)
    target_names = discover_target_names(endpoint)

    print(f"[HF] endpoint: {endpoint}")
    print(f"[HF] repo: {REPO_ID}")
    print(f"[Paths] cache dir: {cache_dir}")
    print(f"[Paths] output dir: {output_dir}")
    print(f"[Plan] targets: {len(target_names)}")

    rebuilt_count = 0
    total_rows = 0
    for target_name in target_names:
        parquet_path = download_parquet(target_name, cache_dir, endpoint)
        rows = load_parquet_rows(parquet_path)
        validate_rows(rows, target_name)
        output_path = output_dir / f"{target_name}.jsonl"
        write_jsonl(rows, output_path)
        rebuilt_count += 1
        total_rows += len(rows)
        print(f"[Load] {target_name} rows: {len(rows)} | {parquet_path}")
        print(f"[Done] wrote: {output_path}")

    print(f"[Done] rebuilt files: {rebuilt_count}")
    print(f"[Done] total rows: {total_rows}")
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
