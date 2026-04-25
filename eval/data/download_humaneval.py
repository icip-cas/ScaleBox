import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from huggingface_hub import hf_hub_download


REPO_ID = "openai/openai_humaneval"
REPO_TYPE = "dataset"
PARQUET_FILE = "openai_humaneval/test-00000-of-00001.parquet"
KNOWN_FIELD_ORDER = [
    "id",
    "prompt",
    "canonical_solution",
    "test",
    "entry_point",
]
REQUIRED_FIELDS = {
    "task_id",
    "prompt",
    "canonical_solution",
    "test",
    "entry_point",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download openai/openai_humaneval and convert it to humaneval.jsonl."
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
    cache_dir = script_dir / "hf_openai_humaneval"
    output_dir = script_dir / "openai_humaneval"
    output_path = output_dir / "humaneval.jsonl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return script_dir, cache_dir, output_path


def configure_hf_endpoint(hf_endpoint: str | None) -> str:
    if hf_endpoint:
        return hf_endpoint
    return "https://huggingface.co"


def download_parquet(cache_dir: Path, endpoint: str) -> Path:
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=PARQUET_FILE,
        local_dir=str(cache_dir),
        endpoint=endpoint,
    )
    return Path(local_path)


def load_parquet_rows(parquet_path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the downloaded parquet file.") from exc

    rows = pq.read_table(parquet_path).to_pylist()
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected parquet payload type: {type(rows).__name__}")
    return rows


def validate_rows(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Downloaded dataset contains no rows.")

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"Row {index} is not a dict: {type(row).__name__}")
        row_missing = sorted(REQUIRED_FIELDS - set(row.keys()))
        if row_missing:
            raise RuntimeError(f"Row {index} missing required fields: {row_missing}")
        if not isinstance(row['task_id'], str):
            raise RuntimeError(f"Row {index} has non-str task_id: {type(row['task_id']).__name__}")
        if not isinstance(row['prompt'], str):
            raise RuntimeError(f"Row {index} has non-str prompt: {type(row['prompt']).__name__}")
        if not isinstance(row['canonical_solution'], str):
            raise RuntimeError(
                f"Row {index} has non-str canonical_solution: {type(row['canonical_solution']).__name__}"
            )
        if not isinstance(row['test'], str):
            raise RuntimeError(f"Row {index} has non-str test: {type(row['test']).__name__}")
        if not isinstance(row['entry_point'], str):
            raise RuntimeError(f"Row {index} has non-str entry_point: {type(row['entry_point']).__name__}")


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
            'w',
            encoding='utf-8',
            dir=output_path.parent,
            prefix=output_path.name + '.',
            suffix='.tmp',
            delete=False,
        ) as tmp:
            temp_path = Path(tmp.name)
            for row in rows:
                tmp.write(json.dumps(order_row(row), ensure_ascii=False) + '\n')
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
    print(f"[HF] dataset: {REPO_ID}")
    print(f"[Paths] cache dir: {cache_dir}")
    print(f"[Paths] output jsonl: {output_path}")

    parquet_path = download_parquet(cache_dir, endpoint)
    print(f"[HF] local parquet: {parquet_path}")

    rows = load_parquet_rows(parquet_path)
    validate_rows(rows)
    write_jsonl(rows, output_path)

    print(f"[Done] rows: {len(rows)}")
    print(f"[Done] wrote: {output_path}")
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1)
