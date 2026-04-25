import argparse
import json
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from huggingface_hub import snapshot_download


REPO_ID = "m-a-p/AetherCode"
REPO_TYPE = "dataset"
ALLOWED_PATTERNS = ["README.md", "**/*.parquet"]
PARQUET_GLOB = "test-*.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download m-a-p/AetherCode, keep parquet files locally, and build special_judge jsonl files."
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


def load_parquet_rows(parquet_path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read AetherCode parquet files.") from exc

    rows = pq.read_table(parquet_path).to_pylist()
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected parquet payload type: {type(rows).__name__}")
    return rows


def sync_version_parquet(version_name: str, parquet_files: list[Path], cache_dir: Path, output_dir: Path) -> None:
    version_dir = output_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in parquet_files:
        relative_path = parquet_path.relative_to(cache_dir)
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parquet_path, destination)


def normalize_checker_row(row: dict, source_name: str, row_index: int) -> tuple[str | int, str] | None:
    if not isinstance(row, dict):
        raise RuntimeError(f"{source_name}: row {row_index} is not a dict: {type(row).__name__}")
    if "id" not in row:
        raise RuntimeError(f"{source_name}: row {row_index} missing required field `id`")

    checker = row.get("checker")
    if checker is None:
        checker = row.get("special_judge_program")
    if checker is None:
        return None
    if not isinstance(checker, str):
        raise RuntimeError(
            f"{source_name}: row {row_index} has non-str checker: {type(checker).__name__}"
        )
    return row["id"], checker


def sort_key(sample_id: object) -> tuple[int, str]:
    if isinstance(sample_id, int):
        return (0, str(sample_id))
    return (1, str(sample_id))


def write_jsonl_rows(rows: list[dict], output_path: Path) -> None:
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
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp_path.replace(output_path)
        output_path.chmod(0o664)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def build_special_judge_files(version_files: dict[str, list[Path]], output_dir: Path) -> tuple[int, list[Path]]:
    merged_checker_map: dict[str | int, str] = {}
    written_paths: list[Path] = []
    total_rows = 0

    for version_name, parquet_files in sorted(version_files.items()):
        version_checker_map: dict[str | int, str] = {}
        for parquet_path in parquet_files:
            rows = load_parquet_rows(parquet_path)
            total_rows += len(rows)
            for row_index, row in enumerate(rows, start=1):
                normalized = normalize_checker_row(row, parquet_path.name, row_index)
                if normalized is None:
                    continue
                sample_id, checker = normalized
                previous = version_checker_map.get(sample_id)
                if previous is not None and previous != checker:
                    raise RuntimeError(
                        f"{version_name}: conflicting checker for id={sample_id!r}"
                    )
                version_checker_map[sample_id] = checker

        version_rows = [
            {"id": sample_id, "checker": version_checker_map[sample_id]}
            for sample_id in sorted(version_checker_map, key=sort_key)
        ]
        version_filename = f"special_judge_{version_name.replace('/', '__')}.jsonl"
        version_output_path = output_dir / version_filename
        write_jsonl_rows(version_rows, version_output_path)
        written_paths.append(version_output_path)

        for sample_id, checker in version_checker_map.items():
            previous = merged_checker_map.get(sample_id)
            if previous is not None and previous != checker:
                raise RuntimeError(
                    f"conflicting merged checker for id={sample_id!r} across versions"
                )
            merged_checker_map[sample_id] = checker

    merged_rows = [
        {"id": sample_id, "checker": merged_checker_map[sample_id]}
        for sample_id in sorted(merged_checker_map, key=sort_key)
    ]
    merged_output_path = output_dir / "special_judge.jsonl"
    write_jsonl_rows(merged_rows, merged_output_path)
    written_paths.append(merged_output_path)

    return total_rows, written_paths


def main() -> int:
    args = parse_args()
    _, cache_dir, output_dir = resolve_paths()
    endpoint = configure_hf_endpoint(args.hf_endpoint)

    print(f"[HF] endpoint: {endpoint}")
    print(f"[HF] dataset: {REPO_ID}")
    print(f"[Paths] cache dir: {cache_dir}")
    print(f"[Paths] output dir: {output_dir}")

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
        print(f"[Sync] {version_name}: {len(parquet_files)} parquet files")

    total_rows, written_paths = build_special_judge_files(version_files, output_dir)

    print(f"[Done] versions: {len(version_files)}")
    print(f"[Done] synced parquet files: {synced_files}")
    print(f"[Done] scanned parquet rows: {total_rows}")
    for path in written_paths:
        print(f"[Done] wrote: {path}")
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
