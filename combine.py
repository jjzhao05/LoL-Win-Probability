from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RANK_ORDER = [
    "IRON",
    "BRONZE",
    "SILVER",
    "GOLD",
    "PLATINUM",
    "EMERALD",
    "DIAMOND",
    "MASTER",
    "GRANDMASTER",
    "CHALLENGER",
]


def find_rank_dirs(input_dir: Path) -> list[tuple[str, Path]]:
    rank_dirs: list[tuple[str, Path]] = []

    for rank in RANK_ORDER:
        candidate = input_dir / f"{rank}_snapshots_parquet"
        if candidate.exists() and candidate.is_dir():
            rank_dirs.append((rank, candidate))

    return rank_dirs


def collect_parquet_files(rank_dirs: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []

    for rank, rank_dir in rank_dirs:
        for file_path in sorted(rank_dir.glob("*.parquet")):
            files.append((rank, file_path))

    return files


def infer_schema(first_file: Path, add_rank_column: bool) -> pa.Schema:
    schema = pq.read_schema(first_file)

    if add_rank_column and "rank" not in schema.names:
        schema = schema.append(pa.field("rank", pa.string()))

    return schema


def align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []

    for field in schema:
        if field.name in table.column_names:
            column = table[field.name]

            if not column.type.equals(field.type):
                column = column.cast(field.type)

            arrays.append(column)
        else:
            arrays.append(pa.nulls(len(table), type=field.type))

    return pa.Table.from_arrays(arrays, schema=schema)


def combine_parquets(
    input_dir: Path,
    output_file: Path,
    batch_size: int,
    compression: str,
    add_rank_column: bool,
) -> None:
    rank_dirs = find_rank_dirs(input_dir)

    if not rank_dirs:
        raise FileNotFoundError(
            f"No rank parquet folders found in {input_dir}. "
            "Expected folders like IRON_snapshots_parquet, SILVER_snapshots_parquet, etc."
        )

    parquet_files = collect_parquet_files(rank_dirs)

    if not parquet_files:
        raise FileNotFoundError("No .parquet files found inside the rank folders.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    first_rank, first_file = parquet_files[0]
    output_schema = infer_schema(first_file, add_rank_column=add_rank_column)

    print(f"Found {len(rank_dirs)} rank folders")
    for rank, rank_dir in rank_dirs:
        count = len(list(rank_dir.glob("*.parquet")))
        print(f"  {rank}: {count} parquet files")

    print(f"\nWriting combined parquet to: {output_file}")
    print(f"Compression: {compression}")
    print(f"Batch size: {batch_size:,}")
    print(f"Add rank column: {add_rank_column}\n")

    total_rows = 0
    total_files = 0

    with pq.ParquetWriter(
        where=output_file,
        schema=output_schema,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
    ) as writer:
        for rank, file_path in parquet_files:
            print(f"Reading {rank}: {file_path.name}")

            parquet_dataset = ds.dataset(file_path, format="parquet")
            scanner = parquet_dataset.scanner(batch_size=batch_size)

            file_rows = 0

            for record_batch in scanner.to_batches():
                table = pa.Table.from_batches([record_batch])

                if add_rank_column and "rank" not in table.column_names:
                    rank_array = pa.array([rank] * len(table), type=pa.string())
                    table = table.append_column("rank", rank_array)

                table = align_table_to_schema(table, output_schema)

                writer.write_table(table)

                batch_rows = len(table)
                file_rows += batch_rows
                total_rows += batch_rows

            total_files += 1
            print(f"  Wrote {file_rows:,} rows")

    print("\nDone.")
    print(f"Files combined: {total_files:,}")
    print(f"Rows written: {total_rows:,}")
    print(f"Output file: {output_file}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else BASE_DIR / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine rank-separated League of Legends parquet snapshot folders into one parquet file."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing rank folders such as IRON_snapshots_parquet and SILVER_snapshots_parquet.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DATA_DIR / "full_dataset.parquet",
        help="Path for the combined output parquet file.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Number of rows to process per batch.",
    )

    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
        help="Parquet compression codec.",
    )

    parser.add_argument(
        "--no-rank-column",
        action="store_true",
        help="Do not add a rank column to the combined output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    compression = None if args.compression == "none" else args.compression
    input_dir = resolve_path(args.input_dir)
    output_file = resolve_path(args.output_file)

    combine_parquets(
        input_dir=input_dir,
        output_file=output_file,
        batch_size=args.batch_size,
        compression=compression,
        add_rank_column=not args.no_rank_column,
    )


if __name__ == "__main__":
    main()