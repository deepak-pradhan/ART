#!/usr/bin/env python
"""
Convert DuckDB/Postgres TCM records into ART trajectory groups.

Usage examples:
  # From DuckDB (random grouping - not ideal for GRPO):
  uv run python scripts/build_trajectories.py --duckdb-path data/tcm.duckdb

  # From GRPO-generated JSONL (proper grouping by condition):
  uv run python scripts/build_trajectories.py --jsonl-path data/grpo_training_data.jsonl

  # From Postgres:
  uv run python scripts/build_trajectories.py --postgres-uri postgresql://user:pass@host/db
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

from art.trajectories import Trajectory, TrajectoryGroup

DEFAULT_QUERY = "SELECT * FROM exports.tcm_instruction_pairs"


def load_groups_from_jsonl(
    path: str, min_reward_variation: float
) -> list[TrajectoryGroup]:
    """Load pre-grouped trajectory data from JSONL (from generate_grpo_data.py)."""
    groups = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            trajectories_data = data.get("trajectories", [])

            if len(trajectories_data) < 2:
                continue

            rewards = [t["reward"] for t in trajectories_data]
            if max(rewards) - min(rewards) < min_reward_variation:
                continue

            trajectories = []
            for t in trajectories_data:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a cautious Traditional Chinese Medicine assistant.",
                    },
                    {
                        "role": "user",
                        "content": f"Patient condition: {t['condition']}\nHerb: {t['herb']} (compounds: {t.get('compounds', 'N/A')})\nProvide treatment recommendation.",
                    },
                    {"role": "assistant", "content": t["response"]},
                ]
                traj = Trajectory(
                    messages_and_choices=messages,
                    reward=t["reward"],
                    metadata={
                        "condition": t["condition"],
                        "herb": t["herb"],
                        "source": "grpo_generated",
                    },
                ).finish()
                trajectories.append(traj)

            groups.append(TrajectoryGroup(trajectories))

    return groups


def load_rows_from_duckdb(path: str, query: str, init_sql: Path | None) -> list[dict]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "duckdb is required for --duckdb-path. Install with `uv add duckdb`."
        ) from exc

    conn = duckdb.connect(path, read_only=False)
    if init_sql:
        conn.execute(init_sql.read_text())
    df = conn.execute(query).fetchdf()
    return df.to_dict("records")


def load_rows_from_postgres(uri: str, query: str, init_sql: Path | None) -> list[dict]:
    try:
        import psycopg  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "psycopg is required for --postgres-uri. Install with `uv add psycopg[binary]`."
        ) from exc

    rows: list[dict] = []
    with psycopg.connect(uri) as conn:
        with conn.cursor() as cur:
            if init_sql:
                cur.execute(init_sql.read_text())
            cur.execute(query)
            col_names = [desc[0] for desc in cur.description]
            for result in cur.fetchall():
                rows.append(dict(zip(col_names, result)))
    return rows


def build_messages(record: dict) -> list[dict]:
    compounds = record.get("compounds") or ""
    compound_line = (
        f"(compounds: {compounds})" if compounds.strip() else "(no compound data)"
    )
    monitoring = record.get("monitoring_clause") or ""

    user_prompt = (
        f"Patient condition: {record.get('condition_name', 'unknown')}.\n"
        f"Highlighted materia medica: {record.get('name_scientific', record.get('name_common', 'unknown'))} {compound_line}.\n"
        "Recommend dosage, preparation details, contraindication reminders, and monitoring guidance."
    )
    if monitoring:
        user_prompt += f"\nMonitoring context: {monitoring}"

    return [
        {
            "role": "system",
            "content": "You are a cautious Traditional Chinese Medicine assistant.",
        },
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": record.get("response_text", "").strip()},
    ]


def build_trajectory(record: dict) -> Trajectory:
    reward = float(record.get("reward", 0.0) or 0.0)
    metadata = {
        "plant_id": record.get("plant_id"),
        "condition": record.get("condition_name"),
        "name_scientific": record.get("name_scientific"),
        "source": record.get("_source", "db"),
    }
    return Trajectory(
        messages_and_choices=build_messages(record),
        reward=reward,
        metadata=metadata,
    ).finish()


def chunk_records(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for idx in range(0, len(rows), size):
        yield rows[idx : idx + size]


def build_groups(
    rows: list[dict],
    *,
    group_size: int,
    min_reward_variation: float,
) -> list[TrajectoryGroup]:
    groups: list[TrajectoryGroup] = []
    for chunk in chunk_records(rows, group_size):
        rewards = {round(float(r.get("reward", 0.0)), 3) for r in chunk}
        if (
            not chunk
            or len(rewards) <= 1
            or max(rewards) - min(rewards) < min_reward_variation
        ):
            continue
        trajectories = [build_trajectory(rec) for rec in chunk]
        groups.append(TrajectoryGroup(trajectories))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb-path", help="Path to DuckDB database file.")
    parser.add_argument(
        "--postgres-uri", help="Postgres URI (postgresql://user:pw@host/db)."
    )
    parser.add_argument(
        "--jsonl-path",
        help="Path to GRPO-generated JSONL file (from generate_grpo_data.py).",
    )
    parser.add_argument(
        "--init-sql",
        type=Path,
        default=Path("project-notes/sql/tcm_instruction_pairs.sql"),
        help="SQL file to execute before running the main query.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="SQL query to fetch rows (defaults to exports.tcm_instruction_pairs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tcm-trajectories.jsonl"),
        help="JSONL file to write trajectory groups into.",
    )
    parser.add_argument(
        "--group-size", type=int, default=4, help="Trajectories per group."
    )
    parser.add_argument(
        "--min-reward-variation",
        type=float,
        default=0.05,
        help="Minimum reward spread required inside a group.",
    )
    parser.add_argument(
        "--max-groups", type=int, help="Optional cap on number of groups."
    )
    parser.add_argument("--seed", type=int, default=4242, help="Shuffle seed.")
    args = parser.parse_args()

    if not args.duckdb_path and not args.postgres_uri and not args.jsonl_path:
        parser.error("Provide --duckdb-path, --postgres-uri, or --jsonl-path.")

    # Load from JSONL (pre-grouped, best for GRPO)
    if args.jsonl_path:
        print(f"Loading pre-grouped data from {args.jsonl_path}...")
        groups = load_groups_from_jsonl(args.jsonl_path, args.min_reward_variation)
        random.Random(args.seed).shuffle(groups)
        if args.max_groups:
            groups = groups[: args.max_groups]
    else:
        # Load from database (random grouping - less ideal for GRPO)
        init_sql = args.init_sql if args.init_sql and args.init_sql.exists() else None

        if args.duckdb_path:
            rows = load_rows_from_duckdb(args.duckdb_path, args.query, init_sql)
        else:
            rows = load_rows_from_postgres(args.postgres_uri, args.query, init_sql)

        rows = [row for row in rows if (row.get("response_text") or "").strip()]
        random.Random(args.seed).shuffle(rows)

        groups = build_groups(
            rows,
            group_size=args.group_size,
            min_reward_variation=args.min_reward_variation,
        )
        if args.max_groups:
            groups = groups[: args.max_groups]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for group in groups:
            f.write(json.dumps(group.model_dump()) + "\n")

    print(
        f"Wrote {len(groups)} trajectory groups "
        f"({len(groups) * args.group_size} rows before filtering) to {args.output}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)
