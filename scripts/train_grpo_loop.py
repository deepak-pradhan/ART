#!/usr/bin/env python
"""
Automated GRPO training loop for TCM model.

Usage:
    # Run 100 iterations (default)
    uv run python scripts/train_grpo_loop.py

    # Run 50 iterations with larger batches
    uv run python scripts/train_grpo_loop.py --max-iterations 50 --batch-size 8

    # Custom learning rate
    uv run python scripts/train_grpo_loop.py --max-iterations 100 --learning-rate 1e-5

    # Resume from where you left off (just run again - it auto-resumes)
    uv run python scripts/train_grpo_loop.py --max-iterations 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from art import Backend, TrainableModel, TrainConfig
from art.trajectories import TrajectoryGroup


async def main(
    max_iterations: int = 100,
    batch_size: int = 4,
    trajectories_path: str = "data/processed/tcm-trajectories.jsonl",
    learning_rate: float = 5e-6,
    shuffle: bool = True,
    seed: int = 42,
    base_url: str = "http://localhost:7999",
    model_name: str = "tcm-formulator",
    project_name: str = "tcm-demo",
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> None:
    """Run automated GRPO training loop."""

    # Load trajectory groups
    print(f"Loading trajectory groups from {trajectories_path}...")
    groups: list[TrajectoryGroup] = []
    with open(trajectories_path) as f:
        for line in f:
            groups.append(TrajectoryGroup.model_validate(json.loads(line)))

    print(f"Loaded {len(groups)} trajectory groups")

    if shuffle:
        random.seed(seed)
        random.shuffle(groups)
        print(f"Shuffled with seed {seed}")

    # Connect to backend
    print(f"\nConnecting to ART backend at {base_url}...")
    backend = Backend(base_url=base_url)

    model = TrainableModel(
        name=model_name,
        project=project_name,
        base_model=base_model,
    )
    await model.register(backend)

    # Get current checkpoint
    start_step = await model.get_step()
    print(f"Starting from checkpoint: {start_step}")

    # Calculate coverage
    total_groups_per_epoch = len(groups)
    groups_per_run = min(max_iterations * batch_size, total_groups_per_epoch)
    coverage_pct = (groups_per_run / total_groups_per_epoch) * 100
    print(f"\nTraining plan:")
    print(f"  - Iterations: {max_iterations}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Groups per iteration: {batch_size}")
    print(f"  - Total groups to see: {groups_per_run}")
    print(f"  - Coverage: {coverage_pct:.1f}% of data")
    print(f"  - Learning rate: {learning_rate}")

    # Training loop
    config = TrainConfig(learning_rate=learning_rate)
    print("\n" + "=" * 60)
    print("Starting GRPO Training Loop")
    print("=" * 60)

    try:
        for iteration in range(max_iterations):
            # Batch selection (cycle through groups)
            start_idx = (iteration * batch_size) % len(groups)
            end_idx = start_idx + batch_size

            if end_idx <= len(groups):
                batch = groups[start_idx:end_idx]
            else:
                # Handle wrap-around at epoch boundary
                batch = groups[start_idx:] + groups[: end_idx - len(groups)]

            # Show progress
            epoch = (iteration * batch_size) // len(groups) + 1
            print(
                f"\n[{iteration + 1:3d}/{max_iterations}] "
                f"Epoch {epoch} | "
                f"Batch {start_idx}-{(start_idx + batch_size - 1) % len(groups)} | "
                f"Training..."
            )

            # Train on batch
            await model.train(
                batch,
                config=config,
                _config={"allow_training_without_logprobs": True},
            )

            # Show checkpoint
            new_step = await model.get_step()
            print(f"         Checkpoint: {new_step}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        final_step = await model.get_step()
        print(f"Last saved checkpoint: {final_step}")
        sys.exit(130)

    # Summary
    final_step = await model.get_step()
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Iterations completed: {max_iterations}")
    print(f"  Starting checkpoint: {start_step}")
    print(f"  Final checkpoint: {final_step}")
    print(f"  Checkpoints created: {final_step - start_step}")
    print(f"\nNext steps:")
    print(f"  - Run more iterations: uv run python scripts/train_grpo_loop.py --max-iterations 100")
    print(f"  - Test inference: curl http://localhost:8000/v1/chat/completions ...")
    print(f"  - Merge LoRA for deployment (see references/Current Training State.md)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated GRPO training loop for TCM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/train_grpo_loop.py --max-iterations 100
  uv run python scripts/train_grpo_loop.py --max-iterations 50 --batch-size 8
  uv run python scripts/train_grpo_loop.py --learning-rate 1e-5
        """,
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Trajectory groups per iteration (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--trajectories",
        default="data/processed/tcm-trajectories.jsonl",
        help="Path to trajectory groups JSONL file",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle trajectory groups",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:7999",
        help="ART backend URL (default: http://localhost:7999)",
    )
    parser.add_argument(
        "--model",
        default="tcm-formulator",
        help="Model name (default: tcm-formulator)",
    )
    parser.add_argument(
        "--project",
        default="tcm-demo",
        help="Project name (default: tcm-demo)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            main(
                max_iterations=args.max_iterations,
                batch_size=args.batch_size,
                trajectories_path=args.trajectories,
                learning_rate=args.learning_rate,
                shuffle=not args.no_shuffle,
                seed=args.seed,
                base_url=args.base_url,
                model_name=args.model,
                project_name=args.project,
            )
        )
    except KeyboardInterrupt:
        sys.exit(130)
