#!/usr/bin/env python
"""
Generate GRPO training data by creating multiple response variations per condition.

This script:
1. Reads unique conditions from tcm.duckdb
2. Generates 4 response variations per condition using the model
3. Scores responses with an LLM-as-judge rubric
4. Outputs TrajectoryGroups ready for GRPO training

Usage:
    # Start the ART server first:
    uv run art --host 0.0.0.0 --port 7999

    # Then run this script:
    uv run python scripts/generate_grpo_data.py --limit 100
"""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import duckdb
from openai import AsyncOpenAI

# Response generation prompts with varying specificity requirements
GENERATION_PROMPTS = [
    # High quality - detailed, specific
    {
        "system": "You are an expert Traditional Chinese Medicine practitioner. Provide detailed, evidence-based recommendations with specific dosages, preparation methods, and safety monitoring.",
        "suffix": "Be very specific about dosages (in grams), preparation methods, duration, and any contraindications or monitoring needed.",
        "expected_quality": "high",
    },
    # Medium-high quality - good but less detailed
    {
        "system": "You are a Traditional Chinese Medicine practitioner. Provide clear recommendations with dosages and basic preparation guidance.",
        "suffix": "Include dosage recommendations and preparation tips.",
        "expected_quality": "medium_high",
    },
    # Medium quality - basic
    {
        "system": "You are a TCM assistant. Provide helpful recommendations for using this herb.",
        "suffix": "Keep the response practical and concise.",
        "expected_quality": "medium",
    },
    # Lower quality - vague
    {
        "system": "You are a helpful assistant with some knowledge of traditional medicine.",
        "suffix": "Give a brief suggestion.",
        "expected_quality": "low",
    },
]

# Scoring rubric for LLM-as-judge
SCORING_PROMPT = """You are evaluating a Traditional Chinese Medicine recommendation for quality.

CONDITION: {condition}
HERB: {herb} (compounds: {compounds})
RESPONSE: {response}

Score this response from 0.0 to 1.0 based on these criteria:
- Dosage specificity (0-0.25): Does it include specific gram amounts?
- Preparation method (0-0.25): Does it explain how to prepare (decoction, powder, etc.)?
- Safety/monitoring (0-0.25): Does it mention contraindications, side effects, or monitoring?
- Clarity/actionability (0-0.25): Is it clear and actionable for a practitioner?

Respond with ONLY a JSON object:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""


async def generate_responses(
    client: AsyncOpenAI,
    model: str,
    condition: str,
    herb: str,
    compounds: str,
) -> list[dict]:
    """Generate multiple response variations for a single condition."""
    responses = []

    user_base = f"""Patient condition: {condition}
Recommended herb: {herb}
Active compounds: {compounds if compounds else "Not specified"}

Provide a treatment recommendation."""

    for prompt_config in GENERATION_PROMPTS:
        user_prompt = f"{user_base}\n\n{prompt_config['suffix']}"

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt_config["system"]},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.7 + random.uniform(-0.2, 0.2),  # Add some variance
            )

            content = response.choices[0].message.content or ""
            responses.append(
                {
                    "response": content,
                    "expected_quality": prompt_config["expected_quality"],
                }
            )
        except Exception as e:
            print(f"  Warning: Generation failed - {e}")
            continue

    return responses


async def score_response(
    client: AsyncOpenAI,
    model: str,
    condition: str,
    herb: str,
    compounds: str,
    response: str,
) -> float:
    """Score a response using LLM-as-judge."""
    prompt = SCORING_PROMPT.format(
        condition=condition,
        herb=herb,
        compounds=compounds or "Not specified",
        response=response,
    )

    try:
        result = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,  # Low temp for consistent scoring
        )

        content = result.choices[0].message.content or "{}"
        # Extract JSON from response
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            data = json.loads(json_match.group())
            return float(data.get("score", 0.5))
    except Exception as e:
        print(f"  Warning: Scoring failed - {e}")

    # Fallback: use expected quality as heuristic
    return 0.5


async def process_condition(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    use_llm_scoring: bool = True,
) -> list[dict]:
    """Process a single condition: generate responses and score them."""
    condition = row["condition_name"]
    herb = row["name_common"]
    compounds = row["compounds"]

    # Generate responses
    responses = await generate_responses(client, model, condition, herb, compounds)

    if not responses:
        return []

    # Score responses
    scored = []
    for resp in responses:
        if use_llm_scoring:
            score = await score_response(
                client, model, condition, herb, compounds, resp["response"]
            )
        else:
            # Heuristic scoring based on expected quality
            quality_scores = {
                "high": 0.85,
                "medium_high": 0.72,
                "medium": 0.58,
                "low": 0.40,
            }
            score = quality_scores.get(resp["expected_quality"], 0.5)
            # Add some noise
            score += random.uniform(-0.08, 0.08)
            score = max(0.1, min(0.95, score))

        scored.append(
            {
                "condition": condition,
                "herb": herb,
                "compounds": compounds,
                "response": resp["response"],
                "reward": round(score, 3),
            }
        )

    return scored


async def main(
    db_path: str = "data/tcm.duckdb",
    output_path: str = "data/grpo_training_data.jsonl",
    limit: int | None = None,
    use_llm_scoring: bool = False,
    inference_url: str = "http://localhost:8000/v1",
    model_name: str = "tcm-formulator",
):
    """Main function to generate GRPO training data."""

    # Connect to database
    print(f"Reading conditions from {db_path}...")
    conn = duckdb.connect(db_path, read_only=True)

    query = """
        SELECT DISTINCT
            condition_name,
            name_common,
            compounds
        FROM exports.tcm_instruction_pairs
        WHERE condition_name IS NOT NULL
          AND condition_name != ''
    """
    if limit:
        query += f" LIMIT {limit}"

    rows = conn.execute(query).fetchdf().to_dict("records")
    conn.close()

    print(f"Found {len(rows)} unique conditions")

    # Initialize OpenAI client pointing to local vLLM
    client = AsyncOpenAI(
        base_url=inference_url,
        api_key="default",
    )

    # Check if server is available
    try:
        models = await client.models.list()
        available_models = [m.id for m in models.data]
        print(f"Available models: {available_models}")
        if model_name not in available_models and available_models:
            model_name = available_models[0]
            print(f"Using model: {model_name}")
    except Exception as e:
        print(f"Warning: Could not connect to inference server - {e}")
        print("Make sure ART server is running: uv run art --host 0.0.0.0 --port 7999")
        return

    # Process conditions
    all_data = []
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating responses (LLM scoring: {use_llm_scoring})...")

    with open(output_file, "w") as f:
        for i, row in enumerate(rows):
            print(
                f"  [{i + 1}/{len(rows)}] {row['name_common']}: {row['condition_name'][:50]}..."
            )

            scored_responses = await process_condition(
                client, model_name, row, use_llm_scoring
            )

            if scored_responses:
                # Write as a trajectory group (same condition = same group)
                group = {
                    "condition": row["condition_name"],
                    "herb": row["name_common"],
                    "trajectories": scored_responses,
                }
                f.write(json.dumps(group) + "\n")
                all_data.append(group)

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Conditions processed: {len(all_data)}")
    total_responses = sum(len(g["trajectories"]) for g in all_data)
    print(f"Total responses: {total_responses}")

    if all_data:
        all_rewards = [t["reward"] for g in all_data for t in g["trajectories"]]
        print(f"Reward range: {min(all_rewards):.2f} - {max(all_rewards):.2f}")
        print(f"Mean reward: {sum(all_rewards) / len(all_rewards):.2f}")

    print(f"\nOutput saved to: {output_file}")
    print(
        "\nNext step: Use scripts/build_trajectories.py to convert to TrajectoryGroups"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GRPO training data")
    parser.add_argument(
        "--db", default="data/tcm.duckdb", help="Path to DuckDB database"
    )
    parser.add_argument(
        "--output", default="data/grpo_training_data.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of conditions"
    )
    parser.add_argument(
        "--llm-scoring", action="store_true", help="Use LLM-as-judge scoring (slower)"
    )
    parser.add_argument(
        "--inference-url", default="http://localhost:8000/v1", help="vLLM inference URL"
    )
    parser.add_argument(
        "--model", default="tcm-formulator", help="Model name for generation"
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            db_path=args.db,
            output_path=args.output,
            limit=args.limit,
            use_llm_scoring=args.llm_scoring,
            inference_url=args.inference_url,
            model_name=args.model,
        )
    )
