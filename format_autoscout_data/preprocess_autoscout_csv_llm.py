import argparse
import json
import subprocess
import time
from typing import Any

import pandas as pd


def fallback_parse(value: str) -> dict[str, Any]:
    parts = value.split()
    maker = parts[0] if parts else None
    model = parts[1] if len(parts) > 1 else "Unknown"
    extra = " ".join(parts[2:]).strip() if len(parts) > 2 else None
    return {
        "maker": maker,
        "model": model,
        "kw": None,
        "extra_info": extra or None,
    }


def call_ollama_parse_batch(
    make_model_texts: list[str], model: str
) -> list[dict[str, Any]]:
    payload = [{"id": i, "text": text} for i, text in enumerate(make_model_texts)]
    prompt = f"""
You are parsing car listing title text.
Input is a JSON array of objects with fields: id, text
{json.dumps(payload, ensure_ascii=True)}

Return ONLY valid JSON array with exactly one object per input id.
Each output object must have exactly these keys:
- id (integer)
- maker (string, required)
- model (string, required)
- kw (string or null)
- extra_info (string or null)

Rules:
- maker and model must always be present.
- kw is power in kW if explicitly present (examples: "47kW", "85 kW"), else null.
- extra_info contains remaining useful trim/engine details not in maker/model/kw, else null.
- Keep maker/model concise and realistic for vehicle naming.
- No markdown, no explanation, JSON only.
""".strip()

    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama failed: {result.stderr.strip()}")

    raw = result.stdout.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model output is not JSON: {raw}")

    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Model output is not a JSON array.")

    by_id: dict[int, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("id")
        if not isinstance(idx, int):
            continue
        by_id[idx] = {
            "maker": item.get("maker"),
            "model": item.get("model"),
            "kw": item.get("kw"),
            "extra_info": item.get("extra_info"),
        }

    outputs: list[dict[str, Any]] = []
    for i, source in enumerate(make_model_texts):
        parsed_item = by_id.get(i)
        if parsed_item is None or not parsed_item.get("maker") or not parsed_item.get("model"):
            outputs.append(fallback_parse(source))
        else:
            outputs.append(parsed_item)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="autoscout_de.csv")
    parser.add_argument("--output", default="autoscout_de_llm_parsed.csv")
    parser.add_argument("--model", default="llama3.2:latest")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="How many unique make_model values to parse per LLM request.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for quick runs/testing.",
    )
    args = parser.parse_args()

    df = pd.read_csv(
        args.input,
        na_values=["", " "],
        keep_default_na=True,
        low_memory=False,
        nrows=args.max_rows,
    )
    original_row_count = len(df)

    empty_columns = [col for col in df.columns if df[col].isna().all()]
    cleaned_df = df.drop(columns=empty_columns).copy()
    cleaned_df["_row_id"] = range(original_row_count)

    # We deduplicate only unique text values for LLM calls (cost/speed),
    # then map back to every row so duplicate rows stay intact.
    unique_values = (
        cleaned_df["make_model"].fillna("").astype(str).str.strip().drop_duplicates()
    )
    parse_cache: dict[str, dict[str, Any]] = {}

    start_time = time.time()

    values_list = list(unique_values)
    total_unique = len(values_list)

    for batch_start in range(0, total_unique, args.batch_size):
        batch_values = values_list[batch_start : batch_start + args.batch_size]
        non_empty_batch = [v for v in batch_values if v]

        for value in batch_values:
            if not value:
                parse_cache[value] = {
                    "maker": None,
                    "model": None,
                    "kw": None,
                    "extra_info": None,
                }

        if non_empty_batch:
            try:
                parsed_batch = call_ollama_parse_batch(non_empty_batch, args.model)
                for source, parsed in zip(non_empty_batch, parsed_batch):
                    parse_cache[source] = parsed
            except Exception:
                for source in non_empty_batch:
                    parse_cache[source] = fallback_parse(source)

        processed = min(batch_start + args.batch_size, total_unique)
        if processed % 50 == 0 or processed == total_unique:
            elapsed_seconds = time.time() - start_time
            print(
                f"Parsed {processed}/{total_unique} unique make_model values... "
                f"(elapsed: {elapsed_seconds:.1f}s)"
            )

    parsed_rows = (
        cleaned_df["make_model"]
        .fillna("")
        .astype(str)
        .str.strip()
        .map(parse_cache)
        .apply(pd.Series)
    )

    cleaned_df = pd.concat([cleaned_df, parsed_rows], axis=1)
    if len(cleaned_df) != original_row_count:
        raise RuntimeError(
            f"Row count changed unexpectedly: {original_row_count} -> {len(cleaned_df)}"
        )

    duplicate_rows_before = int(df.duplicated().sum())
    duplicate_rows_after = int(
        cleaned_df.drop(columns=["_row_id"]).duplicated().sum()
    )

    cleaned_df = cleaned_df.drop(columns=["_row_id"])
    cleaned_df.to_csv(args.output, index=False)

    print(f"Saved parsed CSV to: {args.output}")
    print(
        f"Rows kept: {original_row_count} (duplicates preserved). "
        f"Duplicate rows before: {duplicate_rows_before}, after: {duplicate_rows_after}"
    )
    print("\n=== First 20 rows (make_model parsing) ===")
    print(cleaned_df[["make_model", "maker", "model", "kw", "extra_info"]].head(20))


if __name__ == "__main__":
    main()
