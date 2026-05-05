import pandas as pd
import re


TWO_WORD_MAKERS = {
    "Alfa Romeo",
    "Aston Martin",
    "DS Automobiles",
    "Great Wall",
    "Land Rover",
    "Mercedes Benz",
    "Rolls Royce",
}


def parse_make_model(value: object) -> pd.Series:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return pd.Series({"maker": None, "model": None, "extra": None})

    tokens = text.split()
    maker = tokens[0]
    start_idx = 1

    if len(tokens) >= 2:
        maybe_two_word = f"{tokens[0]} {tokens[1]}"
        if maybe_two_word in TWO_WORD_MAKERS:
            maker = maybe_two_word
            start_idx = 2

    remaining_tokens = tokens[start_idx:]

    # Model is always present and can be up to 2 words.
    if len(remaining_tokens) >= 2:
        first, second = remaining_tokens[0], remaining_tokens[1]
        if re.search(r"\d", second) and not re.search(r"[A-Za-z]", second):
            model_tokens = [first]
            extra_tokens = remaining_tokens[1:]
        else:
            model_tokens = [first, second]
            extra_tokens = remaining_tokens[2:]
    elif len(remaining_tokens) == 1:
        model_tokens = [remaining_tokens[0]]
        extra_tokens = []
    else:
        # Fallback in rare malformed rows; keep model non-null when possible.
        model_tokens = [remaining_tokens[0]] if remaining_tokens else ["Unknown"]
        extra_tokens = remaining_tokens[1:] if len(remaining_tokens) > 1 else []

    model = " ".join(model_tokens).strip() or None
    extra = " ".join(extra_tokens).strip() or None

    return pd.Series(
        {
            "maker": maker or None,
            "model": model,
            "extra": extra,
        }
    )


def main() -> None:
    csv_path = "autoscout_de.csv"
    output_path = "autoscout_de_parsed.csv"

    # Load the CSV and treat empty strings as missing values.
    df = pd.read_csv(csv_path, na_values=["", " "], keep_default_na=True, low_memory=False)

    # A column is considered empty when all values are NaN.
    empty_columns = [col for col in df.columns if df[col].isna().all()]
    cleaned_df = df.drop(columns=empty_columns)

    parsed_make_model = cleaned_df["make_model"].apply(parse_make_model)
    cleaned_df = pd.concat([cleaned_df, parsed_make_model], axis=1)

    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved parsed CSV to: {output_path}")

    print("\n=== Parsed make_model (first 20 rows) ===")
    print(cleaned_df[["make_model", "maker", "model", "extra"]].head(20))

if __name__ == "__main__":
    main()
