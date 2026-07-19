import pandas as pd

CORE_COLUMNS = ["product_id", "name", "category", "description", "ingredients", "price"]
NUMERIC_COLUMNS = ["price", "calories", "popularity_score", "spice_level"]
BOOLEAN_COLUMNS = ["chef_special", "limited_time"]

def analyze_menu_data(df):
    """Return a compact quality summary for the loaded menu data."""
    missing_core = {
        column: int(df[column].isna().sum())
        for column in CORE_COLUMNS
        if column in df.columns
    }
    duplicate_product_ids = 0
    if "product_id" in df.columns:
        duplicate_product_ids = int(df["product_id"].duplicated().sum())

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_core_values": missing_core,
        "duplicate_product_ids": duplicate_product_ids,
    }


def clean_menu_data(df):
    """Normalize the CSV data before the app builds embeddings or shows it."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower().str.replace(" ", "_")

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in BOOLEAN_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "yes": True, "no": False})
                .fillna(False)
            )

    required_columns = [column for column in CORE_COLUMNS if column in cleaned.columns]
    if required_columns:
        cleaned = cleaned.drop_duplicates(subset=["product_id"], keep="first")
        cleaned = cleaned.dropna(subset=required_columns)

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    cleaned = cleaned.fillna("")
    cleaned["search_text"] = _build_search_text(cleaned)
    return cleaned.reset_index(drop=True)


def load_and_process_menu_data(csv_path):
    """Load the CSV, analyze it, clean it, and return the processed dataframe."""
    raw_df = pd.read_csv(csv_path)
    analysis = analyze_menu_data(raw_df)
    processed_df = clean_menu_data(raw_df)
    analysis["processed_rows"] = int(len(processed_df))
    analysis["removed_rows"] = analysis["rows"] - analysis["processed_rows"]
    return processed_df, analysis


def _build_search_text(df):
    text_columns = [
        column
        for column in ["name", "category", "description", "ingredients", "dietary_tags", "mood_tags", "allergens"]
        if column in df.columns
    ]
    if not text_columns:
        return pd.Series([""] * len(df), index=df.index)

    return (
        df[text_columns]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )