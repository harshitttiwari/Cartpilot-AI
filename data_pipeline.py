# data_pipeline.py
import re
import pandas as pd

CORE_COLUMNS = ["product_id", "name", "category", "unit", "price", "description"]
NUMERIC_COLUMNS = ["price"]

# Multilingual and phonetic synonym mappings for grocery items
# Supports English, Hindi/Hinglish, and Spanish voice queries
MULTILINGUAL_SYNONYMS = {
    "milk": ["doodh", "dudh", "leche", "dairy", "cream"],
    "cream": ["malai", "whipped cream", "crema", "heavy cream"],
    "cheese": ["paneer", "queso", "cheddar", "mozzarella", "cottage cheese"],
    "paneer": ["cottage cheese", "cheese", "paneer", "dairy"],
    "butter": ["makkhan", "makhan", "mantequilla"],
    "yogurt": ["dahi", "curd", "yogur", "greek yogurt", "buttermilk", "chaas"],
    "buttermilk": ["chaas", "mattha", "buttermilk", "dahi"],
    "eggs": ["anda", "ande", "huevos", "egg"],
    "egg": ["anda", "ande", "huevos", "eggs"],
    "bread": ["roti", "pav", "pan", "loaf", "toast", "bun", "bagel"],
    "bagel": ["bread", "bakery", "bagels", "bun"],
    "pav": ["bread", "bun", "pav", "roti", "bakery"],
    "croissant": ["bakery", "pastry", "croissant"],
    "apple": ["seb", "saeb", "manzana", "apples", "fruit"],
    "banana": ["kela", "keley", "platano", "bananas", "fruit"],
    "orange": ["santara", "narangi", "naranja", "orange juice", "fruit"],
    "papaya": ["papita", "papaya", "fruit"],
    "lemon": ["nimbu", "limon", "citrus", "lemon"],
    "carrot": ["gajar", "zanahoria", "vegetable", "carrots"],
    "garlic": ["lahsun", "lasun", "ajo", "garlic cloves"],
    "spinach": ["palak", "espinaca", "greens", "spinach"],
    "tomato": ["tamatar", "tomate", "vegetables"],
    "potato": ["aloo", "alu", "papa", "patata"],
    "onion": ["pyaz", "piaz", "cebolla"],
    "oil": ["tel", "cooking oil", "aceite", "mustard oil", "olive oil", "sunflower oil"],
    "poha": ["flattened rice", "chivda", "poha", "aval", "breakfast"],
    "flour": ["atta", "maida", "besan", "harina", "wheat flour"],
    "rice": ["chawal", "arroz", "basmati", "grain"],
    "pasta": ["noodles", "macaroni", "spaghetti", "pasta"],
    "ketchup": ["sauce", "tomato sauce", "salsa de tomate"],
    "coffee": ["cafe", "espresso", "brew", "caffeine"],
    "tea": ["chai", "cha", "te", "green tea"],
    "water": ["pani", "agua", "mineral water", "mineral"],
    "juice": ["ras", "jugo", "fruit juice"],
    "cashews": ["kaju", "anacardos", "nuts", "dry fruits"],
    "trail mix": ["nuts", "snack", "dry fruits", "seeds"],
    "sugar": ["cheeni", "chini", "shakar", "azucar"],
    "salt": ["namak", "sal", "sodium"],
}


def extract_multilingual_aliases(text: str) -> str:
    """Extracts multilingual synonyms (Hindi/Spanish) matching tokens in text."""
    lowered = text.lower()
    matched_aliases = set()
    for keyword, aliases in MULTILINGUAL_SYNONYMS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            matched_aliases.update(aliases)
    return " ".join(matched_aliases)


def analyze_menu_data(df: pd.DataFrame) -> dict:
    """Return a compact quality summary for the loaded grocery catalog."""
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
        "categories": df["category"].value_counts().to_dict() if "category" in df.columns else {},
    }


def clean_menu_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the CSV data before the app builds embeddings and search indices."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower().str.replace(" ", "_")

    # Clean numeric columns
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0.0).round(2)

    # Deduplicate by product_id
    if "product_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["product_id"], keep="first")

    # Clean text columns
    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    # Normalize frequently_bought_together (ensure no NaNs)
    if "frequently_bought_together" in cleaned.columns:
        cleaned["frequently_bought_together"] = cleaned["frequently_bought_together"].fillna("")

    # Drop rows missing core required fields
    required_columns = [column for column in CORE_COLUMNS if column in cleaned.columns]
    if required_columns:
        cleaned = cleaned.dropna(subset=required_columns)

    # Build rich multilingual search text
    cleaned["search_text"] = _build_search_text(cleaned)
    return cleaned.reset_index(drop=True)


def load_and_process_menu_data(csv_path: str):
    """Load the CSV, analyze it, clean it, and return the processed dataframe."""
    raw_df = pd.read_csv(csv_path)
    analysis = analyze_menu_data(raw_df)
    processed_df = clean_menu_data(raw_df)
    analysis["processed_rows"] = int(len(processed_df))
    analysis["removed_rows"] = analysis["rows"] - analysis["processed_rows"]
    return processed_df, analysis


def _build_search_text(df: pd.DataFrame) -> pd.Series:
    """Combines catalog fields and multilingual synonyms into a high-density search string."""
    search_texts = []
    for _, row in df.iterrows():
        name = str(row.get("name", ""))
        category = str(row.get("category", ""))
        unit = str(row.get("unit", ""))
        diet = str(row.get("dietary_tags", ""))
        desc = str(row.get("description", ""))
        
        # Extract multilingual aliases for food terms
        aliases = extract_multilingual_aliases(f"{name} {category} {desc}")
        
        combined = f"{name} {category} {unit} {diet} {aliases} {desc}"
        cleaned_text = re.sub(r"\s+", " ", combined).strip()
        search_texts.append(cleaned_text)
        
    return pd.Series(search_texts, index=df.index)


_CACHED_GRAPH = None


def get_recommendation_graph(df: pd.DataFrame) -> dict:
    """
    Returns an in-memory dictionary mapping product_id -> list of paired product_ids.
    Enables O(1) instant lookup for smart suggestions.
    """
    global _CACHED_GRAPH
    if _CACHED_GRAPH is not None:
        return _CACHED_GRAPH

    graph = {}
    if "product_id" not in df.columns or "frequently_bought_together" not in df.columns:
        return graph

    for _, row in df.iterrows():
        pid = str(row["product_id"]).strip()
        raw_pairs = str(row["frequently_bought_together"]).strip()
        if raw_pairs and raw_pairs != "nan":
            pair_list = [p.strip() for p in raw_pairs.split(";") if p.strip()]
            graph[pid] = pair_list
        else:
            graph[pid] = []
            
    _CACHED_GRAPH = graph
    return _CACHED_GRAPH