# session_memory.py
import re
import streamlit as st
from typing import Dict, List, Optional
from src.database.data_pipeline import get_recommendation_graph
from src.core.log import log_cart_action

# Hybrid Memory Configuration:
# - Sliding Buffer Window (K = 6 turns)
# - Deterministic Entity State (Structured Shopping List & Price Arithmetic)
BUFFER_WINDOW_SIZE = 6

ACTION_ADD_TO_CART = "ADD_TO_CART"
ACTION_REMOVE_ITEM = "REMOVE_ITEM"
ACTION_VIEW_CART = "VIEW_CART"
ACTION_CLEAR_CART = "CLEAR_CART"
ACTION_CHECKOUT = "CHECKOUT"
ACTION_VIEW_MENU = "VIEW_MENU"
ACTION_GENERAL = "GENERAL"

ORDINAL_WORDS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "last": -1,
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "half dozen": 6, "dozen": 12,
}

PRONOUN_REFERENCE_PATTERNS = [
    r"\bprevious item\b", r"\bprevious one\b", r"\bprevious\b",
    r"\bthat item\b", r"\bthat one\b",
    r"\bthis one\b", r"\bthis item\b",
    r"\bsame item\b", r"\bsame one\b",
    r"\blast item\b", r"\blast one\b",
    r"\bit\b", r"\bthat\b",
]


def initialize_session_memory():
    """Initializes the hybrid memory state in Streamlit session_state."""
    if "session_memory" not in st.session_state:
        st.session_state.session_memory = {
            "state_line": "state: idle",
            "recent_turns": [],  # Sliding buffer window (K = 6)
            "order": {
                "selected_items": [],  # Structured shopping list items
                "last_recommendations": [],  # Last shown search results (for ordinal '2nd one')
                "last_suggested_items": [],  # Active smart suggestions (for 'add both', 'yes add it')
                "last_recommended_item": None,  # Single-item anchor (for 'add it')
                "dietary_restrictions": [],
                "preferred_aisles": [],
            },
        }


def record_turn(role: str, content: str):
    """
    Appends a dialogue turn and enforces the Sliding Buffer Window (K = 6).
    """
    initialize_session_memory()
    mem = st.session_state.session_memory
    mem["recent_turns"].append({"role": role, "content": content})
    
    # Enforce Buffer Window Size K = 6
    if len(mem["recent_turns"]) > BUFFER_WINDOW_SIZE:
        mem["recent_turns"] = mem["recent_turns"][-BUFFER_WINDOW_SIZE:]


def add_item_to_cart(item_dict: dict, quantity: int = 1) -> dict:
    """
    Deterministically adds or updates an item in the structured shopping list.
    Computes exact math (zero hallucination).
    """
    initialize_session_memory()
    order = st.session_state.session_memory["order"]
    selected = order["selected_items"]
    
    pid = str(item_dict.get("product_id", "")).strip()
    name = str(item_dict.get("name", "")).strip()
    category = str(item_dict.get("category", "Pantry & Staples")).strip()
    unit = str(item_dict.get("unit", "")).strip()
    price = float(item_dict.get("price", 0.0))
    qty = max(1, int(quantity))

    # Check if item already in cart
    existing = next((item for item in selected if item["product_id"] == pid or item["name"].lower() == name.lower()), None)
    if existing:
        existing["quantity"] += qty
        existing["subtotal"] = round(existing["quantity"] * existing["price"], 2)
        added_entry = existing
    else:
        added_entry = {
            "product_id": pid,
            "name": name,
            "category": category,
            "unit": unit,
            "price": price,
            "quantity": qty,
            "subtotal": round(qty * price, 2),
        }
        selected.append(added_entry)

    # Compute Smart Suggestions for the added item
    suggestions = get_smart_suggestions_for_item(pid)
    order["last_suggested_items"] = suggestions
    total_val = get_cart_total()
    log_cart_action("ADD", name, qty, total_val)

    return {
        "item": added_entry,
        "quantity_added": qty,
        "cart_total": total_val,
        "smart_suggestions": suggestions,
    }


def remove_item_from_cart(target: str) -> Optional[dict]:
    """
    Dynamically identifies and removes an item from the cart using exact match,
    token overlap, or generalized dynamic string similarity against active cart items.
    """
    initialize_session_memory()
    order = st.session_state.session_memory["order"]
    selected = order["selected_items"]
    if not selected:
        return None

    target_lower = target.lower().strip()

    # 1. Handle confirmation / contextual references: 'yes', 'yes remove', 'remove it', 'that one'
    if target_lower in ("yes", "remove", "yes remove", "it", "that", "remove it", "hata do", "yes remove that") and selected:
        removed = selected.pop(-1)
        log_cart_action("REMOVE", removed["name"], removed["quantity"], get_cart_total())
        return removed

    # Strip generic action/filler words dynamically
    clean_target = re.sub(r"\b(remove|delete|hata do|from my cart|from cart|my list|from my card|from my car|from list|please|the)\b", "", target_lower).strip()
    if not clean_target:
        clean_target = target_lower

    removed = None

    # 2. Exact match by product_id
    for idx, item in enumerate(selected):
        if item["product_id"].lower() == clean_target:
            removed = selected.pop(idx)
            break

    # 3. Substring match against full item name or category
    if not removed:
        for idx, item in enumerate(selected):
            name_low = item["name"].lower()
            if clean_target in name_low or name_low in clean_target:
                removed = selected.pop(idx)
                break

    # 4. Token overlap match (e.g. user said "bread" and cart has "FarmFresh Whole Wheat Bread")
    if not removed:
        target_tokens = set(re.findall(r"\b\w{3,}\b", clean_target))
        for idx, item in enumerate(selected):
            name_tokens = set(re.findall(r"\b\w{3,}\b", item["name"].lower()))
            if target_tokens & name_tokens:
                removed = selected.pop(idx)
                break

    # 5. Generalized dynamic string similarity against active cart items (no hardcoded word lists)
    if not removed and clean_target:
        import difflib
        best_match_idx = -1
        best_score = 0.0

        for idx, item in enumerate(selected):
            # Check full name similarity
            sim = difflib.SequenceMatcher(None, clean_target, item["name"].lower()).ratio()
            if sim > best_score and sim >= 0.40:
                best_score = sim
                best_match_idx = idx
            
            # Check individual word similarity in item name
            for word in item["name"].lower().split():
                if len(word) >= 3:
                    w_sim = difflib.SequenceMatcher(None, clean_target, word).ratio()
                    if w_sim > best_score and w_sim >= 0.40:
                        best_score = w_sim
                        best_match_idx = idx

        if best_match_idx >= 0 and best_score >= 0.50:
            removed = selected.pop(best_match_idx)

    if removed:
        log_cart_action("REMOVE", removed["name"], removed["quantity"], get_cart_total())
    return removed


def clear_cart():
    """Clears all items from the structured shopping list."""
    initialize_session_memory()
    st.session_state.session_memory["order"]["selected_items"] = []
    st.session_state.session_memory["order"]["last_suggested_items"] = []
    log_cart_action("CLEAR_ALL", "All Items", 0, 0.0)


def get_cart_total() -> float:
    """Calculates exact cart total deterministically."""
    initialize_session_memory()
    selected = st.session_state.session_memory["order"]["selected_items"]
    return round(sum(item["subtotal"] for item in selected), 2)


def get_cart_items_count() -> int:
    """Returns total count of physical items in the cart."""
    initialize_session_memory()
    selected = st.session_state.session_memory["order"]["selected_items"]
    return sum(item["quantity"] for item in selected)


def get_cart_by_aisles() -> Dict[str, List[dict]]:
    """Groups cart items by their supermarket aisle category."""
    initialize_session_memory()
    selected = st.session_state.session_memory["order"]["selected_items"]
    aisles = {}
    for item in selected:
        cat = item.get("category", "Other Items")
        if cat not in aisles:
            aisles[cat] = []
        aisles[cat].append(item)
    return aisles


def get_smart_suggestions_for_item(product_id: str) -> List[dict]:
    """
    Retrieves cross-category paired products using the frequently_bought_together graph.
    Filters out items already in the shopping list.
    """
    if "df" not in st.session_state or st.session_state.df is None:
        return []
    
    df = st.session_state.df
    graph = get_recommendation_graph(df)
    paired_ids = graph.get(product_id, [])
    
    if not paired_ids:
        return []

    cart_pids = {item["product_id"] for item in st.session_state.session_memory["order"]["selected_items"]}
    
    suggestions = []
    for pid in paired_ids:
        if pid in cart_pids:
            continue
        row = df[df["product_id"] == pid]
        if not row.empty:
            rec = row.iloc[0].to_dict()
            suggestions.append({
                "product_id": str(rec.get("product_id")),
                "name": str(rec.get("name")),
                "category": str(rec.get("category")),
                "unit": str(rec.get("unit")),
                "price": float(rec.get("price", 0.0)),
            })
            if len(suggestions) >= 2:
                break

    return suggestions


def register_shown_items(items: List[dict]):
    """
    Stores an ordered snapshot of search results for positional voice commands ('add the 2nd one').
    """
    initialize_session_memory()
    order = st.session_state.session_memory["order"]

    snapshot = []
    for idx, item in enumerate(items[:6], start=1):
        snapshot.append({
            "index": idx,
            "product_id": str(item.get("product_id")),
            "name": str(item.get("name", "")),
            "category": str(item.get("category", "General")),
            "unit": str(item.get("unit", "")),
            "price": float(item.get("price", 0.0)),
            "dietary_tags": str(item.get("dietary_tags", "")),
            "description": str(item.get("description", "")),
        })
    order["last_recommendations"] = snapshot


def check_ordinal_intent(text: str, parsed_intent=None):
    """
    Checks for ordinal words ('first', 'second', 'third', 'last', etc.)
    Returns (word_found, index_requested, item_matched_or_none, total_items_shown).
    """
    order = st.session_state.session_memory.get("order", {})
    recs = order.get("last_recommendations") or order.get("last_suggested_items", [])
    lowered = text.lower()
    ref = (parsed_intent.target_reference if parsed_intent and parsed_intent.target_reference else "").lower()
    
    for word, idx in ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered) or word == ref or str(idx) in ref:
            if idx == -1:
                item = recs[-1] if recs else None
                return word, len(recs), item, len(recs)
            if 1 <= idx <= len(recs):
                return word, idx, recs[idx - 1], len(recs)
            else:
                alt_recs = order.get("last_suggested_items", [])
                if 1 <= idx <= len(alt_recs):
                    return word, idx, alt_recs[idx - 1], len(alt_recs)
                return word, idx, None, len(recs)
    return None, None, None, len(recs)


def resolve_ordinal_reference(text: str) -> Optional[dict]:
    """Resolves phrases like 'the second one' or 'first item' to the shown results."""
    _, _, item, _ = check_ordinal_intent(text)
    return item


def resolve_pronoun_reference(text: str) -> Optional[dict]:
    """Resolves 'add it', 'that one', 'add both' to recent suggestions."""
    order = st.session_state.session_memory.get("order", {})
    lowered = text.lower()

    # If the user specified an explicit ordinal position ('first', 'second', 'third', 'last'), DO NOT treat as pronoun 'it'
    for ord_w in ORDINAL_WORDS.keys():
        if re.search(rf"\b{re.escape(ord_w)}\b", lowered):
            return None

    # Check for 'add both'
    if "both" in lowered and order.get("last_suggested_items"):
        return order["last_suggested_items"]

    # Check if referring to active smart suggestion
    if any(re.search(pat, lowered) for pat in PRONOUN_REFERENCE_PATTERNS):
        if order.get("last_suggested_items"):
            return order["last_suggested_items"][0]
        if order.get("last_recommended_item"):
            return order["last_recommended_item"]

    return None


def build_memory_context() -> str:
    """
    Constructs a compact structured context block injected into the LLM system prompt.
    """
    initialize_session_memory()
    order = st.session_state.session_memory["order"]
    selected = order["selected_items"]
    total = get_cart_total()

    lines = ["--- SHOPPING LIST SESSION MEMORY ---"]
    if selected:
        lines.append(f"Current Cart Items ({len(selected)} products, Total: ${total:.2f}):")
        for it in selected:
            lines.append(f"  • {it['quantity']}x {it['name']} ({it['unit']}) – ${it['subtotal']:.2f} [{it['category']}]")
    else:
        lines.append("Current Cart: Empty (0 items, $0.00)")

    suggested = order.get("last_suggested_items", [])
    if suggested:
        sug_names = ", ".join(f"{s['name']} (${s['price']:.2f})" for s in suggested)
        lines.append(f"Active Smart Suggestions Offered: {sug_names}")

    restrictions = order.get("dietary_restrictions", [])
    if restrictions:
        lines.append(f"Active Dietary Preferences: {', '.join(restrictions)}")

    lines.append("------------------------------------")
    return "\n".join(lines)


def build_order_confirmation_message(action: str, item_info: Optional[dict] = None) -> str:
    """
    Builds clean, structured markdown confirmation with smart suggestions for the user.
    """
    total = get_cart_total()
    total_count = get_cart_items_count()

    if action == ACTION_CLEAR_CART:
        return "🧹 **Shopping list cleared.** Your cart is now empty."

    if action == ACTION_REMOVE_ITEM:
        name = item_info.get("name", "Item") if item_info else "Item"
        return f"🗑️ Removed **{name}** from your shopping list.\n\n🛒 **Current List ({total_count} items • \\${total:.2f})**."

    if action == ACTION_ADD_TO_CART and item_info:
        item = item_info["item"]
        qty = item_info["quantity_added"]
        suggestions = item_info.get("smart_suggestions", [])

        msg = f"🛒 Added **{qty}x {item['name']}** ({item['unit']}) to your list! (Subtotal: **\\${total:.2f}**)\n\n"
        
        # Add Smart Cross-Aisle Suggestions
        if suggestions:
            sug_text = " or ".join(f"**{s['name']}** (\\${s['price']:.2f})" for s in suggestions)
            msg += f"💡 **Smart Suggestion**: Shoppers who buy {item['name']} often also need {sug_text}. Would you like to add any of these?"

        return msg

    if action == ACTION_VIEW_CART:
        if total_count == 0:
            return "🛒 **Your shopping list is currently empty.** Click the mic or type to add items!"
        
        aisles = get_cart_by_aisles()
        msg = f"🛒 **Your Shopping List ({total_count} items • \\${total:.2f})**:\n\n"
        for aisle, items in aisles.items():
            msg += f"**{aisle}:**\n"
            for it in items:
                msg += f"• **{it['quantity']}x {it['name']}** ({it['unit']}) – \\${it['subtotal']:.2f}\n"
            msg += "\n"
        msg += f"**Total: \\${total:.2f}**\n\n*Say 'Checkout' or 'Clear list' whenever you are ready!*"
        return msg

    if action == ACTION_CHECKOUT:
        if total_count == 0:
            return "Your cart is empty! Add some groceries before checking out."
        return f"🎉 **Order Placed Successfully!**\n\nYour shopping list of **{total_count} items** totaling **\\${total:.2f}** has been confirmed for delivery."

    return "Cart updated."


def update_state_from_user_message(user_query: str, parsed_intent=None) -> dict:
    """
    Deterministic state router that executes cart operations and resolves ambiguous references.
    """
    initialize_session_memory()
    order = st.session_state.session_memory["order"]
    action = parsed_intent.action if parsed_intent else ACTION_GENERAL
    q_lower = user_query.lower()

    # Fast Match: Clear Cart (English & Hindi: 'clear list', 'sab clear kar do', 'kuchh nahin chahie')
    if action == ACTION_CLEAR_CART or any(w in q_lower for w in ["clear", "empty cart", "empty list", "sab clear", "kuchh nahin", "kuch nahi", "kuchh nhi", "sab hata do"]):
        clear_cart()
        return {
            "action": ACTION_CLEAR_CART,
            "cart_changed": True,
            "needs_clarification": False,
            "confirmation_override": build_order_confirmation_message(ACTION_CLEAR_CART),
        }

    # Fast Match: View Cart (English & Hindi: 'what is in my cart', 'updated cart', 'cart dikhao')
    if action == ACTION_VIEW_CART or any(w in q_lower for w in ["what is in my cart", "show cart", "view cart", "updated cart", "mera cart", "list dikhao", "cart dikhao"]):
        return {
            "action": ACTION_VIEW_CART,
            "cart_changed": False,
            "needs_clarification": False,
            "confirmation_override": build_order_confirmation_message(ACTION_VIEW_CART),
        }

    # Check if this is a compound addition command ("Add 2 milk and checkout")
    has_add_intent = bool(parsed_intent and parsed_intent.items) or any(w in q_lower for w in ["add", "put", "need", "dalo", "chahiye", "le lo"])

    # Fast Match: Checkout (English & Hindi: 'checkout', 'check out', 'scout', 'place order')
    if (action == ACTION_CHECKOUT or any(w in q_lower for w in ["checkout", "check out", "scout", "place order", "order place", "order kar do", "final order"])) and not has_add_intent:
        return {
            "action": ACTION_CHECKOUT,
            "cart_changed": False,
            "needs_clarification": False,
            "confirmation_override": build_order_confirmation_message(ACTION_CHECKOUT),
        }

    # 1. Check for Ordinal References ('add the second one', 'first item', 'I got it at the first item')
    ord_word, ord_idx, ord_item, total_shown = check_ordinal_intent(user_query, parsed_intent=parsed_intent)
    if ord_word is not None and (action in (ACTION_ADD_TO_CART, ACTION_GENERAL) or any(w in q_lower for w in ["add", "put", "take", "want", "at", "get", "one", "item"])):
        if ord_item:
            res = add_item_to_cart(ord_item, quantity=1)
            return {
                "action": ACTION_ADD_TO_CART,
                "cart_changed": True,
                "needs_clarification": False,
                "added_info": res,
                "confirmation_override": f"🛒 Added **1x {ord_item['name']}** ({ord_item.get('unit', '')}) to your shopping list! (Subtotal: **\\${get_cart_total():.2f}**)",
            }
        else:
            if total_shown == 0:
                return {
                    "action": ACTION_GENERAL,
                    "cart_changed": False,
                    "needs_clarification": False,
                    "confirmation_override": "There are no previous product recommendations active. Please search for an item first (e.g., *\"Show organic tea\"*).",
                }
            elif total_shown == 1:
                only_item = st.session_state.session_memory["order"]["last_recommendations"][0]["name"]
                return {
                    "action": ACTION_GENERAL,
                    "cart_changed": False,
                    "needs_clarification": False,
                    "confirmation_override": f"There is no **{ord_word}** item in the list shown above. Only **1 item** was found: **{only_item}**. Say *\"add it\"* to include it!",
                }
            else:
                return {
                    "action": ACTION_GENERAL,
                    "cart_changed": False,
                    "needs_clarification": False,
                    "confirmation_override": f"There is no **{ord_word}** item. The previous list only contains **{total_shown} items**.",
                }

    # 2. Check for Pronoun / Smart Suggestion References ('add both', 'add it', 'yes add that')
    if action in (ACTION_ADD_TO_CART, ACTION_GENERAL):
        pronoun_match = resolve_pronoun_reference(user_query)
        if pronoun_match:
            if isinstance(pronoun_match, list):  # 'add both'
                added_names = []
                for p in pronoun_match:
                    add_item_to_cart(p, quantity=1)
                    added_names.append(p["name"])
                order["last_suggested_items"] = []
                names_str = " and ".join(f"**{n}**" for n in added_names)
                return {
                    "action": ACTION_ADD_TO_CART,
                    "cart_changed": True,
                    "needs_clarification": False,
                    "confirmation_override": f"🛒 Added both {names_str} to your shopping list! (Total: **${get_cart_total():.2f}**)",
                }
            else:
                res = add_item_to_cart(pronoun_match, quantity=1)
                order["last_suggested_items"] = []
                return {
                    "action": ACTION_ADD_TO_CART,
                    "cart_changed": True,
                    "needs_clarification": False,
                    "added_info": res,
                    "confirmation_override": f"🛒 Added **1x {pronoun_match['name']}** ({pronoun_match.get('unit', '')}) to your shopping list! (Subtotal: **${get_cart_total():.2f}**)",
                }

    # 5. Remove Item Action
    if action == ACTION_REMOVE_ITEM:
        target = parsed_intent.cleaned_search_query if parsed_intent else user_query
        removed = remove_item_from_cart(target)
        if removed:
            return {
                "action": ACTION_REMOVE_ITEM,
                "cart_changed": True,
                "needs_clarification": False,
                "removed_item": removed,
                "confirmation_override": build_order_confirmation_message(ACTION_REMOVE_ITEM, removed),
            }
        else:
            return {
                "action": ACTION_REMOVE_ITEM,
                "cart_changed": False,
                "needs_clarification": False,
                "confirmation_override": f"Couldn't find '{target}' in your shopping list.",
            }

    # 6. Checkout Action
    if action == ACTION_CHECKOUT:
        return {
            "action": ACTION_CHECKOUT,
            "cart_changed": True,
            "needs_clarification": False,
            "confirmation_override": build_order_confirmation_message(ACTION_CHECKOUT),
        }

    return {
        "action": action,
        "cart_changed": False,
        "needs_clarification": False,
    }


def extract_allergen_restrictions(user_query: str) -> List[str]:
    """Extracts dietary preferences from the query to persist in memory."""
    lowered = user_query.lower()
    found = []
    tags = ["gluten_free", "vegan", "vegetarian", "keto_friendly", "organic", "sugar_free", "lactose_free", "low_calorie"]
    for t in tags:
        clean_tag = t.replace("_", " ")
        if clean_tag in lowered:
            found.append(t)
    return found


def sync_shown_items_from_response(response_text: str):
    """
    Scans the bot's response text for product names in the exact order they were displayed.
    Updates last_recommendations so 'add the second one' matches exactly what the user saw on screen!
    """
    if not response_text or "df" not in st.session_state or st.session_state.df is None:
        return
    
    df = st.session_state.df
    text_lower = response_text.lower()
    matched = []
    
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if name and len(name) > 3 and name.lower() in text_lower:
            pos = text_lower.find(name.lower())
            matched.append((pos, row.to_dict()))
            
    matched.sort(key=lambda x: x[0])
    ordered_items = [item for _, item in matched]
    if ordered_items:
        register_shown_items(ordered_items)