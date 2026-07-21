# session_memory.py
import re
import streamlit as st
from interest_model import predict_intent

WINDOW_SIZE = 8
CONFIDENCE_THRESHOLD = 0.8

ACTION_ADD_TO_CART = "ADD_TO_CART"
ACTION_REMOVE_ITEM = "REMOVE_ITEM"
ACTION_CHECKOUT = "CHECKOUT"
ACTION_VIEW_MENU = "VIEW_MENU"
ACTION_ASK_ALLERGEN = "ASK_ALLERGEN"
ACTION_COMPARE_ITEMS = "COMPARE_ITEMS"
ACTION_GENERAL = "GENERAL"

ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "last": -1,
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Explicit reference phrases that mean "the thing you just showed me" —
# resolved to last_recommended_item, not to a position in a list.
PRONOUN_REFERENCE_PATTERNS = [
    r"\bprevious item\b", r"\bprevious one\b", r"\bprevious\b",
    r"\bthat item\b", r"\bthat one\b",
    r"\bthis one\b", r"\bthis item\b",
    r"\bsame item\b", r"\bsame one\b",
    r"\blast item\b", r"\blast one\b",
    r"\bit\b",  # "add it", "order it", "give me it"
]


def initialize_session_memory():
    if "session_memory" not in st.session_state:
        st.session_state.session_memory = {
            "state_line": "state: idle",
            "recent_turns": [],
            "order": {
                "selected_items": [],
                "pending_actions": [],
                "exclusions": [],
                "preferences": [],
                "restrictions": [],
                "last_recommended_item": None,
                "recommended_items": [],
                "last_recommendations": [],
            },
        }


def register_shown_items(items):
    """
    Store the exact recommendation list shown to the user, in display order.
    Caller (ui_components.py) is responsible for only invoking this on
    genuine menu-browsing turns with relevant search results — never on
    cart-action turns or low-relevance chit-chat, or it will overwrite the
    list ordinal/pronoun references depend on.
    """
    initialize_session_memory()
    order = st.session_state.session_memory["order"]

    snapshot = []
    # Cap to top 4 items — exactly matching the items presented on screen
    for idx, item in enumerate(items[:4], start=1):
        snapshot.append({
            "index": idx,
            "product_id": str(item.get("product_id")),
            "name": item.get("name", ""),
            "price": item.get("price"),
            "category": item.get("category"),
            "calories": item.get("calories"),
            "allergens": item.get("allergens"),
        })

    order["last_recommendations"] = snapshot

    if snapshot:
        # Update the pronoun anchor only when a single item was shown —
        # that's the only case where "it" / "that one" is unambiguous.
        # For multi-item lists, keep the existing pointer so a previous
        # single-item reference stays valid across detail-question turns.
        if len(snapshot) == 1:
            order["last_recommended_item"] = snapshot[0]
        for entry in snapshot:
            pid = entry["product_id"]
            if pid not in order["recommended_items"]:
                order["recommended_items"].append(pid)
        while len(order["recommended_items"]) > 12:
            order["recommended_items"].pop(0)

    st.session_state.session_memory["state_line"] = _build_state_line(
        st.session_state.session_memory
    )


def sync_shown_items_from_response(response_text: str):
    """
    Scans the bot's response text for any actual menu item names and updates
    last_recommendations and last_recommended_item in session_memory.
    This guarantees 100% synchronization between what the bot displays on screen
    and what "add it" / "1st item" resolves to, eliminating hallucinations.
    """
    if not response_text or not isinstance(response_text, str):
        return
    if "df" not in st.session_state or st.session_state.df is None:
        return

    df = st.session_state.df
    matched_items = []
    text_lower = response_text.lower()

    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if name and len(name) > 3 and name.lower() in text_lower:
            matched_items.append(row.to_dict())

    if matched_items:
        register_shown_items(matched_items)


def extract_quantity(text):
    text = text.lower()
    match = re.search(r"\b(\d+)\s+of\b", text)
    if match:
        return max(1, int(match.group(1)))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+of\b", text):
            return value
    cleaned = re.sub(r"\b\d+(st|nd|rd|th)\b", "", text)
    match = re.search(r"\b(\d+)\b", cleaned)
    if match:
        return max(1, int(match.group(1)))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", cleaned):
            return value
    return 1


def resolve_item_references(text):
    """
    Resolve references against the exact last_recommendations snapshot.
    Resolution order: numeric/ordinal position -> exact name mention ->
    pronoun reference ("previous", "that one", "it", etc.) -> last_recommended_item.

    Returns:
    {
        "items": [{"product_id": "...", "quantity": 1}, ...],
        "confidence": 0.0-1.0,
        "ambiguous": bool,
    }
    """
    initialize_session_memory()
    order = st.session_state.session_memory["order"]
    recommendations = order.get("last_recommendations", [])

    query = text.lower()
    quantity = extract_quantity(query)

    if not recommendations:
        # No indexed list to resolve against, but a pronoun reference can
        # still fall back to the single last_recommended_item, if we have one.
        last_item = order.get("last_recommended_item")
        if last_item and any(re.search(p, query) for p in PRONOUN_REFERENCE_PATTERNS):
            return {
                "items": [{"product_id": str(last_item["product_id"]), "quantity": quantity}],
                "confidence": 0.85,
                "ambiguous": False,
            }
        return {"items": [], "confidence": 0.0, "ambiguous": False}

    raw_indices = []
    for num in re.findall(r"#(\d+)", query):
        raw_indices.append(int(num))
    for num in re.findall(r"(\d+)(?:st|nd|rd|th)", query):
        raw_indices.append(int(num))
    for num in re.findall(r"item\s+(\d+)", query):
        raw_indices.append(int(num))
    for word, value in ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", query):
            raw_indices.append(value)

    raw_indices = list(dict.fromkeys(raw_indices))
    # Bulk quantity (e.g. "2 of the first one") only makes sense when a
    # single item is targeted. For multiple ordinals each gets quantity=1
    # so "order the 2nd and last" doesn't silently double both items.
    single_target = len(raw_indices) == 1

    resolved = []
    invalid_indices = []
    for idx in raw_indices:
        if idx == -1:
            item = recommendations[-1]
        elif 1 <= idx <= len(recommendations):
            item = recommendations[idx - 1]
        else:
            invalid_indices.append(idx)
            continue
        resolved.append({"product_id": str(item["product_id"]), "quantity": quantity if single_target else 1})

    if resolved:
        confidence = 1.0 if not invalid_indices else 0.5
        return {"items": resolved, "confidence": confidence, "ambiguous": bool(invalid_indices)}

    if invalid_indices:
        return {"items": [], "confidence": 0.0, "ambiguous": False}

    # Direct name match
    name_matches = [
        item for item in recommendations
        if item["name"] and item["name"].lower() in query
    ]
    if len(name_matches) == 1:
        item = name_matches[0]
        return {
            "items": [{"product_id": str(item["product_id"]), "quantity": quantity}],
            "confidence": 0.9,
            "ambiguous": False,
        }
    if len(name_matches) > 1:
        return {
            "items": [
                {"product_id": str(m["product_id"]), "quantity": quantity}
                for m in name_matches
            ],
            "confidence": 0.4,
            "ambiguous": True,
        }

    # Pronoun reference — "the previous item", "that one", "it", etc.
    # Points at last_recommended_item specifically, which is stable across
    # filler/chit-chat turns as long as the caller gates registration correctly.
    last_item = order.get("last_recommended_item")
    if last_item and any(re.search(p, query) for p in PRONOUN_REFERENCE_PATTERNS):
        return {
            "items": [{"product_id": str(last_item["product_id"]), "quantity": quantity}],
            "confidence": 0.85,
            "ambiguous": False,
        }

    return {"items": [], "confidence": 0.0, "ambiguous": False}


def _describe_candidates(items):
    initialize_session_memory()
    recs = st.session_state.session_memory["order"].get("last_recommendations", [])
    names = []
    for it in items:
        match = next((r for r in recs if str(r["product_id"]) == str(it["product_id"])), None)
        names.append(match["name"] if match else it["product_id"])
    return " or ".join(names)


def classify_action(text, parsed_intent=None):
    if parsed_intent and parsed_intent.action:
        return parsed_intent.action

    lowered = text.lower()

    if any(p in lowered for p in [
        "checkout", "place order", "place my order", "finalize order",
        "pay now", "complete order", "order it", "order my cart",
    ]):
        return ACTION_CHECKOUT
    if any(p in lowered for p in ["allerg", "contains", "gluten", "dairy", "nut", "vegan", "vegetarian", "ingredient"]):
        return ACTION_ASK_ALLERGEN
    if any(p in lowered for p in ["compare", " vs ", "versus", "difference between", "healthier", "which is better"]):
        return ACTION_COMPARE_ITEMS
    if any(p in lowered for p in ["show", "menu", "recommend", "suggest", "options", "what do you have", "looking for", "give me", "find me", "something", "dishes", "craving"]):
        return ACTION_VIEW_MENU
    if _is_negative_intent(lowered):
        return ACTION_REMOVE_ITEM
    if _is_positive_intent(lowered):
        return ACTION_ADD_TO_CART

    try:
        intent, conf = predict_intent(text)
        if intent == "positive" and conf > 0.65:
            return ACTION_VIEW_MENU
    except Exception:
        pass

    return ACTION_GENERAL


def resolve_item_references(user_text, parsed_intent=None):
    initialize_session_memory()
    order = st.session_state.session_memory["order"]
    recommendations = order.get("last_recommendations", [])

    quantity = parsed_intent.quantity if parsed_intent else 1
    query = (parsed_intent.target_reference or user_text).lower() if parsed_intent else user_text.lower()

    if not recommendations and not order.get("last_recommended_item"):
        return {"items": [], "confidence": 0.0, "ambiguous": False}

    raw_indices = []
    for num in re.findall(r"#(\d+)", query):
        raw_indices.append(int(num))
    for num in re.findall(r"(\d+)(?:st|nd|rd|th)", query):
        raw_indices.append(int(num))
    for num in re.findall(r"item\s+(\d+)", query):
        raw_indices.append(int(num))
    for word, value in ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", query):
            raw_indices.append(value)

    raw_indices = list(dict.fromkeys(raw_indices))
    single_target = len(raw_indices) == 1

    resolved = []
    invalid_indices = []
    for idx in raw_indices:
        if idx == -1 and recommendations:
            item = recommendations[-1]
        elif 1 <= idx <= len(recommendations):
            item = recommendations[idx - 1]
        else:
            invalid_indices.append(idx)
            continue
        resolved.append({"product_id": str(item["product_id"]), "quantity": quantity if single_target else 1})

    if resolved:
        if single_target and raw_indices and (1 <= raw_indices[0] <= len(recommendations) or raw_indices[0] == -1):
            order["last_recommended_item"] = recommendations[raw_indices[0] - 1]
        confidence = 1.0 if not invalid_indices else 0.5
        return {"items": resolved, "confidence": confidence, "ambiguous": bool(invalid_indices)}

    if invalid_indices:
        return {"items": [], "confidence": 0.0, "ambiguous": False}

    # Direct name match
    name_matches = [
        item for item in recommendations
        if item["name"] and (item["name"].lower() in query or query in item["name"].lower())
    ]
    if len(name_matches) == 1:
        item = name_matches[0]
        order["last_recommended_item"] = item
        return {
            "items": [{"product_id": str(item["product_id"]), "quantity": quantity}],
            "confidence": 0.9,
            "ambiguous": False,
        }
    if len(name_matches) > 1:
        return {
            "items": [
                {"product_id": str(m["product_id"]), "quantity": quantity}
                for m in name_matches
            ],
            "confidence": 0.4,
            "ambiguous": True,
        }

    # Pronoun reference — "the previous item", "that one", "it", etc.
    last_item = order.get("last_recommended_item")
    if last_item:
        if any(re.search(p, query) for p in PRONOUN_REFERENCE_PATTERNS) or query in ["it", "that", "this", "that one", "this one"]:
            return {
                "items": [{"product_id": str(last_item["product_id"]), "quantity": quantity}],
                "confidence": 0.85,
                "ambiguous": False,
            }

    return {"items": [], "confidence": 0.0, "ambiguous": False}


def _describe_candidates(items):
    initialize_session_memory()
    recs = st.session_state.session_memory["order"].get("last_recommendations", [])
    names = []
    for it in items:
        match = next((r for r in recs if str(r["product_id"]) == str(it["product_id"])), None)
        names.append(match["name"] if match else it["product_id"])
    return " or ".join(names)


def _add_items_to_cart(order, items):
    for resolved_item in items:
        product_id = resolved_item["product_id"]
        quantity = resolved_item.get("quantity", 1)
        existing = next(
            (i for i in order["selected_items"] if i["product_id"] == product_id), None
        )
        if existing:
            existing["quantity"] += quantity
        else:
            order["selected_items"].append({"product_id": product_id, "quantity": quantity})
        order["pending_actions"].append(f"added product {product_id}")


def _remove_items_from_cart(order, items):
    for resolved_item in items:
        product_id = resolved_item["product_id"]
        order["selected_items"] = [
            i for i in order["selected_items"] if i["product_id"] != product_id
        ]
        if product_id not in order["exclusions"]:
            order["exclusions"].append(product_id)
        order["pending_actions"].append(f"removed product {product_id}")


def update_state_from_user_message(user_text, parsed_intent=None):
    """
    Intent -> Reference resolution -> Action.
    """
    initialize_session_memory()
    memory = st.session_state.session_memory
    order = memory["order"]
    lowered = user_text.lower()

    intent_action = classify_action(user_text, parsed_intent=parsed_intent)
    result = {
        "action": intent_action,
        "cart_changed": False,
        "needs_clarification": False,
        "clarification_message": None,
    }

    # Run reference resolution for ALL turns to keep last_recommended_item anchor updated
    resolved = resolve_item_references(user_text, parsed_intent=parsed_intent)

    if intent_action == ACTION_CHECKOUT:
        order["pending_actions"].append("checkout requested")
        result["cart_changed"] = bool(order["selected_items"])

    elif intent_action in (ACTION_ADD_TO_CART, ACTION_REMOVE_ITEM):
        items = resolved["items"]
        confidence = resolved["confidence"]

        if items and (resolved["ambiguous"] or confidence < CONFIDENCE_THRESHOLD):
            result["needs_clarification"] = True
            result["clarification_message"] = (
                f"Just to confirm — did you mean {_describe_candidates(items)}? "
                "Let me know which one."
            )
            result["action"] = ACTION_GENERAL

        elif items and confidence >= CONFIDENCE_THRESHOLD:
            if intent_action == ACTION_ADD_TO_CART:
                _add_items_to_cart(order, items)
                # Persist anchor so subsequent "add it" targets this item
                for rec in order.get("last_recommendations", []):
                    if str(rec["product_id"]) == str(items[0]["product_id"]):
                        order["last_recommended_item"] = rec
                        break
            else:
                _remove_items_from_cart(order, items)
            result["cart_changed"] = True

        else:
            # Nothing resolved at all. Do NOT silently fall through to the
            # LLM here — that's exactly how you got "I've noted you'd like
            # to add this item" with nothing actually added. Ask instead.
            result["needs_clarification"] = True
            result["clarification_message"] = (
                "I couldn't tell which item you mean — could you name it, or "
                "refer to it by position from the last list I showed you "
                "(e.g. \"the 2nd one\")?"
            )
            result["action"] = ACTION_GENERAL

    for pref in _extract_preferences(lowered):
        if pref not in order["preferences"]:
            order["preferences"].append(pref)

    for restriction in _extract_explicit_restrictions(lowered):
        if restriction not in order["restrictions"]:
            order["restrictions"].append(restriction)

    memory["state_line"] = _build_state_line(memory)
    return result


def record_turn(role, content):
    initialize_session_memory()
    turns = st.session_state.session_memory["recent_turns"]
    turns.append({"role": role, "content": content})
    while len(turns) > WINDOW_SIZE:
        turns.pop(0)


def build_memory_context():
    initialize_session_memory()
    memory = st.session_state.session_memory
    order = memory["order"]

    order_items_str = "None"
    if order["selected_items"]:
        df = st.session_state.df
        item_details = []
        for cart_item in order["selected_items"]:
            product_id = str(cart_item["product_id"])
            quantity = cart_item.get("quantity", 1)
            match = df[df["product_id"].astype(str) == product_id]
            if not match.empty:
                row = match.iloc[0]
                name = row.get("name", "Unknown")
                try:
                    price = f"${float(row.get('price', 0)):.2f}"
                except Exception:
                    price = "N/A"
                calories = row.get("calories", "N/A")
                allergens = row.get("allergens", "None")
                item_details.append(
                    f"{quantity} × {name} ({price}, {calories} cal, allergens: {allergens})"
                )
            else:
                item_details.append(f"{quantity} × Unknown Item (ID={product_id})")
        order_items_str = "; ".join(item_details)

    recent_lines = [
        compressed
        for turn in memory.get("recent_turns", [])
        if (compressed := _compress_turn(turn))
    ]

    last_item = order.get("last_recommended_item")
    last_item_name = last_item["name"] if isinstance(last_item, dict) else (last_item or "None")

    context_parts = ["SESSION MEMORY:"]
    context_parts.append(memory["state_line"])
    context_parts.append(f"Ordered items: {order_items_str}")
    context_parts.append(
        f"Exclusions: {order['exclusions'] or 'None'}, "
        f"User dietary restrictions: {order.get('restrictions', []) or 'None'}, "
        f"Likes/Cravings: {order['preferences'] or 'None'}, "
        f"Last recommended item: {last_item_name}"
    )

    recs = order.get("last_recommendations", [])
    if recs:
        context_parts.append(
            "LAST RECOMMENDATIONS SHOWN (exact list, reference only — "
            "cart changes are handled in code, not by you):"
        )
        for entry in recs:
            price = entry.get("price")
            price_str = f"${float(price):.2f}" if price not in (None, "") else "N/A"
            context_parts.append(
                f"  {entry['index']}. {entry['name']} – {price_str}, "
                f"{entry['calories']} cal, {entry['category']}, allergens: {entry['allergens']}"
            )

    if recent_lines:
        context_parts.append("Recent Turns:")
        context_parts.extend(recent_lines)

    return "\n".join(context_parts)


def build_order_confirmation_message(action=ACTION_ADD_TO_CART):
    initialize_session_memory()
    order = st.session_state.session_memory["order"]
    df = st.session_state.df
    cart = order["selected_items"]

    if not cart:
        return "🛒 Your cart is now empty."

    lines = []
    if action == ACTION_ADD_TO_CART:
        lines.append("### ✅ Item(s) added to your cart\n")
    elif action == ACTION_REMOVE_ITEM:
        lines.append("### 🗑️ Item(s) removed from your cart\n")
    elif action == ACTION_CHECKOUT:
        lines.append("### 🧾 Checkout Summary\n")
    else:
        lines.append("### 🛒 Your Cart\n")

    total = 0.0
    for cart_item in cart:
        product_id = str(cart_item["product_id"])
        quantity = cart_item.get("quantity", 1)
        match = df[df["product_id"].astype(str) == product_id]
        if match.empty:
            continue
        row = match.iloc[0]
        name = row["name"]
        try:
            price = float(row["price"])
        except Exception:
            price = 0.0
        subtotal = quantity * price
        total += subtotal
        lines.append(f"- **{quantity} × {name}** — `${subtotal:.2f}`  ")

    lines.append("")
    lines.append(f"**Total : ${total:.2f}**\n")
    return "\n".join(lines)


def _compress_turn(turn):
    content = turn["content"].strip()
    if not content:
        return ""
    lowered = content.lower()
    if turn["role"] == "user":
        if _is_positive_intent(lowered):
            return "User showed interest in the current recommendation."
        if _is_negative_intent(lowered):
            return "User rejected the current recommendation."
        preferences = _extract_preferences(lowered)
        if preferences:
            return f"User mentioned preferences: {', '.join(preferences)}."
    return f"{turn['role'].capitalize()} said: {content[:90]}"


def _build_state_line(memory):
    order = memory["order"]
    state_parts = []
    df = st.session_state.get("df")

    if order["selected_items"]:
        selected = []
        for item in order["selected_items"]:
            product_id = str(item["product_id"])
            quantity = item.get("quantity", 1)
            name = product_id
            if df is not None:
                match = df[df["product_id"].astype(str) == product_id]
                if not match.empty:
                    name = match.iloc[0]["name"]
            selected.append(f"{quantity} × {name}")
        state_parts.append(f"selected={', '.join(selected)}")

    if order["pending_actions"]:
        state_parts.append(f"pending={', '.join(order['pending_actions'][-3:])}")

    if order["exclusions"]:
        excluded = []
        for product_id in order["exclusions"]:
            name = product_id
            if df is not None:
                match = df[df["product_id"].astype(str) == str(product_id)]
                if not match.empty:
                    name = match.iloc[0]["name"]
            excluded.append(name)
        state_parts.append(f"excluded={', '.join(excluded)}")

    if order["preferences"]:
        state_parts.append(f"prefs={', '.join(order['preferences'])}")

    if order["restrictions"]:
        state_parts.append(f"restrictions={', '.join(order['restrictions'])}")

    if order["last_recommended_item"]:
        last = order["last_recommended_item"]
        state_parts.append(f"last={last['name']}")

    if not state_parts:
        return "state: idle"

    return "state: " + "; ".join(state_parts)


def _is_positive_intent(text):
    text = text.lower().strip()
    negative_patterns = [
        r"\bdon't\b", r"\bdont\b", r"\bdo not\b", r"\bdidn'?t\b", r"\bdid not\b",
        r"\bnot\b", r"\bnever\b", r"\bno thanks\b", r"\bcancel\b",
        r"\bremove\b", r"\bdelete\b", r"\bdiscard\b", r"\bexclude\b",
    ]
    for pattern in negative_patterns:
        if re.search(pattern, text):
            return False

    positive_patterns = [
        r"\border\b", r"\badd\b", r"\bbuy\b", r"\bconfirm\b", r"\bcheckout\b",
        r"\bplace order\b", r"\bi'?ll take\b", r"\bi want to order\b", r"\bi want the\b",
        r"\bi want this\b", r"\bi want that\b", r"\bgive me the\b", r"\bget me the\b",
        r"\bgo with\b", r"\bmake it\b", r"\binclude\b", r"\bone more\b", r"\banother\b",
    ]
    for pattern in positive_patterns:
        if re.search(pattern, text):
            return True
    return False


def _is_negative_intent(text):
    text = text.lower().strip()
    negative_patterns = [
        r"\bremove\b", r"\bdelete\b", r"\bcancel\b", r"\bdiscard\b",
        r"\bexclude\b", r"\btake off\b", r"\bwithout\b", r"\bdrop\b",
        r"\bno thanks\b", r"\bnot interested\b", r"\bdon't want\b",
        r"\bdont want\b", r"\bdo not want\b", r"\bdidn'?t want\b",
        r"\bno\b", r"\bnah\b", r"\bskip\b", r"\bchange\b", r"\breplace\b",
        r"\bswap\b", r"\bdifferent\b",
    ]
    for pattern in negative_patterns:
        if re.search(pattern, text):
            return True
    return False


def _extract_preferences(text):
    lowered = text.lower()
    likes = []
    if any(phrase in lowered for phrase in ["spicy", "hot", "kick"]):
        likes.append("spicy")
    if any(phrase in lowered for phrase in ["sweet", "dessert", "sugar", "chocolate"]):
        likes.append("sweet")
    if any(phrase in lowered for phrase in ["crunchy", "crispy", "crunch"]):
        likes.append("crunchy")
    if any(phrase in lowered for phrase in ["comfort food", "hearty", "rich"]):
        likes.append("comfort")
    return likes


def _extract_explicit_restrictions(text):
    lowered = text.lower()
    restrictions = []
    if any(phrase in lowered for phrase in ["vegetarian", "no meat"]):
        restrictions.append("vegetarian")
    if any(phrase in lowered for phrase in ["vegan", "no animal products"]):
        restrictions.append("vegan")
    avoid_map = {
        "gluten": ["no gluten", "gluten free", "avoid gluten", "celiac"],
        "dairy":  ["no dairy", "dairy free", "lactose free", "milk allergy", "avoid dairy"],
        "nuts":   ["no nuts", "nut free", "peanut allergy", "tree nut allergy", "avoid nuts"],
        "soy":    ["no soy", "soy free", "avoid soy"],
        "fried":  ["avoid fried", "no fried", "not fried", "fried food"],
    }
    for tag, keywords in avoid_map.items():
        if any(k in lowered for k in keywords):
            restrictions.append(f"no_{tag}")
    return list(set(restrictions))