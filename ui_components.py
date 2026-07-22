# ui_components.py
import streamlit as st
import time
from datetime import datetime
from bot_logic import get_ai_response, calculate_interest_score, parse_intent_with_llm
from database import DATA_FILE_PATH
from log import log_embedding_generated, log_vector_search
from session_memory import (
    build_memory_context,
    build_order_confirmation_message,
    initialize_session_memory,
    record_turn,
    register_shown_items,
    sync_shown_items_from_response,
    update_state_from_user_message,
)

CART_MUTATING_ACTIONS = {"ADD_TO_CART", "REMOVE_ITEM", "CHECKOUT"}
MENU_RELEVANT_ACTIONS = {"VIEW_MENU", "ASK_ALLERGEN", "COMPARE_ITEMS"}

# Relevance thresholds. Below these a top match is noise, not a real
# recommendation. Tune against your own corpus if scores run differently.
RELEVANCE_THRESHOLD = 0.30
KEYWORD_OVERLAP_THRESHOLD = 0.34


# ----------------- Context Building Helpers -----------------

def _build_enhanced_context(user_query, search_results):
    """Build readable context for the LLM from database search results."""
    if not search_results or not search_results.get("metadatas") or not search_results["metadatas"][0]:
        return "No relevant items found in the menu."

    allergen_restrictions = _detect_allergen_restrictions(user_query)
    request_type = _analyze_request_type(user_query)
    filtered_items = _filter_items_by_restrictions(search_results["metadatas"][0], allergen_restrictions)
    categorized_items = _categorize_items(filtered_items, request_type)

    if not filtered_items:
        allergen_msg = f" (avoiding {', '.join(allergen_restrictions)})" if allergen_restrictions else ""
        return f"No suitable menu items found{allergen_msg}."

    context = "Here are the relevant menu items:\n\n"
    for category, items in categorized_items.items():
        if not items:
            continue
        context += f"**{category.upper()}:**\n"
        for item in items:
            price = _format_price(item.get("price"))
            allergens = item.get("allergens", "None listed") or "None listed"
            allergens_display = "No allergens listed" if allergens == "None listed" else f"Contains: {allergens}"
            calories = item.get("calories", "N/A")

            context += f"• {item.get('name', 'N/A')} – {price}\n"
            if calories and calories != "N/A":
                context += f"    Calories: {calories}\n"
            context += f"    Category: {item.get('category', 'N/A')}\n"
            context += f"    Allergens: {allergens_display}\n"
        context += "\n"
    return context


def _format_price(value):
    try:
        return f"${float(value):.2f}"
    except Exception:
        return "Price N/A"


def _detect_allergen_restrictions(user_query):
    query = user_query.lower()
    allergen_keywords = {
        "soy":    ["no soy", "without soy", "avoid soy", "soy free", "soy allergy"],
        "gluten": ["no gluten", "without gluten", "avoid gluten", "gluten free", "celiac"],
        "dairy":  ["no dairy", "without dairy", "avoid dairy", "dairy free", "lactose", "milk allergy", "no milk", "without milk"],
        "nuts":   ["no nuts", "without nuts", "avoid nuts", "nut free", "peanut allergy", "tree nut allergy", "no peanuts"],
        "egg":    ["no egg", "without egg", "avoid egg", "egg free"],
        "fish":   ["no fish", "without fish", "avoid fish", "fish free", "seafood allergy"],
        "sesame": ["no sesame", "without sesame", "avoid sesame", "sesame free"],
    }
    return [a for a, kws in allergen_keywords.items() if any(k in query for k in kws)]


def _analyze_request_type(user_query):
    query = user_query.lower()
    keywords = {
        "main_dish": ["meal", "dish", "dinner", "lunch"],
        "snack":     ["snack", "appetizer", "light"],
        "drink":     ["drink", "beverage", "thirsty", "cooling", "refreshing", "refresher", "lemonade", "shake", "smoothie"],
        "sweet":     ["sweet", "dessert"],
    }
    return [t for t, kws in keywords.items() if any(k in query for k in kws)]


def _filter_items_by_restrictions(items, restrictions):
    if not restrictions:
        return items
    return [
        item for item in items
        if not any(r in (item.get("allergens", "").lower()) for r in restrictions)
    ]


def _categorize_items(items, request_types):
    categories = {"main_dishes": [], "appetizers_snacks": [], "beverages": [], "desserts": []}
    for item in items:
        cat = (item.get("category") or "").lower()
        name = (item.get("name") or "").lower()

        if cat in ["burgers", "pizza", "tacos & wraps", "salads & healthy options",
                   "breakfast items", "fried chicken"]:
            categories["main_dishes"].append(item)
        elif cat == "sides & appetizers" or any(x in name for x in ["fries", "chips", "bites", "rings", "poppers"]):
            categories["appetizers_snacks"].append(item)
        elif cat == "beverages":
            categories["beverages"].append(item)
        elif cat == "desserts":
            categories["desserts"].append(item)
        else:
            categories["main_dishes"].append(item)

    if "main_dish" in request_types:
        return {"main_dishes": categories["main_dishes"]}
    if "snack" in request_types:
        return {"appetizers_snacks": categories["appetizers_snacks"]}
    if "drink" in request_types:
        return {"beverages": categories["beverages"]}
    return categories


# ----------------- UI Rendering -----------------


def _keyword_overlap_score(prompt, item_metadata):
    """
    Deterministic fallback relevance signal — plain token overlap between the
    query and an item's searchable text. Unlike the ANN vector score (which
    can vary slightly between identical calls due to HNSW's approximate
    nature), this is 100% reproducible, so it's combined with the vector
    score to stop identical queries from flip-flopping between "found" and
    "not found".
    """
    query_tokens = set(prompt.lower().split())
    text = " ".join(
        str(item_metadata.get(k, ""))
        for k in ("name", "category", "description", "ingredients", "dietary_tags")
    )
    item_tokens = set(text.lower().split())
    if not query_tokens or not item_tokens:
        return 0.0
    overlap = query_tokens & item_tokens
    return len(overlap) / max(1, len(query_tokens))


def render_chat_interface(container):
    with container:

        initialize_session_memory()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if not st.session_state.chat_history:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Welcome to Foodie Assistant Bot! Ask me about menu items, recommendations, allergens, or drinks.",
            })
        else:
            st.session_state.chat_history[0]["content"] = "Welcome to Foodie Assistant Bot! Ask me about menu items, recommendations, allergens, or drinks."

        if "interest_score" not in st.session_state:
            st.session_state.interest_score = 50

        if "interest_history" not in st.session_state:
            st.session_state.interest_history = [50]

        if "query_log" not in st.session_state:
            st.session_state.query_log = []

        chat_history_container = st.container(height=520)

        with chat_history_container:
            # Render only the last 50 messages to prevent UI slowdown on long
            # sessions. Full history stays in session_state for LLM context.
            for msg in st.session_state.chat_history[-50:]:
                avatar_icon = "🤖" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar_icon):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask me about the menu..."):

            record_turn("user", prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                start = time.time()
                api_success = False

                # STEP 1: Attempt to call FastAPI REST Backend Server first
                try:
                    import requests
                    res = requests.post(
                        "http://127.0.0.1:8000/api/chat",
                        json={"user_input": prompt, "session_id": "streamlit_session"},
                        timeout=5.0
                    )
                    if res.status_code == 200:
                        data = res.json()
                        response = data["bot_response"]
                        action_type = data["action"]
                        st.session_state.interest_score = data["interest_score"]
                        duration = data["latency_ms"] / 1000.0
                        top_match = "FastAPI Backend"
                        match_score = 1.0
                        api_success = True
                except Exception:
                    api_success = False

                # STEP 2: Fallback to direct Python pipeline if FastAPI server is offline
                if not api_success:
                    parsed_intent = parse_intent_with_llm(st.session_state.llm, prompt)
                    resolved_action = update_state_from_user_message(prompt, parsed_intent=parsed_intent)
                    action_type = resolved_action["action"]

                    should_search = action_type not in CART_MUTATING_ACTIONS
                    search_results = None
                    top_relevance = 0.0
                    keyword_signal = 0.0

                    if should_search:
                        search_prompt = parsed_intent.cleaned_search_query or prompt
                        search_results = _hybrid_search(search_prompt, parsed_intent=parsed_intent)
                        if (
                            search_results
                            and search_results.get("metadatas")
                            and search_results["metadatas"][0]
                        ):
                            top_relevance = 1 - search_results["distances"][0][0]
                            keyword_signal = _keyword_overlap_score(
                                prompt, search_results["metadatas"][0][0]
                            )

                    duration = time.time() - start
                    is_relevant_match = (
                        top_relevance >= RELEVANCE_THRESHOLD
                        or keyword_signal >= KEYWORD_OVERLAP_THRESHOLD
                    )

                    if not is_relevant_match and not resolved_action["needs_clarification"]:
                        prev_recs = st.session_state.session_memory.get(
                            "order", {}
                        ).get("last_recommendations", [])
                        if len(prev_recs) == 1:
                            st.session_state.session_memory["order"][
                                "last_recommended_item"
                            ] = prev_recs[0]

                    if (
                        is_relevant_match
                        and not resolved_action["needs_clarification"]
                        and action_type not in CART_MUTATING_ACTIONS
                    ):
                        register_shown_items(search_results["metadatas"][0])

                    if resolved_action["needs_clarification"]:
                        response = resolved_action["clarification_message"]
                    elif resolved_action["cart_changed"]:
                        response = build_order_confirmation_message(action=action_type)
                        if action_type == "ADD_TO_CART":
                            pairing = _suggest_pairing(resolved_action)
                            if pairing:
                                response += pairing["text"]
                                register_shown_items([pairing["metadata"]])
                    elif should_search and not is_relevant_match:
                        context = (
                            "Specific nutrition parameters (like exact sugar grams or recipe ingredients) "
                            "are not listed in the menu dataset. Gently inform the user and suggest "
                            "naturally low-sugar or healthy categories like Salads & Healthy Options or unsweetened Beverages."
                        )
                        memory_context = build_memory_context()
                        recent_history = st.session_state.chat_history[-6:]
                        response = get_ai_response(
                            st.session_state.llm, prompt, recent_history, context, memory_context,
                        )
                    else:
                        context = _build_enhanced_context(prompt, search_results)
                        memory_context = build_memory_context()
                        recent_history = st.session_state.chat_history[-6:]
                        response = get_ai_response(
                            st.session_state.llm,
                            prompt,
                            recent_history,
                            context,
                            memory_context,
                        )

                    sync_shown_items_from_response(response)

                    st.session_state.interest_score = calculate_interest_score(
                        prompt,
                        st.session_state.interest_score,
                        resolved_action=resolved_action,
                        search_shown=is_relevant_match,
                    )

                    if search_results and is_relevant_match:
                        top_match = search_results["metadatas"][0][0].get("name", "N/A")
                        match_score = top_relevance
                    else:
                        top_match = "No menu match"
                        match_score = 0.0

            record_turn("assistant", response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.interest_history.append(st.session_state.interest_score)

            st.session_state.query_log.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "user_query": prompt,
                "top_match": top_match,
                "match_score": match_score,
                "duration_ms": round(duration * 1000, 2),
                "action": action_type,
            })

            st.rerun()

# ----------------- Complementary Pairing -----------------

def _suggest_pairing(resolved_action):
    """After a successful ADD_TO_CART, dynamically find a complementary item.

    Uses vector search to find the highest-ranking item from a different category
    that isn't already in the user's cart (e.g., pairing a pizza with a drink or side).
    Zero hardcoded category dictionaries needed.
    """
    try:
        order = st.session_state.session_memory["order"]
        cart = order.get("selected_items", [])
        if not cart:
            return None

        # Get the most recently added item
        last_added = cart[-1]
        product_id = str(last_added["product_id"])
        df = st.session_state.df
        match = df[df["product_id"].astype(str) == product_id]
        if match.empty:
            return None

        row = match.iloc[0]
        item_name = row["name"]
        item_category = (row.get("category") or "").lower()

        # Collect categories already present in cart
        cart_categories = {item_category}
        for ci in cart:
            cid = str(ci["product_id"])
            cm = df[df["product_id"].astype(str) == cid]
            if not cm.empty:
                cart_categories.add((cm.iloc[0].get("category") or "").lower())

        # Vector search using the item name to find flavor-compatible pairings
        search_results = _hybrid_search(item_name, top_k=10)
        if not search_results or not search_results.get("metadatas") or not search_results["metadatas"][0]:
            return None

        # Pick the top result whose category is NOT in the cart
        for meta in search_results["metadatas"][0]:
            result_category = (meta.get("category") or "").lower()
            result_id = str(meta.get("product_id"))

            # Skip items already in cart or from the same category
            if result_category in cart_categories or any(str(ci["product_id"]) == result_id for ci in cart):
                continue

            price = "Price N/A"
            try:
                price = f"${float(meta.get('price')):.2f}"
            except Exception:
                pass

            text = (
                f"\n\n---\n"
                f"**🍽️ Goes great with your order:**\n\n"
                f"• {meta.get('name', 'N/A')} — {price}\n"
                f"  Category: {meta.get('category', 'N/A')}\n\n"
                f"_Say \"add it\" to add this to your cart!_"
            )
            return {"text": text, "metadata": meta}

        return None
    except Exception:
        return None


# ----------------- Hybrid Search -----------------

def _hybrid_search(prompt, top_k=10, parsed_intent=None):
    preferred_categories = _preferred_categories_for_query(prompt, parsed_intent=parsed_intent)

    avoid_items = _items_to_avoid_for_query(prompt, parsed_intent=parsed_intent)
    log_embedding_generated(prompt, dim=384)
    query_embedding = st.session_state.embedder.encode(
        [prompt], show_progress_bar=False
    ).tolist()
    vector_results = st.session_state.collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    bm25_scores = st.session_state.bm25.get_scores(prompt.lower().split())
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda item: item[1], reverse=True)[:top_k]

    combined = {}
    vector_metadatas = vector_results.get("metadatas", [[]])[0]
    vector_distances = vector_results.get("distances", [[]])[0]
    for index, metadata in enumerate(vector_metadatas):
        if not metadata:
            continue
        score = 1 - vector_distances[index]
        if _matches_preferred_category(metadata, preferred_categories):
            score += 0.12
        if _should_avoid_item(metadata, avoid_items):
            score -= 0.35
        combined[metadata["product_id"]] = {"metadata": metadata, "score": score * 0.75}

    max_bm25 = max((score for _, score in bm25_ranked), default=0) or 1
    for index, score in bm25_ranked:
        metadata = st.session_state.records[index]
        product_id = metadata.get("product_id")
        bm25_score = score / max_bm25
        if _matches_preferred_category(metadata, preferred_categories):
            bm25_score += 0.15
        if _should_avoid_item(metadata, avoid_items):
            bm25_score -= 0.35
        if product_id in combined:
            combined[product_id]["score"] += bm25_score * 0.25
        else:
            combined[product_id] = {"metadata": metadata, "score": bm25_score * 0.25}

    ranked_items = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
    return {
        "metadatas": [[item["metadata"] for item in ranked_items]],
        "distances": [[max(0.0, min(1.0, 1 - item["score"])) for item in ranked_items]],
    }


def _preferred_categories_for_query(prompt, parsed_intent=None):
    if parsed_intent and parsed_intent.category_preference:
        return [parsed_intent.category_preference.lower()]
    return []


def _matches_preferred_category(metadata, preferred_categories):
    if not preferred_categories:
        return False
    return (metadata.get("category") or "").lower() in preferred_categories


def _items_to_avoid_for_query(prompt, parsed_intent=None):
    is_new = False
    if parsed_intent and getattr(parsed_intent, "is_request_for_new_options", False):
        is_new = True

    if not is_new:
        return set()

    memory = st.session_state.get("session_memory", {})
    order = memory.get("order", {})

    avoid = set()

    for product_id in order.get("recommended_items", []):
        avoid.add(str(product_id))

    last_item = order.get("last_recommended_item")
    if isinstance(last_item, dict):
        avoid.add(str(last_item.get("product_id")))
    elif last_item:
        avoid.add(str(last_item))

    for product_id in order.get("exclusions", []):
        avoid.add(str(product_id))

    return avoid


def _should_avoid_item(metadata, avoid_items):
    if not avoid_items:
        return False
    return str(metadata.get("product_id")) in avoid_items


# ----------------- Analytics & Admin -----------------

def render_analytics_sidebar(container):
    st.markdown("### 📈 Live Analytics")
    if not st.session_state.query_log:
        st.info("💡 Send a message to see real-time search metrics & latency logs.")
    else:
        latest = st.session_state.query_log[-1]
        with st.container(border=True):
            st.markdown("**Latest Query**")
            st.write(f"**User Query:** {latest['user_query']}")
            if latest["top_match"] == "No menu match":
                st.write("**Top Match:** No menu match (conversational turn)")
            else:
                st.write(f"**Top Match:** {latest['top_match']} ({latest['match_score']:.2%})")
            st.write(f"**Action:** {latest.get('action', 'N/A')}")
            st.write(f"**Time:** {latest['duration_ms']} ms")

    with st.container(border=True):
        st.markdown("**Intent Score**")
        st.metric("Current", f"{st.session_state.interest_score}%")
        st.line_chart(st.session_state.interest_history)


def render_admin_panel():
    with st.expander("⚙️ Admin Panel", expanded=False):
        analysis = st.session_state.get("df_analysis")
        if analysis:
            with st.expander("Data quality summary", expanded=False):
                st.write(f"Raw rows: {analysis['rows']}")
                st.write(f"Processed rows: {analysis['processed_rows']}")
                st.write(f"Removed rows: {analysis['removed_rows']}")
                if analysis["duplicate_product_ids"]:
                    st.write(f"Duplicate product IDs removed: {analysis['duplicate_product_ids']}")
                if analysis["missing_core_values"]:
                    st.write("Missing core values:")
                    st.json(analysis["missing_core_values"])

        df = st.data_editor(st.session_state.df, num_rows="dynamic", width="stretch")
        if st.button("Save to CSV"):
            try:
                df.to_csv(DATA_FILE_PATH, index=False)
                st.success("Changes saved!")
            except Exception as e:
                st.error(f"Save failed: {e}")