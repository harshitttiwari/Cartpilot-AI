# ui_components.py
import streamlit as st
import time
from datetime import datetime
from bot_logic import get_ai_response, calculate_interest_score
from database import DATA_FILE_PATH
from session_memory import (
    build_memory_context,
    build_order_confirmation_message,
    initialize_session_memory,
    record_turn,
    register_shown_items,
    update_state_from_user_message,
)

CART_MUTATING_ACTIONS = {"ADD_TO_CART", "REMOVE_ITEM", "CHECKOUT"}

# Below this, a "top match" is noise, not a real recommendation — don't
# register it as last_recommendations and don't report it as if it were
# meaningful in the analytics sidebar. Tune this against your own corpus if
# search scores run differently for you.
RELEVANCE_THRESHOLD = 0.30

# Only these actions represent the user actually asking about the menu.
# Chit-chat ("hello", "okayy good", "thanks") gets no search at all.
MENU_RELEVANT_ACTIONS = {"VIEW_MENU", "ASK_ALLERGEN", "COMPARE_ITEMS"}


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

CART_MUTATING_ACTIONS = {"ADD_TO_CART", "REMOVE_ITEM", "CHECKOUT"}

# Below this, a match is treated as noise rather than a real recommendation.
RELEVANCE_THRESHOLD = 0.30
KEYWORD_OVERLAP_THRESHOLD = 0.34


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
        st.header("Conversational Agent")

        initialize_session_memory()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if not st.session_state.chat_history:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Welcome to FoodieBot! Ask me about menu items, recommendations, allergens, or drinks.",
            })

        if "interest_score" not in st.session_state:
            st.session_state.interest_score = 50

        if "interest_history" not in st.session_state:
            st.session_state.interest_history = [50]

        if "query_log" not in st.session_state:
            st.session_state.query_log = []

        chat_history_container = st.container(height=520, border=True)

        with chat_history_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask me about the menu..."):

            record_turn("user", prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                start = time.time()

                # STEP 1: Intent -> Reference resolution -> Action.
                # Resolves against last turn's last_recommendations, mutates
                # the cart in code if confident enough. Must run BEFORE any
                # new search so a fresh search never clobbers the list being
                # resolved against.
                resolved_action = update_state_from_user_message(prompt)
                action_type = resolved_action["action"]

                # STEP 2: Run hybrid search for anything that isn't a cart
                # mutation (already resolved above). This includes GENERAL
                # turns like "i am having fever" — real recommendations can
                # come out of those too, so they must not be skipped.
                search_results = None
                top_relevance = 0.0
                keyword_signal = 0.0

                should_search = action_type not in CART_MUTATING_ACTIONS

                if should_search:
                    search_results = _hybrid_search(prompt)
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

                # Combine the (approximate, occasionally noisy) vector score
                # with the (deterministic) keyword-overlap score. Either one
                # clearing its bar is enough to call the match relevant.
                is_relevant_match = (
                    top_relevance >= RELEVANCE_THRESHOLD
                    or keyword_signal >= KEYWORD_OVERLAP_THRESHOLD
                )

                # Register recommendations whenever real, relevant items were
                # actually shown — regardless of what classify_action guessed
                # the intent was. A "GENERAL" message can still surface real
                # recommendations, and those must be remembered the same as
                # an explicit "show me options" turn, or ordinal/pronoun
                # references to them ("order the first one") will fail later.
                if (
                    is_relevant_match
                    and not resolved_action["needs_clarification"]
                    and action_type not in CART_MUTATING_ACTIONS
                ):
                    register_shown_items(search_results["metadatas"][0])

                # STEP 3: Decide the response.
                if resolved_action["needs_clarification"]:
                    response = resolved_action["clarification_message"]

                elif resolved_action["cart_changed"]:
                    # Deterministic — LLM never narrates cart state.
                    response = build_order_confirmation_message(action=action_type)

                elif should_search and not is_relevant_match:
                    # Nothing in the menu was actually relevant — let the LLM
                    # handle it conversationally rather than treating a weak
                    # match as if it meant something.
                    context = "No relevant menu items found for this message."
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

            record_turn("assistant", response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # STEP 4: Combined interest score — phrase-based tone signal +
            # action-based outcome signal (resolved_action carries whether a
            # cart change actually happened).
            st.session_state.interest_score = calculate_interest_score(
                prompt,
                st.session_state.interest_score,
                resolved_action=resolved_action,
            )
            st.session_state.interest_history.append(st.session_state.interest_score)

            if search_results and is_relevant_match:
                top_match = search_results["metadatas"][0][0].get("name", "N/A")
                match_score = top_relevance
            else:
                top_match = "No menu match"
                match_score = 0.0

            st.session_state.query_log.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "user_query": prompt,
                "top_match": top_match,
                "match_score": match_score,
                "duration_ms": round(duration * 1000, 2),
                "action": action_type,
            })

            st.rerun()

# ----------------- Hybrid Search -----------------

def _hybrid_search(prompt, top_k=10):
    preferred_categories = _preferred_categories_for_query(prompt)
    avoid_items = _items_to_avoid_for_query(prompt)
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

    ranked_items = sorted(
        combined.values(),
        key=lambda item: (
            not _matches_preferred_category(item["metadata"], preferred_categories),
            _should_avoid_item(item["metadata"], avoid_items),
            -item["score"],
        ),
    )[:top_k]

    return {
        "metadatas": [[item["metadata"] for item in ranked_items]],
        "distances": [[1 - item["score"] for item in ranked_items]],
    }


def _preferred_categories_for_query(prompt):
    query = prompt.lower()
    category_keywords = {
        "beverages":              ["drink", "beverage", "thirsty", "cooling", "refreshing", "refresher", "lemonade", "shake", "smoothie"],
        "breakfast items":        ["breakfast", "morning", "muffin", "toast", "waffle"],
        "desserts":               ["dessert", "sweet", "cake", "cookie", "sundae", "mousse", "brownie"],
        "pizza":                  ["pizza"],
        "burgers":                ["burger"],
        "salads & healthy options": ["salad", "healthy", "greens"],
        "tacos & wraps":          ["taco", "wrap"],
        "sides & appetizers":     ["side", "appetizer", "fries", "rings", "snack"],
    }
    return [
        category
        for category, keywords in category_keywords.items()
        if any(keyword in query for keyword in keywords)
    ]


def _matches_preferred_category(metadata, preferred_categories):
    if not preferred_categories:
        return False
    return (metadata.get("category") or "").lower() in preferred_categories


def _items_to_avoid_for_query(prompt):
    query = prompt.lower()

    new_option_words = [
        "another", "different", "else", "more options",
        "other option", "not this", "change it",
        "show more", "next",
    ]

    if not any(word in query for word in new_option_words):
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
        st.info("No queries yet.")
        return
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

        df = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True)
        if st.button("Save to CSV"):
            try:
                df.to_csv(DATA_FILE_PATH, index=False)
                st.success("Changes saved!")
            except Exception as e:
                st.error(f"Save failed: {e}")