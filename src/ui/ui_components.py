# ui_components.py
import re
import streamlit as st
import time
from datetime import datetime
from src.core.bot_logic import get_ai_response, calculate_interest_score, parse_intent_with_llm
from src.ui.voice_component import render_voice_controller, render_tts_speaker
from src.database.database import DATA_FILE_PATH
from src.core.log import log_embedding_generated, log_vector_search, log_voice_command
from src.core.session_memory import (
    build_memory_context,
    build_order_confirmation_message,
    extract_allergen_restrictions,
    initialize_session_memory,
    record_turn,
    register_shown_items,
    sync_shown_items_from_response,
    update_state_from_user_message,
    add_item_to_cart,
    remove_item_from_cart,
    get_cart_by_aisles,
    get_cart_total,
    get_cart_items_count,
    clear_cart,
)

CART_MUTATING_ACTIONS = {"ADD_TO_CART", "REMOVE_ITEM", "CLEAR_CART", "VIEW_CART", "CHECKOUT"}
RELEVANCE_THRESHOLD = 0.28


# ----------------- Context Building Helpers -----------------

def _build_enhanced_context(user_query, search_results, parsed_intent=None):
    """Build readable grocery catalog context for the LLM from database search results."""
    if not search_results or not search_results.get("metadatas") or not search_results["metadatas"][0]:
        return "No matching products found in the grocery catalog."

    dietary_restrictions = extract_allergen_restrictions(user_query)
    category_pref = parsed_intent.category_preference if parsed_intent else None
    
    items = search_results["metadatas"][0]
    if not items:
        return "No suitable grocery items found."

    categorized = {}
    for item in items:
        cat = item.get("category", "General Grocery")
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(item)

    context = "Available Supermarket Items Matching Request:\n\n"
    for category, item_list in categorized.items():
        context += f"**{category.upper()}:**\n"
        for item in item_list:
            price = f"${float(item.get('price', 0.0)):.2f}"
            unit = item.get("unit", "")
            unit_display = f" ({unit})" if unit else ""
            diet = item.get("dietary_tags", "standard") or "standard"

            context += f"• **{item.get('name', 'N/A')}**{unit_display} – {price}\n"
            context += f"  • Category: {item.get('category', 'N/A')}\n"
            context += f"  • Dietary: [{diet}]\n"
            if item.get("description"):
                context += f"  • Info: {item.get('description')}\n"
            context += "\n"

    return context


# ----------------- UI Rendering -----------------

def render_chat_interface(container):
    with container:
        initialize_session_memory()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if not st.session_state.chat_history:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "👋 **Hello!** What groceries would you like to add to your list today? *(e.g., \"Add 2 milk and bread\")*",
            })

        if "interest_score" not in st.session_state:
            st.session_state.interest_score = 50

        if "interest_history" not in st.session_state:
            st.session_state.interest_history = [50]

        if "query_log" not in st.session_state:
            st.session_state.query_log = []

        # 1. Voice Controller Bar (Mic Button + Language Selector + TTS Switch)
        render_voice_controller()

        # 2. Scrollable Chat History Container (Fills viewport cleanly)
        chat_history_container = st.container(height=450)
        with chat_history_container:
            for msg in st.session_state.chat_history[-50:]:
                avatar_icon = "🤖" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar_icon):
                    role_marker = f"<span class='role-marker-{msg['role']}' style='display:none'></span>"
                    st.markdown(role_marker + msg["content"], unsafe_allow_html=True)

        # 3. Chat Input Box (Voice injects directly here)
        if prompt := st.chat_input("Speak or type grocery commands..."):
            lang = st.session_state.get("selected_voice_lang", "en-IN")
            log_voice_command(lang, prompt)
            record_turn("user", prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Processing command..."):
                start = time.time()
                current_cart = st.session_state.session_memory.get("order", {}).get("selected_items", [])
                parsed_intent = parse_intent_with_llm(st.session_state.llm, prompt, current_cart_items=current_cart)
                action_type = parsed_intent.action
                
                # Check for state mutation (View Cart, Clear Cart, Remove, Ordinal/Pronoun Add)
                resolved_action = update_state_from_user_message(prompt, parsed_intent=parsed_intent)
                
                top_match = "N/A"
                match_score = 1.0
                search_results = None

                # Handle Multi-Item Voice Addition & Compound Add/Remove
                if action_type == "ADD_TO_CART" and not resolved_action.get("cart_changed"):
                    # Process any simultaneous removals first
                    removed_summaries = []
                    if parsed_intent and parsed_intent.remove_items:
                        for rem_target in parsed_intent.remove_items:
                            rem_res = remove_item_from_cart(rem_target)
                            if rem_res:
                                removed_summaries.append(rem_res["name"])

                    items_to_add = parsed_intent.items
                    if not items_to_add and not removed_summaries:
                        # Single query fallback
                        items_to_add = [{"item_name": parsed_intent.cleaned_search_query or prompt, "quantity": 1}]

                    added_summaries = []
                    missing_items = []
                    all_suggestions = []

                    for it in items_to_add:
                        item_name = it.item_name if hasattr(it, "item_name") else it.get("item_name", "")
                        qty = it.quantity if hasattr(it, "quantity") else it.get("quantity", 1)
                        if not item_name or not item_name.strip():
                            continue
                        
                        s_results = _hybrid_search(item_name, top_k=3, parsed_intent=parsed_intent)
                        if s_results and s_results.get("metadatas") and s_results["metadatas"][0]:
                            best_match = s_results["metadatas"][0][0]
                            dist = s_results["distances"][0][0] if s_results.get("distances") else 1.0
                            
                            # Check token / keyword presence in product name or category
                            q_words = set(re.findall(r"\b\w{3,}\b", item_name.lower()))
                            prod_words = set(re.findall(r"\b\w{3,}\b", f"{best_match.get('name', '')} {best_match.get('category', '')}".lower()))
                            # Also check singular/plural prefix matching (e.g. apple in apples)
                            has_direct_match = any(
                                any(qw in pw or pw in qw for pw in prod_words)
                                for qw in q_words
                            ) if q_words else False

                            # Accept if direct keyword match exists OR semantic distance is confident (dist <= 0.74)
                            if has_direct_match or dist <= 0.74:
                                res = add_item_to_cart(best_match, quantity=qty)
                                added_summaries.append(f"**{qty}x {best_match['name']}** ({best_match.get('unit', '')})")
                                all_suggestions.extend(res.get("smart_suggestions", []))
                                register_shown_items(s_results["metadatas"][0])
                            else:
                                missing_items.append(item_name)
                        else:
                            missing_items.append(item_name)

                    total = get_cart_total()
                    total_count = get_cart_items_count()
                    parts = []

                    if removed_summaries:
                        rem_str = ", ".join(f"**{r}**" for r in removed_summaries)
                        parts.append(f"🗑️ Removed {rem_str} from your shopping list.")

                    if added_summaries:
                        items_str = ", ".join(added_summaries)
                        parts.append(f"🛒 Added {items_str} to your shopping list! (Subtotal: **\\${total:.2f}**)")
                        top_match = added_summaries[0]
                    
                    if missing_items:
                        miss_str = ", ".join(f"**{m}**" for m in missing_items)
                        parts.append(f"ℹ️ Currently, {miss_str} is not available in our store catalog.")

                    if parts:
                        response = "\n\n".join(parts)
                        
                        # If user also said "and checkout" / "place order", finalize immediately!
                        if any(w in prompt.lower() for w in ["checkout", "check out", "place order", "order now", "order kar do", "final order"]):
                            response += f"\n\n🎉 **Order Placed Successfully!**\n\nYour shopping list of **{total_count} items** totaling **\\${total:.2f}** has been confirmed for delivery."
                        elif all_suggestions:
                            # Unique suggestions
                            seen_sug = set()
                            unique_sug = []
                            for s in all_suggestions:
                                if s["product_id"] not in seen_sug:
                                    seen_sug.add(s["product_id"])
                                    unique_sug.append(s)
                            
                            # Update active smart suggestion memory so 'add both' matches screen
                            st.session_state.session_memory["order"]["last_suggested_items"] = unique_sug[:2]
                            
                            sug_items = [f"**{s['name']}** (\\${s['price']:.2f})" for s in unique_sug[:2]]
                            sug_text = " or ".join(sug_items)
                            response += f"\n\n💡 **Smart Suggestion**: Shoppers frequently also add {sug_text}. Say *\"add both\"* or *\"add it\"* to include them!"
                    else:
                        response = f"I searched the catalog for '{prompt}' but couldn't locate matching items. Try asking for basic essentials like milk, bread, eggs, apples, or snacks."

                elif resolved_action.get("confirmation_override"):
                    response = resolved_action["confirmation_override"]
                    top_match = action_type

                elif action_type == "VIEW_MENU" or action_type == "GENERAL":
                    search_prompt = parsed_intent.cleaned_search_query or prompt
                    search_results = _hybrid_search(search_prompt, top_k=4, parsed_intent=parsed_intent)
                    top_relevance = 0.0
                    if search_results and search_results.get("metadatas") and search_results["metadatas"][0]:
                        top_relevance = 1 - search_results["distances"][0][0]
                        register_shown_items(search_results["metadatas"][0])
                        top_match = search_results["metadatas"][0][0].get("name", "Item")
                        match_score = top_relevance

                    context = _build_enhanced_context(prompt, search_results, parsed_intent=parsed_intent)
                    memory_context = build_memory_context()
                    recent_history = st.session_state.chat_history[-6:]
                    response = get_ai_response(
                        st.session_state.llm,
                        prompt,
                        recent_history,
                        context,
                        memory_context,
                    )
                    # Sync mentioned items so "add the second one" matches the displayed menu!
                    sync_shown_items_from_response(response)
                else:
                    response = "I've updated your shopping list."

                duration = time.time() - start

                st.session_state.interest_score = calculate_interest_score(
                    prompt,
                    st.session_state.interest_score,
                    resolved_action=resolved_action,
                    search_shown=bool(search_results),
                )

            # Record turn in Hybrid Buffer Window (K = 6)
            record_turn("assistant", response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.interest_history.append(st.session_state.interest_score)
            
            # Read aloud via TTS if enabled
            render_tts_speaker(response)

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

def _hybrid_search(prompt, top_k=10, parsed_intent=None):
    category_pref = (
        parsed_intent.category_preference.lower()
        if (parsed_intent and parsed_intent.category_preference)
        else None
    )

    log_embedding_generated(prompt, dim=384)
    query_embedding = st.session_state.embedder.encode(
        [prompt], show_progress_bar=False
    ).tolist()
    vector_results = st.session_state.collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    # Expand singular/plural search tokens for BM25 (e.g. 'egg' -> 'eggs', 'apples' -> 'apple')
    search_tokens = []
    for tok in prompt.lower().split():
        clean_tok = re.sub(r"[^\w]", "", tok)
        if clean_tok:
            search_tokens.append(clean_tok)
            if clean_tok.endswith("s") and len(clean_tok) > 3:
                search_tokens.append(clean_tok[:-1])
            else:
                search_tokens.append(clean_tok + "s")

    bm25_scores = st.session_state.bm25.get_scores(search_tokens if search_tokens else prompt.lower().split())
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda item: item[1], reverse=True)[:top_k]

    combined = {}
    vector_metadatas = vector_results.get("metadatas", [[]])[0]
    vector_distances = vector_results.get("distances", [[]])[0]
    for index, metadata in enumerate(vector_metadatas):
        if not metadata:
            continue
        score = 1 - vector_distances[index]
        if category_pref and category_pref in (metadata.get("category") or "").lower():
            score += 0.15
        combined[metadata["product_id"]] = {"metadata": metadata, "score": score * 0.70}

    max_bm25 = max((score for _, score in bm25_ranked), default=0) or 1
    for index, score in bm25_ranked:
        metadata = st.session_state.records[index]
        product_id = metadata.get("product_id")
        bm25_score = score / max_bm25
        if category_pref and category_pref in (metadata.get("category") or "").lower():
            bm25_score += 0.15
        if product_id in combined:
            combined[product_id]["score"] += bm25_score * 0.30
        else:
            combined[product_id] = {"metadata": metadata, "score": bm25_score * 0.30}

    ranked_items = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
    
    # 3. Dynamic Price Range Filtering ("under $5", "below 10 dollars")
    if parsed_intent and parsed_intent.max_price is not None:
        ranked_items = [it for it in ranked_items if float(it["metadata"].get("price", 0.0)) <= parsed_intent.max_price]
    if parsed_intent and parsed_intent.min_price is not None:
        ranked_items = [it for it in ranked_items if float(it["metadata"].get("price", 0.0)) >= parsed_intent.min_price]

    top_score = ranked_items[0]["score"] if ranked_items else 0.0
    log_vector_search(prompt, len(ranked_items), top_score)

    return {
        "metadatas": [[item["metadata"] for item in ranked_items]],
        "distances": [[max(0.0, min(1.0, 1 - item["score"])) for item in ranked_items]],
    }


# ----------------- Analytics & Live Shopping Cart Sidebar -----------------

def render_analytics_sidebar(container):
    initialize_session_memory()
    total = get_cart_total()
    total_count = get_cart_items_count()
    aisles = get_cart_by_aisles()

    # 1. Live Query & Engagement (Placed at Upper Position)
    st.markdown("#### 📈 Live Query & Engagement")
    with st.container(border=True):
        if st.session_state.query_log:
            latest = st.session_state.query_log[-1]
            st.write(f"**Query:** {latest['user_query']}")
            c_act, c_lat = st.columns(2)
            with c_act:
                st.caption(f"Action: `{latest.get('action', 'N/A')}`")
            with c_lat:
                st.caption(f"Latency: `{latest['duration_ms']} ms`")
            st.divider()

        score = st.session_state.interest_score
        st.markdown(f"**Engagement Level:** `{score}%`")
        st.progress(max(0.0, min(1.0, score / 100.0)))
        if len(st.session_state.interest_history) > 1:
            st.line_chart(st.session_state.interest_history, height=65)

    st.markdown("---")

    # 2. Live Shopping Cart (Placed Below Engagement)
    st.markdown("#### 🛒 Live Shopping Cart")
    if total_count == 0:
        st.info("Your shopping list is empty. Speak or type to add items!")
    else:
        with st.container(border=True):
            st.markdown(f"**Items:** `{total_count}` | **Total:** `\\${total:.2f}`")
            
            for aisle, items in aisles.items():
                st.caption(f"📍 **{aisle}**")
                for it in items:
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.write(f"• **{it['quantity']}x {it['name']}** ({it['unit']})")
                    with c2:
                        st.write(f"\\${it['subtotal']:.2f}")

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button("🧹 Clear", width="stretch"):
                    clear_cart()
                    st.rerun()
            with bcol2:
                if st.button("🎉 Checkout", type="primary", width="stretch"):
                    st.balloons()
                    st.success(f"Order Placed! Total: \\${total:.2f}")


def render_admin_panel():
    with st.expander("⚙️ Catalog Manager & Admin", expanded=False):
        analysis = st.session_state.get("df_analysis")
        if analysis:
            st.write(f"Total Products: `{analysis['processed_rows']}` across 5 Aisles")
            if analysis.get("categories"):
                st.json(analysis["categories"])

        df = st.data_editor(st.session_state.df, num_rows="dynamic", width="stretch")
        if st.button("💾 Save Catalog to CSV"):
            try:
                df.to_csv(DATA_FILE_PATH, index=False)
                st.success("Catalog changes saved successfully!")
            except Exception as e:
                st.error(f"Save failed: {e}")