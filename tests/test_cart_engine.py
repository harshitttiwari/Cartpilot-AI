"""
Automated Test Suite for Cartpilot-AI
Tests: Cart Arithmetic, Deterministic State Router, Out-of-Bounds Guards, and Cross-Aisle Suggestions.
"""
import pytest
import streamlit as st
from session_memory import (
    initialize_session_memory,
    add_item_to_cart,
    remove_item_from_cart,
    clear_cart,
    get_cart_total,
    get_cart_items_count,
    get_cart_by_aisles,
    register_shown_items,
    update_state_from_user_message,
    check_ordinal_intent,
)
from bot_logic import ParsedUserIntent, ItemSpec


@pytest.fixture(autouse=True)
def setup_clean_session():
    """Initializes a fresh session state before each test."""
    initialize_session_memory()
    clear_cart()
    yield
    clear_cart()


def test_cart_addition_and_arithmetic():
    """Verifies exact subtotal arithmetic and item addition."""
    item1 = {"name": "FarmFresh Condensed Milk", "price": 5.82, "unit": "500g", "product_id": "GROC_001", "category": "Dairy & Eggs"}
    item2 = {"name": "PureHarvest Whole Wheat Bread", "price": 4.15, "unit": "400g", "product_id": "GROC_002", "category": "Bakery"}
    
    add_item_to_cart(item1, quantity=2)  # 2 x 5.82 = 11.64
    add_item_to_cart(item2, quantity=1)  # 1 x 4.15 = 4.15
    
    assert get_cart_items_count() == 3
    assert get_cart_total() == round(11.64 + 4.15, 2)


def test_aisle_categorization():
    """Verifies that items are automatically grouped into supermarket aisles."""
    item1 = {"name": "Organic Tomatoes", "price": 3.50, "unit": "1kg", "product_id": "GROC_003", "category": "Produce"}
    item2 = {"name": "Basmati Rice", "price": 6.00, "unit": "2kg", "product_id": "GROC_004", "category": "Pantry & Staples"}
    
    add_item_to_cart(item1, quantity=1)
    add_item_to_cart(item2, quantity=2)
    
    aisles = get_cart_by_aisles()
    assert "Produce" in aisles
    assert "Pantry & Staples" in aisles
    assert len(aisles["Produce"]) == 1
    assert len(aisles["Pantry & Staples"]) == 1


def test_dynamic_removal():
    """Verifies keyword, token, and conversational removal."""
    item = {"name": "MorningDew Ghee", "price": 9.16, "unit": "500g", "product_id": "GROC_005", "category": "Dairy & Eggs"}
    add_item_to_cart(item, quantity=1)
    assert get_cart_items_count() == 1
    
    removed = remove_item_from_cart("remove ghee from my cart")
    assert removed is not None
    assert removed["name"] == "MorningDew Ghee"
    assert get_cart_items_count() == 0
    assert get_cart_total() == 0.0


def test_out_of_bounds_ordinal_guard():
    """Verifies zero-hallucination guard when requesting positions beyond list bounds."""
    # Register only 1 shown item
    register_shown_items([
        {"name": "OrganicLife Diet Green Tea", "price": 12.53, "unit": "2kg", "product_id": "GROC_006"}
    ])
    
    # User asks for the 'third one'
    word, idx, item, total_shown = check_ordinal_intent("add the third one")
    assert word == "third"
    assert idx == 3
    assert item is None  # Must be None, not an unshown item!
    assert total_shown == 1
    
    # Verify conversational router returns bounds explanation
    res = update_state_from_user_message("add the third one")
    assert "There is no **third** item" in res["confirmation_override"]
    assert "Only **1 item** was found" in res["confirmation_override"]


def test_clear_cart():
    """Verifies cart wipe resets all quantities and total to $0.00."""
    item = {"name": "Everyday Basics Apples", "price": 3.28, "unit": "2 lbs", "product_id": "GROC_007", "category": "Produce"}
    add_item_to_cart(item, quantity=3)
    assert get_cart_items_count() == 3
    
    clear_cart()
    assert get_cart_items_count() == 0
    assert get_cart_total() == 0.0
