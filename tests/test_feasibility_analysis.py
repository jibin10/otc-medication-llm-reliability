from hrp.feasibility_analysis import distinct_ingredient_pair


def test_different_ingredient_pair():
    products = [
        {"product_ndc": "100-1", "in_ids": ("10",)},
        {"product_ndc": "100-2", "in_ids": ("20",)},
        {"product_ndc": "100-3", "in_ids": ("10",)},
    ]

    pair = distinct_ingredient_pair(products)

    assert pair is not None
    assert pair[0]["in_ids"] != pair[1]["in_ids"]


def test_same_ingredients_not_selected():
    products = [
        {"product_ndc": "100-1", "in_ids": ("10",)},
        {"product_ndc": "100-2", "in_ids": ("10",)},
    ]

    assert distinct_ingredient_pair(products) is None