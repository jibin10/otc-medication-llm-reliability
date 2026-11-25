from hrp.data_cleaning import (
    classify_repeated_ndc,
    ingredient_signature,
    normalize_name,
)


def test_name_normalization():
    assert normalize_name("  COLD   PM  ") == "cold pm"
    assert normalize_name("Cold PM") != normalize_name("Cold Day")
    assert normalize_name("Extra Strength") != normalize_name("Regular")


def test_ingredient_signature():
    ingredients = [{"name": "Acetaminophen", "strength": "500 MG"}]

    assert ingredient_signature(ingredients) == (("acetaminophen", "500 mg"),)
    assert ingredient_signature([]) is None


def test_repeated_ndc_classification():
    rows = [
        {
            "fingerprint": "a",
            "ingredients": (("acetaminophen", "500 mg"),),
            "dosage": "tablet",
            "route": ("oral",),
        },
        {
            "fingerprint": "b",
            "ingredients": (("ibuprofen", "200 mg"),),
            "dosage": "tablet",
            "route": ("oral",),
        },
    ]

    assert classify_repeated_ndc(rows) == ("review_formulation_difference")


from hrp.data_cleaning import quantity_key


def test_quantity_conversion_requires_matching_dimensions_and_explicit_percent_basis():
    assert quantity_key("5 mg/mL") == quantity_key("25 mg/5 mL")
    assert quantity_key("50 mg/g") == quantity_key("5% w/w")
    assert quantity_key("5 mg/mL") == quantity_key("0.5% w/v")
    assert quantity_key("5 mg/mL") != quantity_key("5 mg/g")
    assert quantity_key("0.5%") != quantity_key("5 mg/mL")
    assert quantity_key("0.25%") != quantity_key("1.15 mg/g")
    assert quantity_key("5 mg/0 mL") is None
    assert quantity_key("5 " + chr(181) + "g") == quantity_key("5 mcg")
    assert quantity_key("5 " + chr(956) + "g") == quantity_key("5 mcg")
