
from hrp.ingredient_mapping import (
    ndc_product_key,
    rxnorm_package_product_key,
)


def test_product_ndc_normalization():
    assert ndc_product_key("1234-5678") == "012345678"
    assert ndc_product_key("12345-678") == "123450678"
    assert ndc_product_key("invalid") is None


def test_rxnorm_package_ndc():
    assert rxnorm_package_product_key("01234567801") == "012345678"
    assert rxnorm_package_product_key("123456789") is None