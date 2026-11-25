"""Basic check that the project package is installed in the local environment."""

from importlib import import_module


def test_research_modules_can_be_imported() -> None:
    modules = (
        "hrp.data_loading",
        "hrp.data_cleaning",
        "hrp.ingredient_mapping",
        "hrp.feasibility_analysis",
    )

    for module_name in modules:
        assert import_module(module_name) is not None
