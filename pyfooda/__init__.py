from pathlib import Path

from .api import (
    find_ingredients,
    get_coverage,
    get_display_name,
    get_drv_df,
    get_ingredients_df,
    get_matches,
    get_nutrient_coverage,
    get_nutrients,
    get_sources,
    get_vocabulary,
    list_ingredients,
)

__version__ = (Path(__file__).parent / "VERSION").read_text().strip()

__all__ = [
    "__version__",
    "find_ingredients",
    "get_coverage",
    "get_display_name",
    "get_drv_df",
    "get_ingredients_df",
    "get_matches",
    "get_nutrient_coverage",
    "get_nutrients",
    "get_sources",
    "get_vocabulary",
    "list_ingredients",
]
