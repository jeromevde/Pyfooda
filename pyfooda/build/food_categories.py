"""Infer ingredient food families and filter incompatible USDA matches."""

from __future__ import annotations

import re

# Epicure id suffixes → expected USDA food family.
_SUFFIX_FAMILIES: tuple[tuple[str, str], ...] = (
    ("_oil", "oil_fat"),
    ("_flour", "grain_milled"),
    ("_starch", "grain_milled"),
    ("_meal", "grain_milled"),
    ("_cereal", "cereal"),
    ("_flakes", "cereal"),
    ("_flake", "cereal"),
    ("_crisp", "cereal"),
    ("_milk", "dairy"),
    ("_cream", "dairy"),
    ("_butter", "dairy"),
    ("_cheese", "cheese"),
    ("_yogurt", "dairy"),
    ("_juice", "juice"),
    ("_wine", "alcohol"),
    ("_beer", "alcohol"),
    ("_liqueur", "alcohol"),
    ("_vinegar", "condiment"),
    ("_sauce", "condiment"),
    ("_paste", "condiment"),
    ("_syrup", "syrup"),
    ("_jam", "jam"),
    ("_broth", "broth"),
    ("_stock", "broth"),
    ("_extract", "extract"),
    ("_essence", "extract"),
    ("_powder", "powder"),
    ("_chips", "snack"),
    ("_crisps", "snack"),
)

_PRODUCE_SUFFIXES = (
    "_squash",
    "_pepper",
    "_bean",
    "_pea",
    "_lentil",
    "_berry",
    "_apple",
    "_melon",
    "_mushroom",
    "_green",
    "_lettuce",
    "_cabbage",
    "_kale",
    "_spinach",
    "_tomato",
    "_onion",
    "_garlic",
    "_potato",
    "_carrot",
    "_cucumber",
    "_zucchini",
    "_eggplant",
    "_radish",
    "_turnip",
    "_beet",
    "_broccoli",
    "_cauliflower",
    "_asparagus",
    "_artichoke",
    "_leek",
    "_shallot",
    "_chili",
    "_chile",
)

_ALCOHOL_TOKENS = (
    "absinthe",
    "advocaat",
    "aguardiente",
    "amaretto",
    "aperol",
    "aquavit",
    "arak",
    "baijiu",
    "beer",
    "bourbon",
    "brandy",
    "cachaça",
    "cachaca",
    "champagne",
    "cognac",
    "gin",
    "grappa",
    "kirsch",
    "liqueur",
    "mezcal",
    "ouzo",
    "pastis",
    "port",
    "rum",
    "sake",
    "schnapps",
    "sherry",
    "soju",
    "tequila",
    "vermouth",
    "vodka",
    "whiskey",
    "whisky",
    "wine",
)

_MEAT_TOKENS = (
    "abalone",
    "alligator",
    "anchovy",
    "bacon",
    "beef",
    "bison",
    "boar",
    "calamari",
    "carp",
    "catfish",
    "chicken",
    "chorizo",
    "clam",
    "cod",
    "crab",
    "crayfish",
    "duck",
    "eel",
    "elk",
    "fish",
    "goose",
    "haddock",
    "halibut",
    "ham",
    "herring",
    "lamb",
    "lobster",
    "mackerel",
    "mussel",
    "octopus",
    "oyster",
    "pork",
    "prawn",
    "rabbit",
    "salami",
    "salmon",
    "sardine",
    "sausage",
    "scallop",
    "shrimp",
    "squid",
    "sturgeon",
    "tilapia",
    "trout",
    "tuna",
    "turkey",
    "veal",
    "venison",
)

_GRAIN_IDS = frozenset(
    {
        "amaranth",
        "barley",
        "buckwheat",
        "bulgur",
        "corn",
        "couscous",
        "farro",
        "freekeh",
        "kamut",
        "millet",
        "oat",
        "oats",
        "quinoa",
        "rice",
        "rye",
        "semolina",
        "sorghum",
        "spelt",
        "teff",
        "triticale",
        "wheat",
        "wild_rice",
    }
)

_SPICE_IDS = frozenset(
    {
        "ajwain",
        "allspice",
        "anise",
        "annatto",
        "asafoetida",
        "baharat",
        "basil",
        "bay",
        "cacao",
        "caraway",
        "cardamom",
        "cassia",
        "cayenne",
        "chervil",
        "chive",
        "cinnamon",
        "clove",
        "coriander",
        "cumin",
        "curry",
        "dill",
        "epazote",
        "fennel",
        "fenugreek",
        "galangal",
        "ginger",
        "horseradish",
        "juniper",
        "lavender",
        "lemongrass",
        "licorice",
        "mace",
        "marjoram",
        "mint",
        "mustard",
        "nutmeg",
        "oregano",
        "paprika",
        "parsley",
        "pepper",
        "pimento",
        "poppy",
        "rosemary",
        "saffron",
        "sage",
        "savory",
        "sesame",
        "spearmint",
        "star_anise",
        "sumac",
        "tarragon",
        "thyme",
        "turmeric",
        "vanilla",
        "wasabi",
    }
)

# Branded/processed USDA categories incompatible with whole produce.
_WHOLE_FOOD_BLOCKED_CATEGORY = re.compile(
    r"cereal|cand(y|ies)|chocolate|cookie|biscuit|cake|cupcake|pie|ice cream|"
    r"frozen yogurt|yogurt|cheese|\bmilk\b|soda|soft drink|powdered drink|"
    r"snack bar|granola bar|crackers|pretzels|chips|popcorn|"
    r"pasta|pizza|bread|bun|bagel|muffin|waffle|pancake|"
    r"sausage|hotdog|brat|pepperoni|salami|cold cut|bacon|"
    r"jam|jelly|spread|syrup|molasses|"
    r"gelatin|dessert topping|baking decoration|"
    r"fruit drink|nectar|vegetable juice|juice drink|100% juice|"
    r"beverage|soft drink|\bsoda\b|"
    r"soup|broth|gravy|sauce|ketchup|mustard|mayonnaise|dressing|dip|salsa|"
    r"seasoning mix|\bsalt\b|marinade|"
    r"frozen dinner|frozen entree|prepared|meal kit|"
    r"baby food|wholesome snack",
    re.I,
)

# USDA categories/names that indicate raw or minimally processed whole foods.
_WHOLE_FOOD_ALLOWED_CATEGORY = re.compile(
    r"fruits and fruit juices|vegetables and vegetable|legumes and legume|"
    r"nut and seed|finfish and shellfish|poultry|sausage.*lunch meat|"
    r"beef|pork|lamb|veal|game|"
    r"^strawberries$|^apples$|^bananas$|^berries$|^grapes$|^melons$|^citrus$|^peaches$",
    re.I,
)

_NAME_PROCESSED = re.compile(
    r"^(cereal|milk|yogurt|syrup|candy|chocolate|cookie|cake|cupcake|pie|"
    r"danish|bar|drink|soda|juice|spread|sauce|dressing|snack|crackers|"
    r"ice cream|gelatin|pudding|granola|oatmeal)\b",
    re.I,
)

_FAMILY_BLOCKED_USDA: dict[str, re.Pattern[str]] = {
    "whole_food": _WHOLE_FOOD_BLOCKED_CATEGORY,
    "grain": re.compile(
        r"cereal|cand(y|ies)|chocolate|cookie|cake|ice cream|yogurt|"
        r"sausage|pepperoni|salami|finfish|shellfish|poultry products|"
        r"baby food(?!: cereals)",
        re.I,
    ),
    "meat_seafood": re.compile(
        r"cereal|cand(y|ies)|chocolate|cookie|cake|ice cream|yogurt|"
        r"fruit & vegetable juice|soda|pasta(?! sauce)|bread(?! crumb)",
        re.I,
    ),
    "dairy": re.compile(
        r"cereal|cand(y|ies)|chocolate|cookie|cake|sausage|pepperoni|"
        r"finfish|shellfish|poultry products(?!)",
        re.I,
    ),
    "cheese": re.compile(
        r"cereal|cand(y|ies)|chocolate|cookie|cake|yogurt|milk\b|"
        r"finfish|shellfish|poultry products",
        re.I,
    ),
    "cereal": re.compile(
        r"fruits and fruit juices|vegetables and vegetable|"
        r"finfish and shellfish|poultry products|beef products|pork products",
        re.I,
    ),
}


def infer_ingredient_family(ingredient_id: str, display_name: str) -> str:
    """Map an Epicure ingredient to a coarse food family for USDA filtering."""
    iid = ingredient_id.lower().strip()
    _name = display_name.lower().strip()

    for suffix, family in _SUFFIX_FAMILIES:
        if iid.endswith(suffix):
            return family

    if any(token in iid for token in _ALCOHOL_TOKENS):
        return "alcohol"

    if iid in _SPICE_IDS:
        return "spice"

    if iid in _GRAIN_IDS:
        return "grain"

    _PRODUCT_FAMILY = {
        "cereal": "cereal",
        "cookie": "generic",
        "caramel": "generic",
        "fudge": "generic",
    }
    if iid in _PRODUCT_FAMILY:
        return _PRODUCT_FAMILY[iid]

    if iid in ("egg", "eggs") or iid.startswith("egg_"):
        return "egg"

    if iid in ("milk", "butter", "cream", "yogurt") or iid.endswith("_milk"):
        return "dairy"

    if "cheese" in iid:
        return "cheese"

    if any(token in iid for token in _MEAT_TOKENS):
        return "meat_seafood"

    if any(iid.endswith(suffix) for suffix in _PRODUCE_SUFFIXES):
        return "whole_food"

    if "_" not in iid:
        return "whole_food"

    return "generic"


def usda_food_category(row) -> str:
    cat = row.get("food_category") if hasattr(row, "get") else None
    if cat is None or (isinstance(cat, float) and cat != cat):
        return ""
    return str(cat).strip()


def usda_food_name(row) -> str:
    name = row.get("foodName") if hasattr(row, "get") else ""
    return str(name or "").strip()


def _whole_food_allowed(row) -> bool:
    category = usda_food_category(row)
    name = usda_food_name(row)
    data_type = str(row.get("data_type") or "")

    if _WHOLE_FOOD_ALLOWED_CATEGORY.search(category):
        return True

    if re.search(
        r"^(strawberries|apples|bananas|berries|grapes|melons|citrus|peaches|pears|"
        r"plums|oranges|tomatoes|potatoes|onions|carrots|broccoli|spinach|lettuce|"
        r"mushrooms|legumes|beans|peas|nuts|almonds|walnuts)$",
        category,
        re.I,
    ):
        return True

    if _WHOLE_FOOD_BLOCKED_CATEGORY.search(category):
        return False
    if _NAME_PROCESSED.search(name):
        return False

    if data_type in {"survey_fndds_food", "foundation_food", "sr_legacy_food"}:
        if re.search(r",\s*(raw|fresh|frozen|canned|dried|unsweetened)\b", name, re.I):
            return True
        if re.search(
            r"^(strawberries|apples|bananas|blueberries|raspberries|blackberries|"
            r"cherries|grapes|peaches|pears|plums|oranges|lemons|limes|"
            r"tomatoes|potatoes|onions|carrots|broccoli|spinach|lettuce|"
            r"mushrooms|almonds|walnuts|pecans|cashews|peanuts)\b",
            name,
            re.I,
        ):
            return True

    return False


def category_compatible(ingredient_family: str, row) -> bool:
    """Return True when a USDA row plausibly matches the ingredient family."""
    if ingredient_family == "generic":
        return True

    category = usda_food_category(row)
    name = usda_food_name(row)

    if ingredient_family == "whole_food":
        return _whole_food_allowed(row)

    blocked = _FAMILY_BLOCKED_USDA.get(ingredient_family)
    if blocked and blocked.search(category):
        return False

    if ingredient_family == "cereal" and not re.search(r"cereal|granola|oatmeal|muesli", category, re.I):
        if not re.search(r"cereal|granola|oatmeal|muesli", name, re.I):
            return False

    if ingredient_family == "alcohol" and not re.search(
        r"alcohol|beer|wine|liquor|spirit|cocktail", category + " " + name, re.I
    ):
        return False

    if ingredient_family == "spice" and not re.search(
        r"spice|herb|seasoning|pepper|salt", category + " " + name, re.I
    ):
        # Allow sr_legacy spice entries with empty-looking categories.
        if str(row.get("data_type") or "") in {"sr_legacy_food", "foundation_food", "survey_fndds_food"}:
            return True
        return False

    return True


def filter_compatible_candidates(candidates, ingredient_id: str, display_name: str):
    """Drop USDA rows whose food category conflicts with the ingredient."""
    family = infer_ingredient_family(ingredient_id, display_name)
    if family == "generic":
        return candidates

    if hasattr(candidates, "iloc"):
        mask = candidates.apply(lambda row: category_compatible(family, row), axis=1)
        return candidates[mask]

    return [row for row in candidates if category_compatible(family, row)]
