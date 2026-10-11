"""Food-family clusters for the Epicure vocabulary.

Clusters are a browsing aid for the catalog and for other diet projects.
They are assigned from the ingredient id, with a short override list for
names that would otherwise land in the wrong family.
"""

from __future__ import annotations

CLUSTERS: list[tuple[str, str]] = [
    ("fruit", "Fruit"),
    ("vegetables", "Vegetables"),
    ("mushrooms", "Mushrooms"),
    ("meat", "Meat"),
    ("seafood", "Seafood"),
    ("dairy_eggs", "Dairy & eggs"),
    ("grains", "Grains & bread"),
    ("legumes", "Legumes"),
    ("nuts_seeds", "Nuts & seeds"),
    ("fats_oils", "Fats & oils"),
    ("herbs_spices", "Herbs & spices"),
    ("sweets", "Sweets & baking"),
    ("drinks", "Drinks"),
    ("sauces", "Sauces & condiments"),
    ("prepared", "Prepared foods"),
    ("other", "Other"),
]

LABELS: dict[str, str] = dict(CLUSTERS)

# Checked before the keyword rules.
OVERRIDES: dict[str, str] = {
    "crab_apple": "fruit",
    "crab_mushroom": "mushrooms",
    "butterfish": "seafood",
    "butterfly_pea_flower": "vegetables",
    "almond_butter": "nuts_seeds",
    "peanut_butter": "nuts_seeds",
    "cashew_butter": "nuts_seeds",
    "sunflower_butter": "nuts_seeds",
    "sesame_paste": "nuts_seeds",
    "tahini": "nuts_seeds",
    "almond_paste": "sweets",
    "quince_paste": "sweets",
    "red_bean_paste": "sweets",
    "coconut": "nuts_seeds",
    "coconut_cream": "fats_oils",
    "coconut_sugar": "sweets",
    "cocoa_butter": "fats_oils",
    "cocoa_powder": "sweets",
    "cocoa_mass": "sweets",
    "cacao": "sweets",
    "cream_of_tartar": "sweets",
    "cream_of_rice": "grains",
    "cream_of_wheat": "grains",
    "cream_soda": "drinks",
    "ice_cream": "sweets",
    "whipped_topping": "dairy_eggs",
    "green_pea": "vegetables",
    "snow_pea": "vegetables",
    "sugar_snap_pea": "vegetables",
    "bean_sprout": "vegetables",
    "soybean_sprout": "vegetables",
    "alfalfa_sprout": "vegetables",
    "bell_pepper": "vegetables",
    "banana_pepper": "vegetables",
    "cherry_pepper": "vegetables",
    "cubanelle_pepper": "vegetables",
    "pepperoncini": "vegetables",
    "pimento": "vegetables",
    "poblano_pepper": "vegetables",
    "jalapeno": "vegetables",
    "shishito_pepper": "vegetables",
    "corn": "vegetables",
    "baby_corn": "vegetables",
    "sweet_corn": "vegetables",
    "popcorn": "grains",
    "cornmeal": "grains",
    "cornstarch": "grains",
    "corn_syrup": "sweets",
    "corn_oil": "fats_oils",
    "corn_flakes": "grains",
    "avocado": "fruit",
    "olive": "fruit",
    "black_olive": "fruit",
    "green_olive": "fruit",
    "tomato": "vegetables",
    "sun_dried_tomato": "vegetables",
    "eggplant": "vegetables",
    "potato": "vegetables",
    "sweet_potato": "vegetables",
    "purple_sweet_potato": "vegetables",
    "water": "drinks",
    "salt": "herbs_spices",
    "yeast": "other",
    "brewers_yeast": "other",
    "nutritional_yeast": "other",
    "red_yeast": "other",
    "advocaat": "drinks",
    "amaretto": "drinks",
    "anisette": "drinks",
    "benedictine": "drinks",
    "bloody_mary_mix": "drinks",
    "blue_curacao": "drinks",
    "campari": "drinks",
    "chartreuse": "drinks",
    "creme_de_cacao": "drinks",
    "creme_de_cassis": "drinks",
    "creme_de_menthe": "drinks",
    "creme_de_violette": "drinks",
    "cynar": "drinks",
    "drambuie": "drinks",
    "falernum": "drinks",
    "galliano": "drinks",
    "goldschlager": "drinks",
    "grand_marnier": "drinks",
    "herbsaint": "drinks",
    "hpnotiq": "drinks",
    "jagermeister": "drinks",
    "lillet": "drinks",
    "limoncello": "drinks",
    "liquor": "drinks",
    "mocha": "drinks",
    "pina_colada_mix": "drinks",
    "punch": "drinks",
    "sambuca": "drinks",
    "sauternes": "drinks",
    "sour_mix": "drinks",
    "triple_sec": "drinks",
    "vin_santo": "drinks",
    "biscotti": "sweets",
    "butterscotch": "sweets",
    "cajeta": "sweets",
    "candied_fruit": "sweets",
    "creme_anglaise": "sweets",
    "dulce_de_leche": "sweets",
    "fondant": "sweets",
    "frangipane": "sweets",
    "fructose": "sweets",
    "fruit_preserves": "sweets",
    "fruitcake": "sweets",
    "galette": "sweets",
    "gingerbread": "sweets",
    "gingersnap": "sweets",
    "grenadine": "drinks",
    "honeycomb": "sweets",
    "kaya": "sweets",
    "ladyfinger": "sweets",
    "maltose": "sweets",
    "marzipan": "sweets",
    "meringue": "sweets",
    "mizuame": "sweets",
    "red_hots": "sweets",
    "sprinkles": "sweets",
    "stevia": "sweets",
    "sweet_roll": "sweets",
    "swiss_roll": "sweets",
    "sherbet": "sweets",
    "nectar": "drinks",
    "bechamel": "sauces",
    "braising_liquid": "sauces",
    "caper": "sauces",
    "giardiniera": "sauces",
    "guacamole": "sauces",
    "marmite": "sauces",
    "mole": "sauces",
    "pan_drippings": "sauces",
    "sriracha": "sauces",
    "tamari": "sauces",
    "braunschweiger": "meat",
    "foie_gras": "meat",
    "gammon": "meat",
    "grouse": "meat",
    "guinea_hen": "meat",
    "hot_dog": "meat",
    "kielbasa": "meat",
    "linguica": "meat",
    "matambre": "meat",
    "offal": "meat",
    "pate": "meat",
    "squirrel": "meat",
    "turtle": "seafood",
    "unagi_kabayaki": "seafood",
    "bottarga": "seafood",
    "capelin": "seafood",
    "cobia": "seafood",
    "corvina": "seafood",
    "geoduck": "seafood",
    "gourami": "seafood",
    "hanpen": "seafood",
    "loach": "seafood",
    "mentaiko": "seafood",
    "miiuy_croaker": "seafood",
    "milt": "seafood",
    "mullet": "seafood",
    "pacific_saury": "seafood",
    "red_drum": "seafood",
    "sea_bream": "seafood",
    "tarako": "seafood",
    "whiting": "seafood",
    "yellow_croaker": "seafood",
    "caciocavallo": "dairy_eggs",
    "paneer": "dairy_eggs",
    "queso_anejo": "dairy_eggs",
    "queso_blanco": "dairy_eggs",
    "queso_fresco": "dairy_eggs",
    "ricotta_salata": "dairy_eggs",
    "bolillo": "grains",
    "bran": "grains",
    "cornbread": "grains",
    "crepe": "grains",
    "crispbread": "grains",
    "crouton": "grains",
    "crumb_crust": "sweets",
    "farina": "grains",
    "fideo": "grains",
    "flatbread": "grains",
    "grissini": "grains",
    "idli": "grains",
    "kinako": "grains",
    "masa_harina": "grains",
    "matzo": "grains",
    "mixed_grains": "grains",
    "pancake": "grains",
    "papad": "grains",
    "paratha": "grains",
    "ragi": "grains",
    "roti": "grains",
    "rusk": "grains",
    "senbei": "grains",
    "sushki": "grains",
    "tortellini": "grains",
    "waffle": "grains",
    "youtiao": "grains",
    "breadfruit": "fruit",
    "baked_beans": "legumes",
    "falafel": "legumes",
    "natto": "legumes",
    "okara_powder": "legumes",
    "refried_beans": "legumes",
    "soybean": "legumes",
    "mixed_seeds": "nuts_seeds",
    "nut": "nuts_seeds",
    "nut_butter": "nuts_seeds",
    "ajwain": "herbs_spices",
    "angelica_root": "herbs_spices",
    "baharat": "herbs_spices",
    "chamomile": "herbs_spices",
    "chervil": "herbs_spices",
    "chiltepin": "herbs_spices",
    "epazote": "herbs_spices",
    "gochugaru": "herbs_spices",
    "green_peppercorn": "herbs_spices",
    "hibiscus": "herbs_spices",
    "mace": "herbs_spices",
    "mixed_herbs": "herbs_spices",
    "peppermint": "herbs_spices",
    "safflower": "herbs_spices",
    "sambar_powder": "herbs_spices",
    "savory": "herbs_spices",
    "arrowhead": "vegetables",
    "celtuce": "vegetables",
    "dragon_whisker_vegetable": "vegetables",
    "dried_vegetable": "vegetables",
    "escarole": "vegetables",
    "frisee": "vegetables",
    "garland_chrysanthemum": "vegetables",
    "hair_vegetable": "vegetables",
    "hearts_of_palm": "vegetables",
    "lambs_quarters": "vegetables",
    "malanga": "vegetables",
    "mirepoix": "vegetables",
    "mixed_vegetable": "vegetables",
    "moringa": "vegetables",
    "pickled_vegetable": "vegetables",
    "vegetable_puree": "vegetables",
    "water_chestnut": "vegetables",
    "water_mimosa": "vegetables",
    "dragon_fruit": "fruit",
    "fruit": "fruit",
    "fruit_cocktail": "fruit",
    "fruit_puree": "fruit",
    "gac_fruit": "fruit",
    "gardenia_fruit": "fruit",
    "mac_mat_fruit": "fruit",
    "rosehip": "fruit",
    "sau_fruit": "fruit",
    "carob": "sweets",
    "baking_powder": "sweets",
    "baking_soda": "sweets",
    "gelatin": "other",
    "agar": "other",
    "pectin": "sweets",
    "bone": "meat",
    "caul_fat": "fats_oils",
    "chicken_fat": "fats_oils",
    "duck_fat": "fats_oils",
    "goose_fat": "fats_oils",
    "lard": "fats_oils",
    "tallow": "fats_oils",
    "shortening": "fats_oils",
    "ghee": "fats_oils",
    "copha": "fats_oils",
    "cooking_spray": "fats_oils",
    "margarine": "fats_oils",
    "butter": "dairy_eggs",
    "buttermilk": "dairy_eggs",
    "egg": "dairy_eggs",
    "century_egg": "dairy_eggs",
    "balut": "meat",
    "quorn": "prepared",
    "textured_soy_protein": "legumes",
    "soy_protein_isolate": "legumes",
    "wheat_gluten": "grains",
    "seitan": "grains",
    "tofu": "legumes",
    "tempeh": "legumes",
    "miso": "sauces",
    "doenjang": "sauces",
    "gochujang": "sauces",
    "chunjang": "sauces",
    "vegemite": "sauces",
    "honey": "sweets",
    "maple_syrup": "sweets",
    "molasses": "sweets",
    "black_treacle": "sweets",
    "agave_syrup": "sweets",
    "chocolate": "sweets",
    "white_chocolate": "sweets",
    "vanilla": "herbs_spices",
    "saffron": "herbs_spices",
    "rose_water": "herbs_spices",
    "orange_blossom_water": "herbs_spices",
    "seaweed": "seafood",
    "wakame": "seafood",
    "nori": "seafood",
    "kombu": "seafood",
    "aonori": "seafood",
    "spirulina": "seafood",
    "kelp": "seafood",
    "dulse": "seafood",
    "hijiki": "seafood",
    "amazake": "drinks",
    "chikuwa": "seafood",
    "oil": "fats_oils",
    "malt": "grains",
    "vegetable_extract": "sauces",
    "fern": "vegetables",
    "pokeweed": "vegetables",
    "borage": "herbs_spices",
    "licorice_root": "herbs_spices",
    "junket": "dairy_eggs",
    "ice": "drinks",
}

_DRINK_PARTS = (
    "wine", "beer", "tea", "coffee", "juice", "liqueur", "brandy", "whiskey",
    "whisky", "vodka", "rum", "sake", "cider", "soda", "champagne", "cognac",
    "bourbon", "absinthe", "lager", "stout", "kombucha", "horchata", "cola",
    "eggnog", "sangria", "prosecco", "sherry", "port", "vermouth", "gin",
    "tequila", "mezcal", "cachaca", "soju", "makgeolli", "amaro", "bitters",
    "smoothie", "milkshake", "tonic", "kefir",
)
_ANIMAL_MILK = {
    "milk", "buttermilk", "condensed_milk", "evaporated_milk", "powdered_milk",
    "dry_milk", "buffalo_milk", "goat_milk", "sheep_milk", "whole_milk",
    "skim_milk", "lowfat_milk",
}
_SAUCE_PARTS = (
    "sauce", "dressing", "vinegar", "relish", "chutney", "ketchup",
    "mayonnaise", "aioli", "pesto", "salsa", "sofrito", "tapenade",
    "tzatziki", "marinade", "gravy", "broth", "stock", "bouillon",
    "consomme", "paste", "wasabi", "horseradish", "mustard",
)
_SWEET_PARTS = (
    "sugar", "syrup", "candy", "cookie", "cake", "jam", "jelly", "pudding",
    "icing", "caramel", "toffee", "custard", "doughnut", "donut", "pastry",
    "frosting", "sprinkle", "marshmallow", "nougat", "fudge", "praline",
    "wafer", "biscuit", "brownie", "fudge", "pie_crust", "pie_filling",
    "chocolate", "cocoa", "honey", "molasses", "treacle",
)
_SPICE_PARTS = (
    "pepper", "chile", "chili", "spice", "seasoning", "cinnamon", "cumin",
    "paprika", "turmeric", "saffron", "nutmeg", "clove", "basil", "thyme",
    "oregano", "rosemary", "sage", "dill", "mint", "parsley", "cilantro",
    "coriander", "cardamom", "allspice", "anise", "bay_leaf", "tarragon",
    "marjoram", "chive", "lemongrass", "fenugreek", "sumac", "zaatar",
    "garam", "masala", "curry_powder", "herb", "wasabi", "ginger",
    "garlic_powder", "onion_powder",
)
_NUT_IDS = {
    "almond", "pecan", "walnut", "cashew", "pistachio", "hazelnut", "macadamia",
    "peanut", "pine_nut", "brazil_nut", "chestnut", "acorn", "bitter_almond",
    "black_sesame_seed", "sesame_seed", "pumpkin_seed", "sunflower_seed",
    "chia_seed", "poppy_seed", "flaxseed", "flax_seed", "hemp_seed",
    "watermelon_seed", "lotus_seed",
}
_NUT_PARTS = ("almond", "pecan", "walnut", "cashew", "pistachio", "hazelnut", "macadamia", "peanut", "chestnut")
_MEAT_PARTS = (
    "beef", "pork", "chicken", "lamb", "mutton", "duck", "turkey", "veal",
    "bacon", "sausage", "ham", "prosciutto", "salami", "pepperoni", "venison",
    "bison", "rabbit", "quail", "goose", "pheasant", "goat", "liver", "steak",
    "meat", "chorizo", "bratwurst", "andouille", "bresaola", "pancetta",
    "pastrami", "mortadella", "capon", "pigeon", "squab", "boar", "bear",
    "buffalo", "yak", "donkey", "horse", "frog", "sweetbread", "gizzard",
    "heart", "kidney", "tripe", "oxtail", "brisket", "rib",
)
_SEA_PARTS = (
    "fish", "shrimp", "prawn", "crab", "lobster", "clam", "mussel", "oyster",
    "scallop", "squid", "octopus", "anchovy", "salmon", "tuna", "cod", "trout",
    "sardine", "herring", "mackerel", "eel", "caviar", "roe", "abalone",
    "snail", "crawfish", "crayfish", "tilapia", "halibut", "snapper", "bass",
    "catfish", "swordfish", "shark", "anchovy", "bonito", "pollock", "haddock",
    "shellfish", "seafood", "whelk", "conch", "cockle", "periwinkle", "urchin",
    "jellyfish", "cuttlefish", "calamari", "smelt", "carp", "pike", "perch",
    "walleye", "whitefish", "bluefish", "pompano", "branzino", "dory", "turbot",
    "sturgeon", "monkfish", "grouper", "flounder", "sole", "anchovy",
)
_GRAIN_PARTS = (
    "rice", "wheat", "bread", "noodle", "pasta", "flour", "oat", "barley",
    "rye", "quinoa", "millet", "sorghum", "buckwheat", "couscous", "tortilla",
    "bagel", "pita", "naan", "bun", "dough", "semolina", "bulgur", "farro",
    "spelt", "amaranth", "teff", "cereal", "cracker", "dumpling", "ramen",
    "udon", "soba", "orzo", "polenta", "grits", "breadcrumb", "bread_crumb",
    "ciabatta", "brioche", "croissant", "muffin", "crumpet", "pretzel",
    "phyllo", "yufka", "wonton", "wrapper", "lasagne", "lasagna", "ravioli",
    "gnocchi", "spelt", "einkorn", "kamut", "freekeh", "cornmeal", "popcorn",
    "vermicelli", "macaroni", "spaghetti", "fettuccine", "linguine", "penne",
    "rigatoni", "fusilli", "rotini", "shell", "orzo", "rice_cake", "mochi",
    "tapioca", "sago", "arrowroot",
)
_LEGUME_PARTS = (
    "bean", "lentil", "chickpea", "soy", "tofu", "tempeh", "dal", "edamame",
    "hummus", "pea", "legume", "mung", "urad", "toor", "chana",
)
_FRUIT_TOKENS = {
    "apple", "apricot", "banana", "berry", "blackberry", "blueberry", "boysenberry",
    "cherry", "citrus", "cranberry", "currant", "date", "fig", "grape", "grapefruit",
    "lemon", "lime", "mango", "melon", "orange", "papaya", "peach", "pear", "plum",
    "raspberry", "strawberry", "watermelon", "pineapple", "kiwi", "kiwifruit",
    "lychee", "litchi", "guava", "pomegranate", "quince", "persimmon", "tangerine",
    "clementine", "mandarin", "nectarine", "cantaloupe", "honeydew", "plantain",
    "raisin", "prune", "avocado", "olive", "durian", "jackfruit", "rambutan",
    "starfruit", "acai", "acerola", "tamarind", "mulberry", "gooseberry",
    "elderberry", "lingonberry", "cloudberry", "huckleberry", "kumquat", "pomelo",
    "ugli", "satsuma", "yuzu", "sudachi", "kabosu", "calamansi", "passionfruit",
    "passion", "guava", "feijoa", "sapote", "mamey", "soursop", "cherimoya",
    "loquat", "medlar", "quince", "rhubarb", "coconut", "longan", "mangosteen",
    "salak", "dragonfruit", "pitaya", "physalis", "groundcherry", "aronia",
    "barberry", "sea_buckthorn", "jujube", "hawthorn",
}
_VEG_TOKENS = {
    "artichoke", "arugula", "asparagus", "beet", "beetroot", "bok", "broccoli",
    "cabbage", "carrot", "cauliflower", "celery", "chard", "chicory", "collard",
    "cucumber", "daikon", "endive", "fennel", "garlic", "kale", "leek", "lettuce",
    "onion", "shallot", "parsnip", "potato", "radish", "radicchio", "rutabaga",
    "spinach", "squash", "tomato", "turnip", "zucchini", "eggplant", "aubergine",
    "pumpkin", "okra", "kohlrabi", "celeriac", "salsify", "burdock", "taro",
    "yam", "cassava", "jicama", "lotus", "bamboo", "watercress", "cress",
    "mustard_green", "brussels", "sprout", "choy", "napa", "kimchi", "sauerkraut",
    "pickle", "gourd", "melon", "cucumber", "pepper", "chile", "scallion",
    "ramp", "fiddlehead", "nopal", "cactus", "seaweed", "kelp", "samphire",
    "purslane", "amaranth", "malabar", "chayote", "bitter", "kale", "broccolini",
    "romanesco", "corn", "hominy", "tomatillo", "pepperoncini", "jalapeno",
    "poblano", "serrano", "habanero", "chipotle", "ancho", "guajillo", "pasilla",
    "cabbage", "kale", "spinach", "sorrel", "dock", "nettle", "dandelion",
    "radish", "turnip", "swede", "celeriac", "fennel", "anise", "dill",
    "ginger", "turmeric", "galangal", "horseradish", "wasabi", "garlic",
    "onion", "leek", "shallot", "chive", "asparagus", "artichoke", "cardoon",
    "sunchoke", "jerusalem", "sweet_potato", "cassava", "manioc", "yuca",
    "plantain", "breadfruit", "jackfruit",
}


def _parts(ingredient_id: str) -> set[str]:
    return set(ingredient_id.split("_"))


def _has(ingredient_id: str, needles: tuple[str, ...] | set[str]) -> bool:
    """True when a needle is a whole underscore-delimited term, not a substring."""
    for needle in needles:
        if (
            ingredient_id == needle
            or ingredient_id.startswith(needle + "_")
            or ingredient_id.endswith("_" + needle)
            or f"_{needle}_" in ingredient_id
        ):
            return True
    return False


def assign_cluster(ingredient_id: str) -> str:
    """Return the cluster id for one ingredient id."""
    ingredient_id = ingredient_id.strip().lower()
    if ingredient_id in OVERRIDES:
        return OVERRIDES[ingredient_id]

    if "soup" in _parts(ingredient_id):
        return "prepared"

    if (
        "mushroom" in ingredient_id
        or ingredient_id in {"truffle", "white_truffle", "morel", "enoki", "shiitake", "porcini", "chanterelle", "matsutake", "maitake", "shimeji", "lions_mane"}
    ) and not ingredient_id.endswith("_sauce"):
        return "mushrooms"

    if ingredient_id in _ANIMAL_MILK or _has(ingredient_id, ("cheese",)):
        return "dairy_eggs"
    if ingredient_id.endswith("_milk") or ingredient_id.endswith("_water") or ingredient_id in {"water", "tonic_water", "coconut_water"}:
        if ingredient_id.endswith("_milk") and ingredient_id not in _ANIMAL_MILK:
            return "drinks"
        if ingredient_id.endswith("_water") or ingredient_id == "water":
            return "drinks"

    if _has(ingredient_id, _DRINK_PARTS) and "vinegar" not in ingredient_id:
        return "drinks"

    if ingredient_id.endswith("_oil") or ingredient_id in {"lard", "tallow", "shortening", "ghee", "margarine"} or ingredient_id.endswith("_fat"):
        return "fats_oils"

    if _has(ingredient_id, _SAUCE_PARTS) and ingredient_id not in {"mustard_seed", "mustard_green", "pickled_mustard_green"}:
        if ingredient_id.endswith("_seed"):
            pass
        else:
            return "sauces"

    if any(part in ingredient_id for part in ("soup", "stew", "pizza", "burger", "sandwich", "taco", "burrito", "lasagna", "lasagne", "ravioli", "casserole", "risotto", "sushi", "sashimi", "curry", "dumpling", "spring_roll", "egg_roll", "tamale", "empanada", "arepa", "quesadilla")):
        if ingredient_id.endswith("_powder") or ingredient_id.endswith("_seasoning") or "curry_powder" in ingredient_id:
            return "herbs_spices"
        return "prepared"

    if _has(ingredient_id, _SWEET_PARTS) or ingredient_id in {"sugar", "honey", "pie_crust", "pie_filling"}:
        return "sweets"

    if ingredient_id in _NUT_IDS or any(ingredient_id == nut or ingredient_id.startswith(nut + "_") for nut in _NUT_PARTS):
        if ingredient_id.endswith("_oil"):
            return "fats_oils"
        return "nuts_seeds"
    if ingredient_id.endswith("_seed") or ingredient_id.endswith("_nut"):
        return "nuts_seeds"

    if _has(ingredient_id, _SPICE_PARTS) or ingredient_id.endswith("_leaf") or ingredient_id.endswith("_leaves"):
        # Fresh alliums and leaves eaten as vegetables stay with vegetables.
        if ingredient_id in {"garlic", "onion", "ginger", "fennel", "dill", "basil", "mint", "parsley", "cilantro", "chive", "scallion", "shallot", "leek"}:
            return "vegetables" if ingredient_id in {"garlic", "onion", "ginger", "fennel", "scallion", "shallot", "leek"} else "herbs_spices"
        return "herbs_spices"

    if _has(ingredient_id, ("egg",)) and "eggplant" not in ingredient_id and "egg_roll" not in ingredient_id and "egg_noodle" not in ingredient_id:
        return "dairy_eggs"
    if any(token in ingredient_id for token in ("yogurt", "yoghurt", "cream", "whey", "curd", "kefir", "ghee")):
        if ingredient_id.startswith("cream_of_"):
            return "prepared"
        return "dairy_eggs"

    if _has(ingredient_id, _SEA_PARTS) or ingredient_id.endswith("fish"):
        return "seafood"
    if _has(ingredient_id, _MEAT_PARTS):
        return "meat"

    if _has(ingredient_id, _LEGUME_PARTS) and ingredient_id not in {"green_pea", "snow_pea", "sugar_snap_pea"}:
        return "legumes"
    if _has(ingredient_id, _GRAIN_PARTS):
        return "grains"

    tokens = _parts(ingredient_id)
    if tokens & _FRUIT_TOKENS or ingredient_id.endswith("berry") or ingredient_id.endswith("_melon"):
        if ingredient_id in {"watermelon_radish", "bitter_melon", "winter_melon"}:
            return "vegetables"
        return "fruit"
    if tokens & _VEG_TOKENS or "sprout" in ingredient_id or ingredient_id.endswith("_green") or ingredient_id.endswith("_greens"):
        return "vegetables"

    return "other"


def cluster_label(cluster_id: str) -> str:
    return LABELS.get(cluster_id, LABELS["other"])


def list_cluster_summaries(ingredient_ids: list[str]) -> list[dict]:
    """Return clusters that contain at least one ingredient, in catalog order."""
    counts: dict[str, int] = {cluster_id: 0 for cluster_id, _ in CLUSTERS}
    for ingredient_id in ingredient_ids:
        counts[assign_cluster(ingredient_id)] += 1
    return [
        {"id": cluster_id, "label": label, "count": counts[cluster_id]}
        for cluster_id, label in CLUSTERS
        if counts[cluster_id]
    ]


def ingredients_in_cluster(ingredient_ids: list[str], cluster_id: str) -> list[str]:
    """Return ingredient ids that belong to one cluster."""
    return [ingredient_id for ingredient_id in ingredient_ids if assign_cluster(ingredient_id) == cluster_id]
